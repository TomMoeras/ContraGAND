"""Local open-source-model gender classifier - HPC-friendly.

Mirrors the classify subcommand of classify_gender.py but uses Hugging Face
transformers (or vLLM if available) instead of the Anthropic Batch API.

Reuses the prompt constants and the long-form expansion from
classify_gender so all conditions (zero_shot / few_shot / few_shot_full /
few_shot_v3..v7) work identically.

The output schema is the same as the Claude path so the existing
`classify_gender.py evaluate` command can rank local models against the
Claude baselines.

Usage (transformers backend):
    python classify_gender_local.py \\
        --input data/test-00000-of-00001_with_variants.parquet \\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --condition few_shot_full \\
        --output results/classify/test_llama8b_few_shot_full.parquet \\
        --backend transformers --batch-size 8

Usage (vLLM backend):
    python classify_gender_local.py \\
        --input data/test-00000-of-00001_with_variants.parquet \\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --condition few_shot_full \\
        --output results/classify/test_llama8b_few_shot_full.parquet \\
        --backend vllm --max-num-seqs 32
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Iterable

import pandas as pd

from classify_gender import (
    SYSTEM_TASK,
    build_system_prompt,
    example_sentences_for,
    find_target_in_open_response,
    hygiene_check,
    is_open_condition,
    parse_json_response,
    to_long_form,
)


def chat_messages(system_prompt: str, sentence: str, referent: str,
                   open_task: bool = False,
                   demo_msgs: list[dict] | None = None) -> list[dict]:
    if open_task:
        user = f"Sentence: {sentence}"
    else:
        user = f"Sentence: {sentence}\nReferent: {referent}"
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    if demo_msgs:
        msgs.extend(demo_msgs)
    msgs.append({"role": "user", "content": user})
    return msgs


_CANNED_REASON = {
    "masculine": "the sentence contains a male-marking signal (title, pronoun, or gendered noun) bound to the referent",
    "feminine":  "the sentence contains a female-marking signal (title, pronoun, or gendered noun) bound to the referent",
    "ambiguous": "no gendered title, pronoun, or noun in the sentence reveals the referent's gender",
}


def build_demo_pool(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'masculine','feminine','ambiguous'} -> df with (sentence, referent)."""
    masc = train_df[["EN_masculine_sentence", "masc_referent"]].rename(
        columns={"EN_masculine_sentence": "sentence", "masc_referent": "referent"})
    fem = train_df[["EN_feminine_sentence", "fem_referent"]].rename(
        columns={"EN_feminine_sentence": "sentence", "fem_referent": "referent"})
    amb = train_df[["EN_source_sentence", "referent"]].rename(
        columns={"EN_source_sentence": "sentence"})
    pool = {}
    for label, df in [("masculine", masc), ("feminine", fem), ("ambiguous", amb)]:
        df = df.dropna().reset_index(drop=True)
        df = df[(df["sentence"].str.strip() != "") & (df["referent"].str.strip() != "")]
        pool[label] = df.reset_index(drop=True)
    return pool


def sample_demos(pool: dict[str, pd.DataFrame], rng, per_class: int = 2,
                 ambiguous_only: bool = False) -> list[dict]:
    """Sample demos for one test row.

    Default: per_class from each gold class (= 6 for per_class=2).
    ambiguous_only=True: per_class * len(pool) demos all from pool['ambiguous'],
        each gold-labeled 'ambiguous'. Ablation control - tests whether the
        contrastive class mix carries signal vs any 6-demo prompt.
    """
    demos: list[dict] = []
    if ambiguous_only:
        df = pool["ambiguous"]
        n = per_class * len(pool)
        idx = rng.choice(len(df), size=n, replace=False)
        for i in idx:
            row = df.iloc[int(i)]
            demos.append({"sentence": str(row["sentence"]),
                          "referent": str(row["referent"]),
                          "gold": "ambiguous"})
    else:
        for gold, df in pool.items():
            idx = rng.choice(len(df), size=per_class, replace=False)
            for i in idx:
                row = df.iloc[int(i)]
                demos.append({"sentence": str(row["sentence"]),
                              "referent": str(row["referent"]),
                              "gold": gold})
    rng.shuffle(demos)
    return demos


def _think_block(demo: dict, open_task: bool) -> str:
    """Short canned <think>...</think> trace per class - enables reasoning-model
    demos to keep their thinking pattern (Magistral copies what it sees)."""
    gold = demo["gold"]
    if open_task:
        body = (f"I read the sentence and look for role/occupation/relational nouns. "
                f"For \"{demo['referent']}\" I check the surrounding context for "
                f"gendered pronouns, titles, or morphology bound to it. "
                f"{_CANNED_REASON[gold]}, so the gender is {gold}.")
    else:
        body = (f"The referent is \"{demo['referent']}\". I check the sentence for "
                f"gendered pronouns, titles, or morphology bound to it. "
                f"{_CANNED_REASON[gold]}, so the gender is {gold}.")
    return f"<think>\n{body}\n</think>\n"


def demos_to_messages(demos: list[dict], open_task: bool,
                       include_think: bool = False) -> list[dict]:
    msgs: list[dict] = []
    for d in demos:
        if open_task:
            user = f"Sentence: {d['sentence']}"
            asst = json.dumps({
                "referents": [{"referent": d["referent"], "gender": d["gold"],
                               "confidence": 5,
                               "reasoning": _CANNED_REASON[d["gold"]]}]})
        else:
            user = f"Sentence: {d['sentence']}\nReferent: {d['referent']}"
            asst = json.dumps({"gender": d["gold"], "confidence": 5,
                               "reasoning": _CANNED_REASON[d["gold"]]})
        if include_think:
            asst = _think_block(d, open_task) + asst
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": asst})
    return msgs


def run_transformers(model_name: str, system_prompt: str, prompts_df: pd.DataFrame,
                     batch_size: int = 8, max_new_tokens: int = 200,
                     dtype: str = "auto", open_task: bool = False,
                     demo_msgs_list: list[list[dict]] | None = None) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[transformers] loading {model_name} (dtype={dtype}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch_dtype = torch.bfloat16 if dtype in ("bf16", "auto") else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"[transformers] model loaded on {device}", flush=True)

    outputs: list[str] = []
    n = len(prompts_df)
    for start in range(0, n, batch_size):
        chunk = prompts_df.iloc[start:start + batch_size]
        chats = []
        for offset, (_, row) in enumerate(chunk.iterrows()):
            row_idx = start + offset
            demos = demo_msgs_list[row_idx] if demo_msgs_list else None
            chats.append(chat_messages(system_prompt, row["sentence"], row["referent"],
                                       open_task=open_task, demo_msgs=demos))
        prompt_strs = [
            tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in chats
        ]
        enc = tok(prompt_strs, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tok.pad_token_id,
            )
        for i in range(len(chunk)):
            new_tokens = gen[i][enc["input_ids"].shape[1]:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            outputs.append(text)
        print(f"  done {min(start + batch_size, n)}/{n}", flush=True)
    return outputs


def run_vllm(model_name: str, system_prompt: str, prompts_df: pd.DataFrame,
             max_new_tokens: int = 200, max_num_seqs: int = 32,
             tensor_parallel_size: int = 1, dtype: str = "bfloat16",
             max_model_len: int = 8192,
             gpu_memory_utilization: float = 0.9,
             open_task: bool = False,
             demo_msgs_list: list[list[dict]] | None = None) -> list[str]:
    from vllm import LLM, SamplingParams

    print(f"[vllm] loading {model_name} (TP={tensor_parallel_size}, dtype={dtype}, "
          f"max_model_len={max_model_len}, gpu_mem_util={gpu_memory_utilization}) ...",
          flush=True)
    llm = LLM(
        model=model_name,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tok = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    chats = []
    for i, (_, row) in enumerate(prompts_df.iterrows()):
        demos = demo_msgs_list[i] if demo_msgs_list else None
        chats.append(chat_messages(system_prompt, row["sentence"], row["referent"],
                                   open_task=open_task, demo_msgs=demos))
    prompts = [
        tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in chats
    ]
    print(f"[vllm] generating {len(prompts)} responses ...", flush=True)
    results = llm.generate(prompts, sampling)
    return [r.outputs[0].text for r in results]


async def _post_chat_completion(session, base_url: str, model: str,
                                 messages: list, max_tokens: int,
                                 temperature: float, semaphore,
                                 top_p: float = 1.0,
                                 chat_template_kwargs: dict | None = None,
                                 max_retries: int = 4) -> str:
    import asyncio
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(
                    f"{base_url}/v1/chat/completions", json=payload,
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
                    body = await resp.json()
                    return body["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"  [WARN] request failed after {max_retries} retries: "
                      f"{type(exc).__name__}: {str(exc)[:200]}", flush=True)
                return ""
            await asyncio.sleep(2 ** attempt)
    return ""


def run_openai_api(model_name: str, system_prompt: str, prompts_df: pd.DataFrame,
                   base_url: str, max_new_tokens: int = 200,
                   max_parallel: int = 64,
                   chat_template_kwargs: dict | None = None,
                   open_task: bool = False,
                   temperature: float = 0.0,
                   top_p: float = 1.0,
                   demo_msgs_list: list[list[dict]] | None = None) -> list[str]:
    import asyncio

    import aiohttp

    chats = []
    for i, (_, row) in enumerate(prompts_df.iterrows()):
        demos = demo_msgs_list[i] if demo_msgs_list else None
        chats.append(chat_messages(system_prompt, row["sentence"], row["referent"],
                                   open_task=open_task, demo_msgs=demos))

    async def _wrapped(session, idx, messages, semaphore):
        text = await _post_chat_completion(session, base_url, model_name,
                                            messages, max_new_tokens, temperature,
                                            semaphore,
                                            top_p=top_p,
                                            chat_template_kwargs=chat_template_kwargs)
        return idx, text

    async def _all() -> list[str]:
        timeout = aiohttp.ClientTimeout(total=600)
        connector = aiohttp.TCPConnector(limit=max_parallel)
        semaphore = asyncio.Semaphore(max_parallel)
        results: list[str] = [""] * len(chats)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            coros = [_wrapped(session, i, c, semaphore) for i, c in enumerate(chats)]
            done = 0
            for fut in asyncio.as_completed(coros):
                i, text = await fut
                results[i] = text
                done += 1
                if done % 100 == 0 or done == len(chats):
                    print(f"  [openai-api] {done}/{len(chats)} done", flush=True)
        return results

    print(f"[openai-api] {len(chats)} prompts -> {base_url}", flush=True)
    return asyncio.run(_all())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--condition", required=True,
                        choices=["zero_shot", "few_shot", "few_shot_full",
                                 "few_shot_v3", "few_shot_v4", "few_shot_v5",
                                 "few_shot_v6", "few_shot_v7",
                                 "zero_shot_identify"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", default="transformers",
                        choices=["transformers", "vllm", "openai-api"])
    parser.add_argument("--batch-size", type=int, default=8,
                        help="transformers backend only")
    parser.add_argument("--max-num-seqs", type=int, default=32,
                        help="vllm backend only")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--dtype", default="bfloat16",
                        help="bfloat16 / float16 / auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="vllm backend only")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="vllm backend only")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="vllm backend only")
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="openai-api backend only (vLLM server endpoint)")
    parser.add_argument("--max-parallel", type=int, default=64,
                        help="openai-api backend only")
    parser.add_argument("--chat-template-kwargs", default=None,
                        help="JSON string passed to vLLM as chat_template_kwargs "
                             "(e.g. '{\"enable_thinking\": false}' for Qwen3)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (Magistral wants 0.7).")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Sampling top_p (Magistral wants 0.95).")
    parser.add_argument("--system-prompt-file", default=None,
                        help="Path to a file whose contents replace the built-in "
                             "system prompt (used for reasoning models like Magistral).")
    parser.add_argument("--fewshot-train-input", default=None,
                        help="Path to train parquet. When set, sample per-class demos "
                             "from this split per test example and prepend as "
                             "user/assistant chat turns.")
    parser.add_argument("--fewshot-per-class", type=int, default=2,
                        help="Demos sampled per class (masc/fem/ambig); total = 3*this.")
    parser.add_argument("--fewshot-seed", type=int, default=42,
                        help="RNG seed for per-row demo sampling (deterministic).")
    parser.add_argument("--fewshot-include-think", action="store_true",
                        help="Wrap each demo assistant message with a <think>...</think> "
                             "block (use for Magistral so it keeps reasoning at test time).")
    parser.add_argument("--fewshot-ambiguous-only", action="store_true",
                        help="Sample all 6 demos from the ambiguous (source) pool "
                             "instead of 2/2/2 across masc/fem/amb. Ablation control "
                             "for whether the contrastive class mix carries signal.")
    parser.add_argument("--lora-name", default=None,
                        help="When set, this string is sent as the `model` field in the "
                             "openai-api JSON payload instead of --model. Use the LoRA "
                             "alias passed to vLLM via --lora-modules name=path.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap to first N long-form examples (smoke testing)")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")
    long_df = to_long_form(df)
    print(f"Expanded to {len(long_df)} examples ({args.condition}).")

    hygiene_check(long_df, args.condition)

    if args.limit is not None:
        long_df = long_df.iloc[:args.limit].copy()
        print(f"--limit {args.limit}: truncated to {len(long_df)} examples.")

    if args.system_prompt_file:
        with open(args.system_prompt_file, "r") as f:
            system_prompt = f.read()
        print(f"[system-prompt] loaded {len(system_prompt)} chars from {args.system_prompt_file}", flush=True)
    else:
        system_prompt = build_system_prompt(args.condition)
    open_task = is_open_condition(args.condition)

    demo_msgs_list = None
    per_row_demos: list[list[dict]] | None = None
    if args.fewshot_train_input:
        import numpy as np
        train_df = pd.read_parquet(args.fewshot_train_input)
        pool = build_demo_pool(train_df)
        sizes = {k: len(v) for k, v in pool.items()}
        print(f"[fewshot] loaded {len(train_df)} train rows; demo pool sizes: {sizes}", flush=True)
        rng = np.random.default_rng(args.fewshot_seed)
        per_row_demos = [sample_demos(pool, rng,
                                       per_class=args.fewshot_per_class,
                                       ambiguous_only=args.fewshot_ambiguous_only)
                         for _ in range(len(long_df))]
        demo_msgs_list = [demos_to_messages(d, open_task=open_task,
                                              include_think=args.fewshot_include_think)
                          for d in per_row_demos]
        print(f"[fewshot] pre-computed {len(demo_msgs_list)} per-row demo sets "
              f"(per_class={args.fewshot_per_class}, seed={args.fewshot_seed}, "
              f"include_think={args.fewshot_include_think}, "
              f"ambiguous_only={args.fewshot_ambiguous_only})", flush=True)

    if args.backend == "transformers":
        raw_responses = run_transformers(
            args.model, system_prompt, long_df,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            open_task=open_task,
            demo_msgs_list=demo_msgs_list,
        )
    elif args.backend == "vllm":
        raw_responses = run_vllm(
            args.model, system_prompt, long_df,
            max_new_tokens=args.max_new_tokens,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            open_task=open_task,
            demo_msgs_list=demo_msgs_list,
        )
    else:
        ctk = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
        served_model = args.lora_name or args.model
        if args.lora_name:
            print(f"[lora] sending model={served_model!r} (LoRA alias) to openai-api", flush=True)
        raw_responses = run_openai_api(
            served_model, system_prompt, long_df,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            max_parallel=args.max_parallel,
            chat_template_kwargs=ctk,
            open_task=open_task,
            temperature=args.temperature,
            top_p=args.top_p,
            demo_msgs_list=demo_msgs_list,
        )

    predictions, confidences, reasonings = [], [], []
    if open_task:
        for text, target in zip(raw_responses, long_df["referent"]):
            data = parse_json_response(text)
            if not data:
                predictions.append("parse_error")
                confidences.append(None)
                reasonings.append(None)
                continue
            entry = find_target_in_open_response(data, target)
            if not entry:
                predictions.append("not_found")
                confidences.append(None)
                reasonings.append(None)
                continue
            gender = entry.get("gender")
            if gender in {"masculine", "feminine", "ambiguous"}:
                predictions.append(gender)
            else:
                predictions.append("parse_error")
            conf = entry.get("confidence")
            confidences.append(int(conf) if isinstance(conf, (int, float)) else None)
            reasonings.append(entry.get("reasoning"))
    else:
        for text in raw_responses:
            data = parse_json_response(text)
            gender = data.get("gender")
            if gender in {"masculine", "feminine", "ambiguous"}:
                predictions.append(gender)
            else:
                predictions.append("parse_error")
            conf = data.get("confidence")
            confidences.append(int(conf) if isinstance(conf, (int, float)) else None)
            reasonings.append(data.get("reasoning"))

    long_df = long_df.copy()
    long_df["condition"] = args.condition
    long_df["predicted_label"] = predictions
    long_df["confidence"] = confidences
    long_df["reasoning"] = reasonings
    long_df["correct"] = long_df["predicted_label"] == long_df["expected_label"]
    long_df["raw_response"] = raw_responses
    if per_row_demos is not None:
        long_df["demos_used"] = [json.dumps(d) for d in per_row_demos]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    long_df.to_parquet(args.output, index=False)
    long_df.to_csv(args.output.replace(".parquet", ".csv"), index=False)

    total = len(long_df)
    correct = int(long_df["correct"].sum())
    parse_err = int((long_df["predicted_label"] == "parse_error").sum())
    print(f"\nClassified {total} examples. Correct: {correct}/{total} ({correct/total:.1%})")
    print(f"Parse errors: {parse_err}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
