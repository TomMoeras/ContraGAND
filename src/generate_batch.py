"""Batch generation of contrastive gender pairs using the Anthropic Batch API.

Two-phase pipeline:
  Phase 1: Submit generation batch (Sonnet) -> collect results
  Phase 2: Submit evaluation batch (Opus) -> collect results, flag failures
  Phase 3 (optional): Re-submit failures with feedback -> re-evaluate

Usage:
    # Phase 1: Generate
    python generate_batch.py generate --split train --sample 100 --seed 1

    # Phase 2: Evaluate
    python generate_batch.py evaluate generated/batch_train_sample100_seed1.parquet

    # Phase 3: Retry failures
    python generate_batch.py retry generated/batch_train_sample100_seed1_evaluated.parquet

    # Check batch status
    python generate_batch.py status <batch_id>
"""

import argparse
import json
import os
import time

import anthropic
import pandas as pd

from generate import load_reference_material, build_system_prompt
from grounding import analyze, load_gendered_pairs_lookup
from rotators import SurnameRotator, RelationalPairRotator
from validators import validate_row


def create_generation_requests(df, system_prompt, model, use_grounding=True):
    """Build batch requests for generation.

    When use_grounding=True, runs symbolic pre-analysis on each row and
    injects a [HINTS] block with recommended strategy, referent pair,
    surname assignment, and relational pair assignment.
    """
    requests = []
    hints_applied = []
    if use_grounding:
        lookup = load_gendered_pairs_lookup()
        surname_rot = SurnameRotator()
        rel_rot = RelationalPairRotator()
    else:
        lookup = None
        surname_rot = None
        rel_rot = None

    for idx, row in df.iterrows():
        referent = row["referent"]
        sentence = row["EN_source_sentence"]
        user_msg = f"Referent: {referent}\nOriginal sentence: {sentence}"
        hint_summary = None
        if use_grounding:
            hint = analyze({"referent": referent, "EN_source_sentence": sentence}, lookup)
            extra_parts = []
            if hint.recommended_strategy:
                extra_parts.append(hint.render().strip())
                hint_summary = hint.recommended_strategy
            assigned_surname = surname_rot.pick()
            assigned_rel = rel_rot.pick()
            extra_parts.append(
                f"ASSIGNED_SURNAME_IF_TITLE: {assigned_surname}\n"
                f"ASSIGNED_RELATIONAL_PAIR_IF_RELATIONAL: {assigned_rel[0]}/{assigned_rel[1]}"
            )
            if extra_parts:
                user_msg = user_msg + "\n\n" + "\n\n".join(extra_parts)
        hints_applied.append(hint_summary)

        requests.append({
            "custom_id": f"gen_{idx}",
            "params": {
                "model": model,
                "max_tokens": 1024,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": user_msg,
                    }
                ],
            },
        })
    return requests, hints_applied


EVAL_SYSTEM_PROMPT = """You evaluate contrastive gender sentence pairs (masculine + feminine variants of an originally gender-ambiguous sentence) for a fixed referent.

Score 1-5 each (5=best):
1. REFERENT_DISAMBIGUATED (5=the referent is clearly gendered, 1=something else was changed instead).
   - NOTE: If the referent is the addressee ("you" or a direct address), adding sir/ma'am or a title DOES disambiguate the referent and should score 5.
   - If a pronoun is inserted with an appositive clarifying it refers to the referent (e.g., "providing him, the publisher, a sense of rhythm" for referent "publisher"), that DOES disambiguate - score 5.
2. CORRECTNESS (5=grammatically correct, 1=grammatical errors)
3. NATURALNESS (5=reads naturally, 1=awkward or forced)
4. NO_EXTRA_SENTENCE (5=no new sentence added after a period, 1=a full new sentence was added. Short clauses via commas or dashes are FINE.)
5. CONSISTENCY (5=symmetric balanced edits, 1=asymmetric)

Also classify the strategy used with ONE of these labels:
- pronoun_swap: existing pronouns swapped (they -> he/she)
- pronoun_insertion: a pronoun was inserted (his/her crew, or "providing him, the X,")
- context_modifier: a nearby neutral word replaced with a gendered one (the person -> the man)
- relational_insertion: a gendered family relation inserted as appositive (his brother, her sister, his uncle, her aunt, etc.)
- appositive_dash: "- a man -" or "- a woman -" dash-delimited appositive inserted
- adjective: "male"/"female" adjective added before the referent
- referent_swap: the referent word itself replaced (waiter -> waitress)
- title_insertion: "Mr./Mrs." + a name inserted
- sir_maam: "sir" or "ma'am" added as a vocative (only valid when the referent is the addressee)
- combination: two or more of the above
- no_change: the variant is identical to the original (rare)

Also flag POS_CHANGE: true if the edit turned the referent from a noun into something else (e.g., "the local" where "local" was a noun becomes "the local man" where "local" is now an adjective). Otherwise false.

Also flag REJECT: true if the sentence is unfixable (source is broken, multi-sentence, or referent can't be naturally disambiguated).

If the pair has any problem (score below 5 on any criterion, or fixable issue), provide a concrete SUGGESTION for a better edit. The suggestion should:
- Pick a specific strategy from: pronoun_swap, pronoun_insertion, context_modifier, relational_insertion, appositive_dash, adjective, referent_swap, title_insertion, sir_maam
- Give concrete masculine and feminine sentence candidates
- Keep it brief: one or two sentences describing the fix and a sample pair
If no problems or the pair is already perfect, leave suggestion as empty string.

Respond with ONLY JSON:
{"referent_disambiguated": N, "correctness": N, "naturalness": N, "no_extra_sentence": N, "consistency": N, "strategy_label": "pronoun_swap|pronoun_insertion|context_modifier|relational_insertion|appositive_dash|adjective|referent_swap|title_insertion|sir_maam|combination|no_change", "pos_change": false, "reject": false, "issues": "brief problems or empty string", "suggestion": "concrete fix with proposed strategy and sample M/F sentences, or empty string"}"""


def create_eval_requests(df, eval_model, only_unevaluated=False):
    """Build batch requests for evaluation.

    The static rubric lives in the system message with cache_control so the
    Batch API can reuse it across rows. Per-row data goes in the user message.
    """
    requests = []
    for idx, row in df.iterrows():
        if pd.isna(row.get("EN_masculine_sentence")) or pd.isna(row.get("EN_feminine_sentence")):
            continue
        if only_unevaluated:
            accepted = row.get("accepted")
            if pd.notna(accepted) and bool(accepted):
                continue
            reject = row.get("eval_reject", False)
            if pd.notna(reject) and bool(reject):
                continue

        user_msg = (
            f'Referent: "{row["referent"]}"\n'
            f"Original: {row['EN_source_sentence']}\n"
            f"Masculine: {row['EN_masculine_sentence']}\n"
            f"Feminine: {row['EN_feminine_sentence']}"
        )

        requests.append({
            "custom_id": f"eval_{idx}",
            "params": {
                "model": eval_model,
                "max_tokens": 256,
                "system": [
                    {
                        "type": "text",
                        "text": EVAL_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": user_msg}],
            },
        })
    return requests


def create_retry_requests(df, system_prompt, model, max_retries=5):
    """Build batch requests for retrying failed rows with feedback."""
    requests = []
    for idx, row in df.iterrows():
        if row.get("accepted", True):
            continue
        if bool(row.get("eval_reject", False)):
            continue
        rc = row.get("retry_count", 0)
        retry_count = 0 if pd.isna(rc) else int(rc)
        if retry_count >= max_retries:
            continue

        orig = row["EN_source_sentence"]
        masc = row.get("EN_masculine_sentence", "")
        fem = row.get("EN_feminine_sentence", "")
        issues = row.get("eval_issues", "")
        suggestion = row.get("eval_suggestion", "")

        feedback_parts = []
        if pd.notna(masc) and pd.notna(fem) and masc == fem:
            feedback_parts.append("The masculine and feminine variants are IDENTICAL. They MUST differ.")
        if pd.notna(masc) and masc == orig and pd.notna(fem) and fem == orig:
            feedback_parts.append("Both variants are UNCHANGED. You MUST make edits.")
        ref_disam = row.get("eval_referent_disambiguated", 5)
        if pd.notna(ref_disam) and ref_disam <= 3:
            feedback_parts.append(f"CRITICAL: The referent '{row['referent']}' was NOT disambiguated. Make sure the gender of '{row['referent']}' specifically is clear.")
        no_extra = row.get("eval_no_extra_sentence", 5)
        if pd.notna(no_extra) and no_extra <= 3:
            feedback_parts.append("CRITICAL: You added a second sentence. All edits must be WITHIN the original sentence.")
        if bool(row.get("eval_pos_change", False)):
            feedback_parts.append(f"CRITICAL: You turned the referent '{row['referent']}' from a noun into an adjective. Keep it as a noun and try a different strategy.")
        correctness = row.get("eval_correctness", 5)
        if pd.notna(correctness) and correctness < 4:
            feedback_parts.append(f"Correctness issue: {issues}")
        naturalness = row.get("eval_naturalness", 5)
        if pd.notna(naturalness) and naturalness < 4:
            feedback_parts.append(f"Naturalness issue: {issues}")
        sym_failures = row.get("symbolic_check_failures", "")
        if pd.notna(sym_failures) and str(sym_failures).strip():
            feedback_parts.append(f"SYMBOLIC CHECK FAILURES: {sym_failures}")
        if pd.notna(suggestion) and str(suggestion).strip():
            feedback_parts.append(f"SUGGESTED FIX FROM EVALUATOR: {suggestion}")
        if not feedback_parts:
            feedback_parts.append(f"Previous attempt had issues: {issues}. Try a different strategy.")

        feedback = " ".join(feedback_parts)

        requests.append({
            "custom_id": f"retry_{idx}",
            "params": {
                "model": model,
                "max_tokens": 1024,
                "system": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Referent: {row['referent']}\n"
                            f"Original sentence: {orig}\n\n"
                            f"FEEDBACK ON PREVIOUS ATTEMPT: {feedback}\n\n"
                            f"Try again with a better strategy."
                        ),
                    }
                ],
            },
        })
    return requests


def submit_batch(client, requests, description=""):
    """Submit a batch and return the batch object."""
    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.processing_status}")
    return batch


def wait_for_batch(client, batch_id, poll_interval=30):
    """Poll until batch completes."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        print(
            f"\r  Status: {batch.processing_status} | "
            f"{done}/{total} done "
            f"(ok={counts.succeeded} err={counts.errored} "
            f"cancel={counts.canceled} exp={counts.expired})",
            end="",
            flush=True,
        )
        if batch.processing_status == "ended":
            print()
            return batch
        time.sleep(poll_interval)


def collect_results(client, batch_id):
    """Collect results from a completed batch. Returns dict of custom_id -> result."""
    results = {}
    for result in client.messages.batches.results(batch_id):
        results[result.custom_id] = result
    return results


def parse_gen_result(result):
    """Parse a generation result into a dict with all fields."""
    if result.result.type != "succeeded":
        return {}

    text = result.result.message.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        start = text.index("{")
        depth = 0
        for j, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start : j + 1]
                    break
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None, None, None


def parse_eval_result(result):
    """Parse an evaluation result into scores dict."""
    if result.result.type != "succeeded":
        return {}

    text = result.result.message.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        start = text.index("{")
        depth = 0
        for j, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start : j + 1]
                    break
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}


def cmd_generate(args):
    """Phase 1: Generate contrastive pairs via batch."""
    client = anthropic.Anthropic()
    lexicon, lists = load_reference_material()
    system_prompt = build_system_prompt(lexicon, lists)

    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]

    for split in splits:
        df = pd.read_parquet(f"data/{split}-00000-of-00001.parquet")
        if args.sample is not None:
            df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
        print(f"\n=== {split}: {len(df)} rows ===")

        requests, hints_applied = create_generation_requests(
            df, system_prompt, args.model, use_grounding=getattr(args, "use_grounding", True),
        )
        if getattr(args, "use_grounding", True):
            hint_counts = {}
            for h in hints_applied:
                if h:
                    hint_counts[h] = hint_counts.get(h, 0) + 1
            print(f"Symbolic hints: {hint_counts}")
        batch = submit_batch(client, requests)

        if not args.no_wait:
            batch = wait_for_batch(client, batch.id)
            results = collect_results(client, batch.id)

            mascs, fems, strats, masc_refs, fem_refs = [], [], [], [], []
            for idx in df.index:
                cid = f"gen_{idx}"
                if cid in results:
                    data = parse_gen_result(results[cid])
                    mascs.append(data.get("masculine_sentence"))
                    fems.append(data.get("feminine_sentence"))
                    strats.append(data.get("strategy", "unknown"))
                    masc_refs.append(data.get("masc_referent", df.loc[idx, "referent"]))
                    fem_refs.append(data.get("fem_referent", df.loc[idx, "referent"]))
                else:
                    mascs.append(None)
                    fems.append(None)
                    strats.append(None)
                    masc_refs.append(None)
                    fem_refs.append(None)

            df = df.copy()
            df["EN_masculine_sentence"] = mascs
            df["EN_feminine_sentence"] = fems
            df["disambiguation_strategy"] = strats
            df["masc_referent"] = masc_refs
            df["fem_referent"] = fem_refs
            df["recommended_strategy"] = [h if h else "" for h in hints_applied]

            sample_str = f"_sample{args.sample}" if args.sample else ""
            seed_str = f"_seed{args.seed}" if args.seed != 42 else ""
            out_path = os.path.join(
                args.output_dir,
                f"batch_{split}{sample_str}{seed_str}.parquet",
            )
            os.makedirs(args.output_dir, exist_ok=True)
            df.to_parquet(out_path, index=False)

            success = sum(1 for m in mascs if m is not None)
            print(f"Success: {success}/{len(df)}")
            print(f"Saved to {out_path}")
        else:
            print(f"Batch submitted: {batch.id}")
            print("Use 'status' command to check progress, then collect results manually.")


def cmd_evaluate(args):
    """Phase 2: Evaluate generated pairs via batch."""
    client = anthropic.Anthropic()

    df = pd.read_parquet(args.input_file)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    requests = create_eval_requests(df, args.eval_model, only_unevaluated=getattr(args, "only_unevaluated", False))
    if not requests:
        print("No rows to evaluate (all already accepted/rejected).")
        out_path = args.input_file.replace(".parquet", "_evaluated.parquet")
        df = _reorder_review_columns(df)
        df.to_parquet(out_path, index=False)
        df.to_csv(out_path.replace(".parquet", ".csv"), index=False)
        return
    batch = submit_batch(client, requests)

    if not args.no_wait:
        batch = wait_for_batch(client, batch.id)
        results = collect_results(client, batch.id)

        eval_keys = ["referent_disambiguated", "correctness", "naturalness",
                     "no_extra_sentence", "consistency"]
        for idx in df.index:
            cid = f"eval_{idx}"
            if cid in results:
                scores = parse_eval_result(results[cid])
                for k in eval_keys:
                    df.loc[idx, f"eval_{k}"] = scores.get(k)
                df.loc[idx, "eval_strategy_label"] = scores.get("strategy_label", "unknown")
                df.loc[idx, "eval_pos_change"] = bool(scores.get("pos_change", False))
                df.loc[idx, "eval_reject"] = bool(scores.get("reject", False))
                df.loc[idx, "eval_issues"] = scores.get("issues", "")
                df.loc[idx, "eval_suggestion"] = scores.get("suggestion", "")

                vals = [scores.get(k, 0) or 0 for k in eval_keys]
                avg = sum(vals) / len(vals)
                ref_disam = scores.get("referent_disambiguated", 5)
                no_extra = scores.get("no_extra_sentence", 5)
                rejected = bool(scores.get("reject", False))
                pos_change = bool(scores.get("pos_change", False))
                if rejected:
                    df.loc[idx, "accepted"] = False
                elif pos_change:
                    df.loc[idx, "accepted"] = False
                elif ref_disam is not None and ref_disam <= 2:
                    df.loc[idx, "accepted"] = False
                elif no_extra is not None and no_extra <= 2:
                    df.loc[idx, "accepted"] = False
                else:
                    df.loc[idx, "accepted"] = avg >= args.threshold

        if getattr(args, "use_symbolic", True):
            sym_override = 0
            for idx in df.index:
                if pd.isna(df.loc[idx].get("accepted")):
                    continue
                if not bool(df.loc[idx]["accepted"]):
                    continue
                result = validate_row(df.loc[idx])
                df.loc[idx, "symbolic_checks_passed"] = bool(result["passed"])
                df.loc[idx, "symbolic_check_failures"] = "; ".join(result["messages"]) if result["messages"] else ""
                if not result["passed"]:
                    df.loc[idx, "accepted"] = False
                    existing_issues = str(df.loc[idx].get("eval_issues", "") or "")
                    sym_msg = " | ".join(result["messages"])
                    df.loc[idx, "eval_issues"] = (existing_issues + " [SYMBOLIC] " + sym_msg).strip()
                    sym_override += 1
            if sym_override > 0:
                print(f"  Symbolic validators overrode {sym_override} previously-accepted rows.")

        out_path = args.input_file.replace(".parquet", "_evaluated.parquet")
        df = _reorder_review_columns(df)
        df.to_parquet(out_path, index=False)
        df.to_csv(out_path.replace(".parquet", ".csv"), index=False)

        _print_batch_summary(df, eval_keys, out_path)

        rejected_df = df[df.get("eval_reject", False) == True] if "eval_reject" in df.columns else df.head(0)
        if len(rejected_df) > 0:
            reject_path = out_path.replace(".parquet", "_rejected.csv")
            rejected_df.to_csv(reject_path, index=False)
            print(f"  {len(rejected_df)} rejected rows saved to {reject_path}")
    else:
        print(f"Batch submitted: {batch.id}")


def _reorder_review_columns(df):
    preferred = [
        "referent", "EN_source_sentence",
        "masc_referent", "EN_masculine_sentence",
        "fem_referent", "EN_feminine_sentence",
        "disambiguation_strategy", "eval_strategy_label", "recommended_strategy",
        "eval_issues", "eval_suggestion",
        "eval_referent_disambiguated", "eval_correctness", "eval_naturalness",
        "eval_no_extra_sentence", "eval_consistency",
        "eval_pos_change", "eval_reject",
        "symbolic_checks_passed", "symbolic_check_failures",
        "accepted", "retry_count",
    ]
    existing = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in existing]
    return df[existing + rest]


def _print_batch_summary(df, eval_keys, out_path):
    total = len(df)
    accepted = int(df["accepted"].sum()) if "accepted" in df.columns else 0
    print(f"\n=== BATCH SUMMARY ===")
    print(f"Accepted: {accepted}/{total} ({accepted/total:.0%})")
    for k in eval_keys:
        col = f"eval_{k}"
        if col in df.columns:
            print(f"  {k}: {df[col].mean():.2f}")
    if "eval_strategy_label" in df.columns:
        print(f"\nStrategy distribution (evaluator's labels):")
        dist = df[df["accepted"] == True]["eval_strategy_label"].value_counts()
        for label, count in dist.items():
            print(f"  {label}: {count} ({count/max(accepted,1):.0%})")
        appositive_count = int(dist.get("appositive_dash", 0))
        if accepted > 0 and appositive_count / accepted > 0.15:
            print(f"\n  !! SOFT WARNING: appositive_dash is {appositive_count}/{accepted} ({appositive_count/accepted:.0%}) of accepted rows - over 15% threshold.")
            examples = df[(df["accepted"] == True) & (df["eval_strategy_label"] == "appositive_dash")].head(3)
            for _, row in examples.iterrows():
                print(f"     e.g. [{row['referent']}] {str(row['EN_masculine_sentence'])[:90]}")
    if "eval_reject" in df.columns:
        reject_count = int(df["eval_reject"].sum()) if df["eval_reject"].dtype == bool else int((df["eval_reject"] == True).sum())
        if reject_count > 0:
            print(f"\n  !! {reject_count} rows flagged as unfixable (eval_reject=True)")
    print(f"\nSaved to {out_path}")


def cmd_retry(args):
    """Phase 3: Retry failed rows via batch."""
    client = anthropic.Anthropic()
    lexicon, lists = load_reference_material()
    system_prompt = build_system_prompt(lexicon, lists)

    df = pd.read_parquet(args.input_file)
    if "retry_count" not in df.columns:
        df["retry_count"] = 0
    failures = df[df["accepted"] == False]
    if "eval_reject" in df.columns:
        rejected = failures[failures["eval_reject"] == True]
        capped = failures[failures["retry_count"].fillna(0) >= args.max_retries]
        retryable = failures[(failures["eval_reject"] != True) & (failures["retry_count"].fillna(0) < args.max_retries)]
        print(f"Loaded {len(df)} rows, {len(failures)} failures ({len(rejected)} rejected/unfixable, {len(capped)} at retry cap, {len(retryable)} retryable)")
    else:
        retryable = failures[failures["retry_count"].fillna(0) < args.max_retries]
        print(f"Loaded {len(df)} rows, {len(failures)} failures, {len(retryable)} retryable")

    if len(retryable) == 0:
        print("No retryable failures!")
        out_path = args.input_file.replace("_evaluated", "_retried")
        df.to_parquet(out_path, index=False)
        return

    requests = create_retry_requests(retryable, system_prompt, args.model, max_retries=args.max_retries)
    batch = submit_batch(client, requests)

    if not args.no_wait:
        batch = wait_for_batch(client, batch.id)
        results = collect_results(client, batch.id)

        for idx in retryable.index:
            cid = f"retry_{idx}"
            if cid in results:
                data = parse_gen_result(results[cid])
                if data.get("masculine_sentence") is not None:
                    df.loc[idx, "EN_masculine_sentence"] = data["masculine_sentence"]
                    df.loc[idx, "EN_feminine_sentence"] = data["feminine_sentence"]
                    df.loc[idx, "disambiguation_strategy"] = data.get("strategy", "unknown")
                    df.loc[idx, "masc_referent"] = data.get("masc_referent", df.loc[idx, "referent"])
                    df.loc[idx, "fem_referent"] = data.get("fem_referent", df.loc[idx, "referent"])

        numeric_eval_cols = [
            "eval_referent_disambiguated", "eval_correctness", "eval_naturalness",
            "eval_no_extra_sentence", "eval_consistency",
        ]
        for col in numeric_eval_cols:
            if col in df.columns:
                df[col] = df[col].astype("Float64")
        if "accepted" in df.columns:
            df["accepted"] = df["accepted"].astype("boolean")
        if "eval_reject" in df.columns:
            df["eval_reject"] = df["eval_reject"].astype("boolean")
        if "eval_pos_change" in df.columns:
            df["eval_pos_change"] = df["eval_pos_change"].astype("boolean")

        for idx in retryable.index:
            for col in numeric_eval_cols + ["eval_issues", "accepted", "eval_strategy_label", "eval_reject", "eval_pos_change", "eval_suggestion"]:
                if col in df.columns:
                    df.loc[idx, col] = pd.NA
            prev_rc = df.loc[idx, "retry_count"]
            prev_rc = 0 if pd.isna(prev_rc) else int(prev_rc)
            df.loc[idx, "retry_count"] = prev_rc + 1

        out_path = args.input_file.replace("_evaluated", "_retried")
        df.to_parquet(out_path, index=False)
        print(f"Saved retried results to {out_path}")
        print(f"Now run: python generate_batch.py evaluate {out_path}")
    else:
        print(f"Batch submitted: {batch.id}")


def cmd_status(args):
    """Check batch status."""
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(args.batch_id)
    counts = batch.request_counts
    print(f"Batch: {batch.id}")
    print(f"Status: {batch.processing_status}")
    print(f"Created: {batch.created_at}")
    print(f"Counts: succeeded={counts.succeeded} errored={counts.errored} "
          f"processing={counts.processing} canceled={counts.canceled} expired={counts.expired}")
    if batch.results_url:
        print(f"Results URL: {batch.results_url}")


def main():
    parser = argparse.ArgumentParser(description="Batch contrastive gender generation")
    sub = parser.add_subparsers(dest="command", required=True)

    # Generate
    gen = sub.add_parser("generate", help="Phase 1: Generate contrastive pairs")
    gen.add_argument("--split", default="train", choices=["train", "validation", "test", "all"])
    gen.add_argument("--model", default="claude-sonnet-4-6")
    gen.add_argument("--sample", type=int, default=None)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--output-dir", default="generated")
    gen.add_argument("--no-grounding", dest="use_grounding", action="store_false",
                     help="Disable symbolic grounding hints (pure v15-style generation)")
    gen.add_argument("--no-wait", action="store_true", help="Submit and return immediately")
    gen.set_defaults(use_grounding=True)

    # Evaluate
    ev = sub.add_parser("evaluate", help="Phase 2: Evaluate generated pairs")
    ev.add_argument("input_file")
    ev.add_argument("--eval-model", default="claude-opus-4-6")
    ev.add_argument("--threshold", type=float, default=4.0)
    ev.add_argument("--only-unevaluated", action="store_true",
                    help="Skip rows that are already accepted or rejected")
    ev.add_argument("--no-symbolic-checks", dest="use_symbolic", action="store_false",
                    help="Disable symbolic validators after LLM eval")
    ev.add_argument("--no-wait", action="store_true")
    ev.set_defaults(use_symbolic=True)

    # Retry
    rt = sub.add_parser("retry", help="Phase 3: Retry failed rows")
    rt.add_argument("input_file")
    rt.add_argument("--model", default="claude-sonnet-4-6")
    rt.add_argument("--max-retries", type=int, default=5)
    rt.add_argument("--no-wait", action="store_true")

    # Autopilot
    ap = sub.add_parser("autopilot", help="Run evaluate+retry loop up to N rounds")
    ap.add_argument("input_file")
    ap.add_argument("--model", default="claude-opus-4-6")
    ap.add_argument("--eval-model", default="claude-opus-4-6")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--max-retries", type=int, default=5)

    # Status
    st = sub.add_parser("status", help="Check batch status")
    st.add_argument("batch_id")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "retry":
        cmd_retry(args)
    elif args.command == "autopilot":
        cmd_autopilot(args)
    elif args.command == "status":
        cmd_status(args)


def cmd_autopilot(args):
    """Loop evaluate -> retry -> evaluate for up to N rounds.

    Starts from a generated parquet, runs the full eval+retry loop
    until no more retryable rows or max_retries reached.
    """
    import types

    current_file = args.input_file
    for round_num in range(args.max_retries + 1):
        print(f"\n### AUTOPILOT ROUND {round_num}: evaluate {current_file}")
        eval_args = types.SimpleNamespace(
            input_file=current_file,
            eval_model=args.eval_model,
            threshold=args.threshold,
            only_unevaluated=True,
            no_wait=False,
        )
        cmd_evaluate(eval_args)

        evaluated = current_file.replace(".parquet", "_evaluated.parquet")
        df = pd.read_parquet(evaluated)
        failures = df[df["accepted"] == False]
        retryable = failures
        if "eval_reject" in df.columns:
            retryable = failures[failures["eval_reject"] != True]
        if "retry_count" in df.columns:
            retryable = retryable[retryable["retry_count"].fillna(0) < args.max_retries]

        if len(retryable) == 0:
            print(f"\n### AUTOPILOT DONE after round {round_num}: no more retryable rows")
            break

        if round_num >= args.max_retries:
            print(f"\n### AUTOPILOT DONE: reached max {args.max_retries} retry rounds")
            break

        print(f"\n### AUTOPILOT ROUND {round_num}: retrying {len(retryable)} rows")
        retry_args = types.SimpleNamespace(
            input_file=evaluated,
            model=args.model,
            max_retries=args.max_retries,
            no_wait=False,
        )
        cmd_retry(retry_args)
        current_file = evaluated.replace("_evaluated", "_retried")


if __name__ == "__main__":
    main()
