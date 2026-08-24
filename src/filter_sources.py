"""Pre-filter GAND sources for data quality issues before generation.

Flags three classes of problems:
  1. Multi-sentence sources (violates GAND single-sentence rule).
  2. Pre-gendered sources (gendered pronoun within 5 tokens of the referent).
  3. Grammatically broken sources (Haiku grammaticality score <= 2).

Thresholds (tuned on the first 100 reviewed rows):
  - Multi-sentence: sentence contains '. ' or '! ' or '? ' followed by a capital
    letter AND token count > 30. Lower token sentences with periods (e.g.
    "I like it. So do you.") are kept since GAND allowed them.
  - Pre-gendered: a gendered pronoun ('he','she','his','her','him','himself',
    'herself') occurs within 5 whitespace-separated tokens of the referent
    (or any case-insensitive match of the referent).
  - Grammaticality: Haiku-4-5 batch rates the sentence 1-5; flag <= 2.

Usage:
    python filter_sources.py data/train-00000-of-00001.parquet \
        --output-dir filtered/ --grammaticality
"""

import argparse
import json
import os
import re
import time

import anthropic
import pandas as pd

GENDERED_PRONOUNS = {"he", "she", "his", "her", "him", "himself", "herself", "hers"}


def detect_multi_sentence(sentence: str) -> bool:
    if not isinstance(sentence, str):
        return False
    if len(sentence.split()) <= 30:
        return False
    return bool(re.search(r"[.!?]\s+[A-Z]", sentence))


def detect_pre_gendered(sentence: str, referent: str, window: int = 5) -> bool:
    if not isinstance(sentence, str) or not isinstance(referent, str):
        return False
    tokens = re.findall(r"[\w']+", sentence.lower())
    ref_lower = referent.lower()
    for i, tok in enumerate(tokens):
        if tok == ref_lower or tok.startswith(ref_lower):
            lo = max(0, i - window)
            hi = min(len(tokens), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                if tokens[j] in GENDERED_PRONOUNS:
                    return True
    return False


def submit_grammaticality_batch(client, df, model="claude-haiku-4-5-20251001"):
    requests = []
    for idx, row in df.iterrows():
        sentence = row["EN_source_sentence"]
        prompt = f"""Rate this English sentence for grammaticality on a 1-5 scale (5=fully grammatical, 1=severely broken). Consider only grammar - not punctuation style, capitalization, or occasional typos that don't affect parseability.

Sentence: {sentence}

Respond with ONLY a JSON object:
{{"score": N}}"""
        requests.append({
            "custom_id": f"gram_{idx}",
            "params": {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
            },
        })
    batch = client.messages.batches.create(requests=requests)
    print(f"Grammaticality batch submitted: {batch.id}")
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        total = counts.processing + done
        print(f"  {batch.processing_status}: {done}/{total}", end="\r", flush=True)
        if batch.processing_status == "ended":
            print()
            break
        time.sleep(30)

    scores = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type != "succeeded":
            continue
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
            data = json.loads(text)
            scores[result.custom_id] = data.get("score")
        except (json.JSONDecodeError, ValueError):
            pass
    return scores


def main():
    parser = argparse.ArgumentParser(description="Filter GAND sources for data quality")
    parser.add_argument("input_file", help="Path to parquet input")
    parser.add_argument("--output-dir", default="filtered")
    parser.add_argument("--grammaticality", action="store_true",
                        help="Run Haiku grammaticality check (slower, uses API)")
    parser.add_argument("--grammaticality-threshold", type=int, default=2)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.input_file)
    if args.sample is not None:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)

    print(f"Loaded {len(df)} rows")

    df = df.copy()
    df["flag_multi_sentence"] = [detect_multi_sentence(s) for s in df["EN_source_sentence"]]
    df["flag_pre_gendered"] = [
        detect_pre_gendered(row["EN_source_sentence"], row["referent"])
        for _, row in df.iterrows()
    ]
    print(f"Multi-sentence: {int(df['flag_multi_sentence'].sum())}")
    print(f"Pre-gendered:   {int(df['flag_pre_gendered'].sum())}")

    if args.grammaticality:
        client = anthropic.Anthropic()
        scores = submit_grammaticality_batch(client, df)
        df["grammaticality_score"] = [
            scores.get(f"gram_{idx}") for idx in df.index
        ]
        df["flag_low_grammaticality"] = [
            (s is not None and s <= args.grammaticality_threshold)
            for s in df["grammaticality_score"]
        ]
        print(f"Low grammaticality: {int(df['flag_low_grammaticality'].sum())}")
    else:
        df["grammaticality_score"] = None
        df["flag_low_grammaticality"] = False

    df["flagged"] = df["flag_multi_sentence"] | df["flag_pre_gendered"] | df["flag_low_grammaticality"]

    flagged = df[df["flagged"]].copy()
    clean = df[~df["flagged"]].copy()

    reasons = []
    for _, row in flagged.iterrows():
        parts = []
        if row["flag_multi_sentence"]:
            parts.append("multi_sentence")
        if row["flag_pre_gendered"]:
            parts.append("pre_gendered")
        if row["flag_low_grammaticality"]:
            parts.append(f"low_grammaticality(score={row['grammaticality_score']})")
        reasons.append(", ".join(parts))
    flagged["flag_reasons"] = reasons

    base = os.path.splitext(os.path.basename(args.input_file))[0]
    clean_path = os.path.join(args.output_dir, f"{base}_filtered.parquet")
    skipped_path = os.path.join(args.output_dir, f"{base}_skipped.csv")
    clean.drop(columns=["flagged"]).to_parquet(clean_path, index=False)
    flagged.drop(columns=["flagged"]).to_csv(skipped_path, index=False)

    print(f"\nClean rows:   {len(clean)}  -> {clean_path}")
    print(f"Skipped rows: {len(flagged)} -> {skipped_path}")


if __name__ == "__main__":
    main()
