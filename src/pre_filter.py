"""Split a GAND parquet into generatable and auto-reject rows.

- Auto-reject: multi-sentence sources and grammatically broken sources
  (optionally, via Haiku grammaticality check).
- Pre-gendered sources are flagged but NOT auto-rejected - the source is already
  disambiguated so the task becomes trivial, but downstream consumers may want
  to keep them for corpus coverage.

By default this module writes two parquets and one CSV and is purely
informational - generation still runs on the full file. Use `--apply-autoreject`
to actually skip the rejected rows.

Usage:
    python pre_filter.py data/train-00000-of-00001.parquet \
        --sample 100 --seed 42 --grammaticality --output-dir filtered
"""

import argparse
import os

import pandas as pd

from filter_sources import (
    detect_multi_sentence,
    detect_pre_gendered,
    submit_grammaticality_batch,
)


def pre_filter(df: pd.DataFrame, use_grammaticality: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["filter_multi_sentence"] = [detect_multi_sentence(s) for s in df["EN_source_sentence"]]
    df["filter_pre_gendered"] = [
        detect_pre_gendered(row["EN_source_sentence"], row["referent"])
        for _, row in df.iterrows()
    ]
    if use_grammaticality:
        import anthropic
        client = anthropic.Anthropic()
        scores = submit_grammaticality_batch(client, df)
        df["grammaticality_score"] = [scores.get(f"gram_{idx}") for idx in df.index]
        df["filter_low_grammaticality"] = [
            (s is not None and s <= 2) for s in df["grammaticality_score"]
        ]
    else:
        df["grammaticality_score"] = None
        df["filter_low_grammaticality"] = False

    df["filter_any"] = (
        df["filter_multi_sentence"] | df["filter_pre_gendered"] | df["filter_low_grammaticality"]
    )
    df["auto_reject"] = df["filter_multi_sentence"] | df["filter_low_grammaticality"]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output-dir", default="filtered")
    parser.add_argument("--grammaticality", action="store_true")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply-autoreject", action="store_true",
                        help="Actually split into generatable + rejected files. "
                             "Without this, just tags informational columns.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_parquet(args.input_file)
    if args.sample is not None:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)

    tagged = pre_filter(df, use_grammaticality=args.grammaticality)
    print(f"{len(tagged)} rows total")
    print(f"  multi_sentence:     {int(tagged['filter_multi_sentence'].sum())}")
    print(f"  pre_gendered:       {int(tagged['filter_pre_gendered'].sum())}")
    print(f"  low_grammaticality: {int(tagged['filter_low_grammaticality'].sum())}")
    print(f"  auto_reject:        {int(tagged['auto_reject'].sum())}")

    base = os.path.splitext(os.path.basename(args.input_file))[0]
    if args.apply_autoreject:
        clean = tagged[~tagged["auto_reject"]].copy()
        rejected = tagged[tagged["auto_reject"]].copy()
        clean_path = os.path.join(args.output_dir, f"{base}_generatable.parquet")
        reject_path = os.path.join(args.output_dir, f"{base}_autoreject.csv")
        clean.to_parquet(clean_path, index=False)
        rejected.to_csv(reject_path, index=False)
        print(f"\nGeneratable: {len(clean)} -> {clean_path}")
        print(f"Auto-reject:  {len(rejected)} -> {reject_path}")
    else:
        tagged_path = os.path.join(args.output_dir, f"{base}_tagged.parquet")
        tagged.to_parquet(tagged_path, index=False)
        print(f"\nAll rows tagged: {len(tagged)} -> {tagged_path}")
        print("(Use --apply-autoreject to actually split out rejected rows.)")


if __name__ == "__main__":
    main()
