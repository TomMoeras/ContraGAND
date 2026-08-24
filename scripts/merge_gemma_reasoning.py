#!/usr/bin/env python3
"""Merge Gemma's closed-task reasoning into a wide-form GAND parquet.

Inputs:
  --original  path to a wide-form parquet from data/
              (e.g. final-train-...with_variants.parquet)
  --gemma     path to the long-form Gemma predictions parquet
              (one row per (source, variant); columns: row_id, variant,
              reasoning, raw_response, predicted_label, correct, ...)
  --output    path to write the new parquet (e.g. ...with_variants_with_reasoning.parquet)

Output is the original parquet with 6 new columns:
  gemma_reasoning_{source,masculine,feminine}
  gemma_raw_response_{source,masculine,feminine}

Each new column contains Gemma's reasoning for the corresponding sentence
variant (source = ambiguous, masculine = masc-rewrite, feminine = fem-rewrite).

The reasoning column is the parsed JSON `reasoning` field. raw_response
contains the full chat-template output including the <think>/thought block,
useful for distillation / training data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VARIANTS = ("source", "masculine", "feminine")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--gemma", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    orig = pd.read_parquet(args.original)
    gem = pd.read_parquet(args.gemma)
    print(f"original: {orig.shape}  ({Path(args.original).name})")
    print(f"gemma:    {gem.shape}    ({Path(args.gemma).name})")

    if "row_id" not in gem.columns or "variant" not in gem.columns:
        raise SystemExit("gemma parquet must have row_id and variant columns")

    expected_long = len(orig) * 3
    if len(gem) != expected_long:
        print(f"  [WARN] gemma rows ({len(gem)}) != orig*3 ({expected_long})")

    # Pivot reasoning + raw_response into wide form keyed by row_id
    pieces = []
    for var in VARIANTS:
        sub = gem[gem["variant"] == var][["row_id", "reasoning", "raw_response",
                                            "predicted_label", "correct"]]
        sub = sub.rename(columns={
            "reasoning":      f"gemma_reasoning_{var}",
            "raw_response":   f"gemma_raw_response_{var}",
            "predicted_label": f"gemma_predicted_{var}",
            "correct":         f"gemma_correct_{var}",
        }).set_index("row_id")
        pieces.append(sub)
    wide = pd.concat(pieces, axis=1)

    # Join with original. Original index is the source-row position;
    # to_long_form assigns row_id = positional index 0..N-1, so a positional
    # left-join works.
    if not (wide.index.is_monotonic_increasing
            and wide.index.min() == 0 and wide.index.max() == len(orig) - 1):
        print(f"  [WARN] gemma row_id span [{wide.index.min()}, {wide.index.max()}], "
              f"orig has {len(orig)} rows")

    out = orig.copy().reset_index(drop=True)
    out = out.join(wide, how="left")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    out.to_csv(args.output.replace(".parquet", ".csv"), index=False)
    print(f"wrote {args.output}  shape={out.shape}")
    print()
    print("per-variant Gemma accuracy:")
    for var in VARIANTS:
        col = f"gemma_correct_{var}"
        if col in out.columns:
            n = int(out[col].notna().sum())
            tp = int(out[col].fillna(False).sum())
            print(f"  {var:10s}  {tp}/{n} = {tp/n:.3f}" if n else f"  {var}: empty")


if __name__ == "__main__":
    main()
