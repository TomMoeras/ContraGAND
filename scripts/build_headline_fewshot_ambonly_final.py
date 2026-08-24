#!/usr/bin/env python3
"""4-row headline for the ambiguous-only fewshot ablation on the human-reviewed test split.

Companion to scripts/build_headline_fewshot_final.py - same models, same
test set, but every test row's 6 demos are sampled from the ambiguous
(source) pool only (gold-labeled `ambiguous`) instead of 2/2/2 across
masc/fem/amb. Side-by-side compare with `headline_fewshot_final.csv` to
isolate the contribution of contrastive class mix.

Reads `prf_zero_shot.csv` and `prf_zero_shot_identify.csv` from each of:
  results/eval/{qwen3_6_27b,gemma_4_31b,gpt_oss_20b,magistral_small_2506}_ambonly_fewshot_thinking_final/

Output: results/eval/headline_fewshot_ambonly_final.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/eval"
OUT = RES / "headline_fewshot_ambonly_final.csv"

MODELS = [
    ("Qwen3.6-27B",          "qwen3_6_27b_ambonly_fewshot_thinking_final"),
    ("Gemma-4-31B-it",       "gemma_4_31b_ambonly_fewshot_thinking_final"),
    ("gpt-oss-20b",          "gpt_oss_20b_ambonly_fewshot_thinking_final"),
    ("Magistral-Small-2506", "magistral_small_2506_ambonly_fewshot_thinking_final"),
]


def _read_prf(path: Path) -> dict | None:
    if not path.exists():
        return None
    prf = pd.read_csv(path).set_index("label")
    classes = ["masculine", "feminine", "ambiguous"]
    total_tp = prf.loc[classes, "tp"].sum()
    total_n  = total_tp + prf.loc[classes, "fn"].sum()
    return {
        "accuracy":     float(total_tp) / float(total_n) if total_n else None,
        "macro_f1":     float(prf.loc["macro", "f1"]),
        "masc_recall":  float(prf.loc["masculine", "recall"]),
        "fem_recall":   float(prf.loc["feminine", "recall"]),
        "ambig_recall": float(prf.loc["ambiguous", "recall"]),
    }


def main() -> None:
    rows = []
    for label, dirname in MODELS:
        closed = _read_prf(RES / dirname / "prf_zero_shot.csv")
        opened = _read_prf(RES / dirname / "prf_zero_shot_identify.csv")
        if closed is None and opened is None:
            print(f"  [skip] {label}: no prf CSVs in {dirname}")
            continue
        row = {"model": label}
        for prefix, src in [("closed", closed), ("open", opened)]:
            if src is None:
                row.update({f"{prefix}_{k}": None for k in
                            ["accuracy","macro_f1","masc_recall","fem_recall","ambig_recall"]})
            else:
                row.update({f"{prefix}_{k}": v for k, v in src.items()})
        if closed and opened:
            row["closed_minus_open_acc"] = round(closed["accuracy"] - opened["accuracy"], 4)
        rows.append(row)

    if not rows:
        print("No models had results. Run the evaluator first.")
        return

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"  wrote {OUT.relative_to(ROOT)} ({len(out_df)} rows)")
    print()
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", "{:.3f}".format):
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
