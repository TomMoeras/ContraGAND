#!/usr/bin/env python3
"""Headline leaderboard for the closed-task QLoRA fine-tuning sweep.

Reads `prf_zero_shot.csv` from each `results/eval/<student>_ft_<COND>_thinking_final/`
and writes a single `results/eval/headline_finetune_final.csv` with one row per
(student × condition).

Conditions:
  DAUTH  - authentic-only (~3,918 source rows, all gold=ambiguous; no synthetic masc/fem)
  DA     - full 2/2/2 contrastive synthetic data, canned reasoning
  DB     - DA + Gemma reasoning (filtered to gemma_correct=True)
  DC     - DA + full Gemma raw_response (CoT distillation)

Notes (from this round):
  - Qwen3.5-4B fine-tuned cleanly (4/4 conditions completed; DC timed out at 3.83/5
    epochs, used checkpoint-700 for inference).
  - Gemma-4-E4B-it: blocked by `Gemma4ClippableLinear` PEFT incompatibility.
  - Ministral-3-3B-Instruct-2512: blocked because it ships pre-quantized as
    FineGrainedFP8 (incompatible with QLoRA's BitsAndBytesConfig).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/eval"
OUT = RES / "headline_finetune_final.csv"

MODELS = [
    # (display label, results-dir suffix)
    ("Qwen3.5-4B / FT D-AUTH",  "qwen3_5_4b_ft_DAUTH_thinking_final"),
    ("Qwen3.5-4B / FT D-A",     "qwen3_5_4b_ft_DA_thinking_final"),
    # Gemma-4-E4B-it / Ministral-3-3B blocked by per-arch issues this round.
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
        m = _read_prf(RES / dirname / "prf_zero_shot.csv")
        if m is None:
            print(f"  [skip] {label}: no prf CSV in {dirname}")
            continue
        rows.append({"model": label, **m})

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
