#!/usr/bin/env python3
"""Build the 5 figure source CSVs in figures/data/ for scripts/make_figures.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/eval"
OUT = ROOT / "figures" / "data"
OUT.mkdir(parents=True, exist_ok=True)

QWEN  = "Qwen3.6-27B"
GEMMA = "Gemma-4-31B-it"

DIRS = {
    (QWEN,  False): RES / "qwen3_6_27b_step2",
    (QWEN,  True):  RES / "qwen3_6_27b_thinking",
    (GEMMA, False): RES / "gemma_4_31b_step2",
    (GEMMA, True):  RES / "gemma_4_31b_thinking",
}


def headline() -> None:
    rows = []
    for (model, thinking), d in DIRS.items():
        df = pd.read_csv(d / "all_conditions_summary.csv").set_index("metric")
        rows.append({"model": model, "thinking": thinking, "task": "closed",
                     "accuracy":     float(df.loc["overall", "zero_shot"]),
                     "macro_f1":     float(df.loc["macro_f1", "zero_shot"]),
                     "ambig_recall": float(df.loc["ambiguity_recall", "zero_shot"]),
                     "masc_recall":  float(df.loc["masculine_recall", "zero_shot"]),
                     "fem_recall":   float(df.loc["feminine_recall", "zero_shot"])})
        rows.append({"model": model, "thinking": thinking, "task": "open",
                     "accuracy":     float(df.loc["overall", "zero_shot_identify"]),
                     "macro_f1":     float(df.loc["macro_f1", "zero_shot_identify"]),
                     "ambig_recall": float(df.loc["ambiguity_recall", "zero_shot_identify"]),
                     "masc_recall":  float(df.loc["masculine_recall", "zero_shot_identify"]),
                     "fem_recall":   float(df.loc["feminine_recall", "zero_shot_identify"])})
    pd.DataFrame(rows).to_csv(OUT / "01_headline.csv", index=False)
    print(f"  wrote 01_headline.csv ({len(rows)} rows)")


def per_strategy() -> None:
    qw = pd.read_csv(DIRS[(QWEN,  False)] / "by_strategy_zero_shot.csv")
    gm = pd.read_csv(DIRS[(GEMMA, False)] / "by_strategy_zero_shot.csv")
    qw = qw[(qw["variant"] == "masculine") & (qw["strategy"] != "(source rows - ambiguous)")]
    gm = gm[(gm["variant"] == "masculine") & (gm["strategy"] != "(source rows - ambiguous)")]
    out = qw[["strategy", "n", "accuracy"]].rename(
        columns={"n": "n", "accuracy": "qwen_closed_zs"}
    ).merge(
        gm[["strategy", "accuracy"]].rename(columns={"accuracy": "gemma_closed_zs"}),
        on="strategy", how="outer"
    )
    out.to_csv(OUT / "02_per_strategy.csv", index=False)
    print(f"  wrote 02_per_strategy.csv ({len(out)} rows)")


def open_confusion() -> None:
    rows = []
    for model, dirname in [(QWEN,  "qwen3_6_27b_step2"),
                            (GEMMA, "gemma_4_31b_step2")]:
        cm = pd.read_csv(RES / dirname / "confusion_zero_shot_identify.csv", index_col=0)
        for true_label, row in cm.iterrows():
            total = row.sum()
            for pred_label, count in row.items():
                rows.append({"model": model, "true_label": true_label,
                             "pred_label": pred_label, "count": int(count),
                             "fraction": float(count) / float(total) if total else 0.0})
    pd.DataFrame(rows).to_csv(OUT / "03_open_confusion.csv", index=False)
    print(f"  wrote 03_open_confusion.csv ({len(rows)} rows)")


def thinking_delta() -> None:
    rows = []
    for model in [QWEN, GEMMA]:
        nt = pd.read_csv(DIRS[(model, False)] / "all_conditions_summary.csv").set_index("metric")
        th = pd.read_csv(DIRS[(model, True)]  / "all_conditions_summary.csv").set_index("metric")
        for task, col in [("closed", "zero_shot"), ("open", "zero_shot_identify")]:
            base = float(nt.loc["overall", col])
            think = float(th.loc["overall", col])
            rows.append({"model": model, "task": task,
                         "no_thinking": base, "thinking": think,
                         "delta": think - base})
    pd.DataFrame(rows).to_csv(OUT / "04_thinking_delta.csv", index=False)
    print(f"  wrote 04_thinking_delta.csv ({len(rows)} rows)")


def stereotype_overlap() -> None:
    sources = [
        (QWEN,  "closed", DIRS[(QWEN,  False)] / "ambiguity_bias_zero_shot.csv"),
        (QWEN,  "open",   DIRS[(QWEN,  False)] / "ambiguity_bias_zero_shot_identify.csv"),
        (GEMMA, "closed", DIRS[(GEMMA, False)] / "ambiguity_bias_zero_shot.csv"),
        (GEMMA, "open",   DIRS[(GEMMA, False)] / "ambiguity_bias_zero_shot_identify.csv"),
    ]
    referents_by_cell = {}
    rows = []
    for model, task, path in sources:
        df = pd.read_csv(path)
        df["referent"] = df["referent"].astype(str).str.lower()
        referents_by_cell[(model, task)] = set(df["referent"])
        for _, r in df.iterrows():
            rows.append({"model": model, "task": task,
                         "referent": r["referent"],
                         "n_confident_guesses": int(r["n_confident_guesses"]),
                         "guessed_masculine": int(r["guessed_masculine"]),
                         "guessed_feminine": int(r["guessed_feminine"]),
                         "male_share": float(r["male_share"])})
    df = pd.DataFrame(rows)
    overlap_open   = referents_by_cell[(QWEN, "open")]   & referents_by_cell[(GEMMA, "open")]
    overlap_closed = referents_by_cell[(QWEN, "closed")] & referents_by_cell[(GEMMA, "closed")]

    def cross(row):
        return row["referent"] in (overlap_open if row["task"] == "open" else overlap_closed)
    df["cross_model_overlap"] = df.apply(cross, axis=1)
    df.to_csv(OUT / "05_stereotype_overlap.csv", index=False)
    print(f"  wrote 05_stereotype_overlap.csv ({len(df)} rows)")
    print(f"    closed overlap: {sorted(overlap_closed)}")
    print(f"    open overlap:   {sorted(overlap_open)}")


def main() -> None:
    headline()
    per_strategy()
    open_confusion()
    thinking_delta()
    stereotype_overlap()


if __name__ == "__main__":
    main()
