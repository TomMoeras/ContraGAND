"""One-shot script: human-review stats + cleaned test parquet.

Reads the reviewer's CSV, normalises the `eval_human` column, computes
ratings counts (overall and per disambiguation strategy), writes a
paper-ready markdown report and a CSV companion, and saves a filtered
test set with bad-rated rows removed.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / (
    "human_reviewed_test-00000-of-00001_with_variants - "
    "test-00000-of-00001_with_variants.csv"
)
OUT_PARQUET = ROOT / "data" / "final_reviewed_test-00000-of-00001_with_variants.parquet"
OUT_CSV = ROOT / "data" / "final_reviewed_test-00000-of-00001_with_variants.csv"
REPORT_MD = ROOT / "results" / "human_review_stats.md"
REPORT_CSV = ROOT / "results" / "human_review_stats.csv"

VALID_RATINGS = {"good", "meh", "bad"}


def normalise_rating(value) -> str:
    if pd.isna(value):
        return "(unrated)"
    s = str(value).strip().lower()
    if s == "":
        return "(unrated)"
    return s if s in VALID_RATINGS else f"(other: {s})"


def has_great_marker(value) -> bool:
    if pd.isna(value):
        return False
    return bool(re.search(r"\bgreat\b", str(value), flags=re.IGNORECASE))


def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"ERROR: input not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    n_total = len(df)

    df["rating"] = df["eval_human"].map(normalise_rating)
    df["has_great"] = df["eval_human_explained"].map(has_great_marker)

    counts = df["rating"].value_counts().to_dict()
    good = int(counts.get("good", 0))
    meh = int(counts.get("meh", 0))
    bad = int(counts.get("bad", 0))
    other = sum(v for k, v in counts.items() if k not in {"good", "meh", "bad"})
    ok = good + meh
    n_great = int(df["has_great"].sum())

    rating_rows = [
        {"rating": "good", "count": good, "percent": round(100 * good / n_total, 1)},
        {"rating": "meh", "count": meh, "percent": round(100 * meh / n_total, 1)},
        {"rating": "bad", "count": bad, "percent": round(100 * bad / n_total, 1)},
        {"rating": "ok (good+meh)", "count": ok, "percent": round(100 * ok / n_total, 1)},
    ]
    if other > 0:
        rating_rows.append({"rating": "other/unrated", "count": other,
                             "percent": round(100 * other / n_total, 1)})

    df["strategy"] = df["eval_strategy_label"].fillna("(missing)")
    by_strat = (
        df.groupby(["strategy", "rating"]).size().unstack(fill_value=0)
    )
    for col in ["good", "meh", "bad"]:
        if col not in by_strat.columns:
            by_strat[col] = 0
    by_strat["n"] = by_strat.sum(axis=1)
    by_strat["pct_ok"] = (
        100 * (by_strat["good"] + by_strat["meh"]) / by_strat["n"]
    ).round(1)
    by_strat = by_strat.sort_values("n", ascending=False)
    by_strat = by_strat[["n", "good", "meh", "bad", "pct_ok"]]

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# Human review of v16 test set",
        "",
        f"Total rows: **{n_total}**",
        "",
        "| rating | count | percent |",
        "|---|---:|---:|",
    ]
    for r in rating_rows:
        md_lines.append(f"| {r['rating']} | {r['count']} | {r['percent']}% |")
    md_lines += [
        "",
        f'"great" comments in `eval_human_explained`: **{n_great}** rows.',
        "",
        "## Ratings by `eval_strategy_label`",
        "",
        "| strategy | n | good | meh | bad | ok % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strat, row in by_strat.iterrows():
        md_lines.append(
            f"| {strat} | {int(row['n'])} | {int(row['good'])} | "
            f"{int(row['meh'])} | {int(row['bad'])} | {row['pct_ok']}% |"
        )

    REPORT_MD.write_text("\n".join(md_lines) + "\n")
    by_strat.to_csv(REPORT_CSV)

    keep_mask = df["rating"].isin({"good", "meh"})
    cleaned = df.loc[keep_mask].drop(columns=["rating", "has_great", "strategy"])

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUT_PARQUET, index=False)
    cleaned.to_csv(OUT_CSV, index=False)

    print(f"Total rows:          {n_total}")
    print(f"  good:              {good}")
    print(f"  meh:               {meh}")
    print(f"  bad (excluded):    {bad}")
    print(f"  other/unrated:     {other}")
    print(f'  "great" comments:  {n_great}')
    print()
    print(f"Cleaned set: {len(cleaned)} rows -> {OUT_PARQUET.name} / .csv")
    print(f"Stats:                                -> {REPORT_MD.name} / .csv")


if __name__ == "__main__":
    main()
