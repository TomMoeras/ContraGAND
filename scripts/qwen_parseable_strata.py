"""Per-strategy within-parseable breakdown for Qwen-4B open-task ZS.

Produces the per-strategy stratification of the within-parseable accuracy headline
(0.847) reported in App. I, Table tab:qwen-parseable-strat.

Requires the full 465-row human-reviewed test set (released on HuggingFace
post-acceptance) to be present at TEST below. The 20-row public sample
(data/sample_test_20rows.parquet) is too small to reproduce the stratification.

Usage:
    python scripts/qwen_parseable_strata.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results/classify/test_Qwen_Qwen3.5-4B_zero_shot_identify_thinking_final.parquet"
TEST = ROOT / "data/final_reviewed_test-00000-of-00001_with_variants_with_reasoning.parquet"
OUT = ROOT / "results/eval/qwen_parseable_strata.csv"

PAPER_NAME = {
    "referent_swap": "lexical swap",
    "pronoun_swap": "pronoun rewrite",
    "pronoun_insertion": "possessive tag",
    "context_modifier": "distal noun cue",
    "relational_insertion": "kin appositive",
    "title_insertion": "title appositive",
    "sir_maam": "vocative address",
    "adjective": "gender adjective",
    "appositive_dash": "dashed appositive",
}
LOCAL = {
    "gender adjective",
    "possessive tag",
    "pronoun rewrite",
    "lexical swap",
    "distal noun cue",
    "dashed appositive",
}
CROSS = {"title appositive", "vocative address", "kin appositive"}


def main() -> None:
    pred = pd.read_parquet(PRED)
    test = pd.read_parquet(TEST).reset_index(drop=True)
    test["row_id"] = test.index
    df = pred.merge(test[["row_id", "disambiguation_strategy"]], on="row_id", how="left")

    df["strategy"] = df["disambiguation_strategy"].map(PAPER_NAME)
    df["parsed"] = df["predicted_label"] != "parse_error"
    df["matched"] = ~df["predicted_label"].isin(["parse_error", "not_found"])
    df["correct_in_match"] = df["matched"] & df["correct"]
    df["cue_group"] = df["strategy"].apply(
        lambda s: "local" if s in LOCAL else ("cross" if s in CROSS else None)
    )

    def summary(group_df: pd.DataFrame, name: str) -> dict:
        n = len(group_df)
        matched = int(group_df["matched"].sum())
        correct = int(group_df["correct_in_match"].sum())
        return {
            "stratum": name,
            "n": n,
            "parse_pct": 100 * group_df["parsed"].mean(),
            "match_pct": 100 * group_df["matched"].mean(),
            "acc_match": 100 * correct / matched if matched else float("nan"),
            "acc_all": 100 * correct / n,
            "n_matched": matched,
            "n_correct": correct,
        }

    rows = []
    for s in sorted(LOCAL, key=lambda s: -df[df["strategy"] == s].shape[0]):
        rows.append(summary(df[df["strategy"] == s], s))
    for s in sorted(CROSS, key=lambda s: -df[df["strategy"] == s].shape[0]):
        rows.append(summary(df[df["strategy"] == s], s))
    rows.append(summary(df[df["cue_group"] == "local"], "LOCAL (all)"))
    rows.append(summary(df[df["cue_group"] == "cross"], "CROSS-CLAUSE (all)"))
    rows.append(summary(df, "TOTAL"))

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT.relative_to(ROOT)}")

    # Representativeness: chi-square parse x cue_group
    ct = pd.crosstab(df["cue_group"], df["parsed"])
    chi2, p, dof, _ = chi2_contingency(ct)
    print("\nParse-rate by cue group:")
    print(ct)
    print(f"chi^2 = {chi2:.2f}, dof = {dof}, p = {p:.2e}")

    # Audit the 0.847 headline
    matched = df[df["matched"]]
    print(
        f"\nWithin-matched accuracy (headline): {matched['correct'].mean():.4f} "
        f"on n={len(matched)} matched rows."
    )


if __name__ == "__main__":
    main()
