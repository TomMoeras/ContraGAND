import argparse
import json

import anthropic
import pandas as pd


def compute_edit_distance_words(original, modified):
    """Word-level Levenshtein edit distance."""
    orig_words = original.lower().split()
    mod_words = modified.lower().split()
    n, m = len(orig_words), len(mod_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if orig_words[i - 1] == mod_words[j - 1] else 1
            dp[j], prev = min(dp[j] + 1, dp[j - 1] + 1, prev + cost), dp[j]
    return dp[m]


def compute_metrics(df):
    """Compute per-row metrics for the generated contrastive pairs."""
    rows = []
    for idx, row in df.iterrows():
        original = row["EN_source_sentence"]
        masc = row["EN_masculine_sentence"]
        fem = row["EN_feminine_sentence"]

        if pd.isna(masc) or pd.isna(fem):
            rows.append(
                {
                    "index": idx,
                    "referent": row["referent"],
                    "masc_edit_dist": None,
                    "fem_edit_dist": None,
                    "masc_equals_original": None,
                    "fem_equals_original": None,
                    "masc_equals_fem": None,
                    "error": True,
                }
            )
            continue

        masc_dist = compute_edit_distance_words(original, masc)
        fem_dist = compute_edit_distance_words(original, fem)

        rows.append(
            {
                "index": idx,
                "referent": row["referent"],
                "strategy": row.get("disambiguation_strategy", "unknown"),
                "masc_edit_dist": masc_dist,
                "fem_edit_dist": fem_dist,
                "masc_equals_original": masc.strip() == original.strip(),
                "fem_equals_original": fem.strip() == original.strip(),
                "masc_equals_fem": masc.strip() == fem.strip(),
                "error": False,
            }
        )
    return pd.DataFrame(rows)


def llm_evaluate_sample(client, model, df, sample_size=10, seed=42):
    """Use an LLM to evaluate a sample of generated pairs on naturalness,
    semantic preservation, and gender clarity."""
    sample = df.dropna(subset=["EN_masculine_sentence", "EN_feminine_sentence"])
    sample = sample.sample(n=min(sample_size, len(sample)), random_state=seed)

    results = []
    for _, row in sample.iterrows():
        prompt = f"""Evaluate these contrastive gender sentence pairs. Score each criterion 1-5 (5=best).

Original sentence: {row["EN_source_sentence"]}
Referent: {row["referent"]}
Masculine variant: {row["EN_masculine_sentence"]}
Feminine variant: {row["EN_feminine_sentence"]}
Strategy used: {row["disambiguation_strategy"]}

Criteria:
1. MINIMALITY (5=very few edits, 1=heavily rewritten)
2. NATURALNESS (5=sounds completely natural, 1=awkward/ungrammatical)
3. SEMANTIC_PRESERVATION (5=same meaning minus gender, 1=meaning changed significantly)
4. GENDER_CLARITY (5=gender is completely unambiguous, 1=still ambiguous)
5. MASCULINE_FEMININE_CONSISTENCY (5=both variants use the same edit strategy and feel balanced, 1=asymmetric or inconsistent)

Respond with ONLY a JSON object:
{{"minimality": N, "naturalness": N, "semantic_preservation": N, "gender_clarity": N, "consistency": N, "issues": "brief description of any problems or empty string"}}"""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0].strip()
            scores = json.loads(text)
            scores["referent"] = row["referent"]
            scores["original"] = row["EN_source_sentence"][:80]
            scores["masculine"] = row["EN_masculine_sentence"][:80]
            scores["feminine"] = row["EN_feminine_sentence"][:80]
            results.append(scores)
        except Exception as e:
            print(f"  LLM eval error for {row['referent']}: {e}")

    return results


def print_report(df, metrics_df, llm_results=None):
    """Print a summary evaluation report."""
    print("=" * 70)
    print("CONTRASTIVE GENERATION EVALUATION REPORT")
    print("=" * 70)

    total = len(metrics_df)
    errors = metrics_df["error"].sum()
    print(f"\nTotal rows: {total}")
    print(f"Errors (failed generation): {errors}")
    print(f"Success rate: {(total - errors) / total:.1%}")

    valid = metrics_df[~metrics_df["error"]]
    if len(valid) == 0:
        print("\nNo valid rows to evaluate.")
        return

    print(f"\n--- Edit Distance (word-level) ---")
    print(f"Masculine avg: {valid['masc_edit_dist'].mean():.1f} words changed")
    print(f"Feminine avg:  {valid['fem_edit_dist'].mean():.1f} words changed")
    print(f"Masculine max: {valid['masc_edit_dist'].max():.0f}")
    print(f"Feminine max:  {valid['fem_edit_dist'].max():.0f}")

    print(f"\n--- Flags ---")
    print(
        f"Masculine = original (no edit): {valid['masc_equals_original'].sum()}"
    )
    print(
        f"Feminine = original (no edit):  {valid['fem_equals_original'].sum()}"
    )
    print(
        f"Masculine = feminine (identical): {valid['masc_equals_fem'].sum()}"
    )

    print(f"\n--- Strategy Distribution ---")
    print(valid["strategy"].value_counts().to_string())

    # Show high edit distance rows
    high_edit = valid[
        (valid["masc_edit_dist"] > 5) | (valid["fem_edit_dist"] > 5)
    ]
    if len(high_edit) > 0:
        print(f"\n--- High Edit Distance Rows (>{5} words changed) ---")
        for _, m in high_edit.iterrows():
            orig_row = df.loc[m["index"]] if m["index"] in df.index else None
            if orig_row is not None:
                print(f"\n  Referent: {m['referent']}")
                print(f"  Original:  {orig_row['EN_source_sentence'][:100]}")
                print(
                    f"  Masculine: {orig_row['EN_masculine_sentence'][:100]}"
                )
                print(
                    f"  Feminine:  {orig_row['EN_feminine_sentence'][:100]}"
                )
                print(
                    f"  Edit dist: masc={m['masc_edit_dist']}, fem={m['fem_edit_dist']}"
                )

    if llm_results:
        print(f"\n--- LLM Quality Scores (sample of {len(llm_results)}) ---")
        criteria = [
            "minimality",
            "naturalness",
            "semantic_preservation",
            "gender_clarity",
            "consistency",
        ]
        for c in criteria:
            vals = [r[c] for r in llm_results if c in r]
            if vals:
                print(f"  {c}: {sum(vals)/len(vals):.1f}/5")

        issues = [r for r in llm_results if r.get("issues")]
        if issues:
            print(f"\n--- Issues Found ---")
            for r in issues:
                if r["issues"]:
                    print(f"  [{r['referent']}] {r['issues']}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contrastive gender sentence pairs"
    )
    parser.add_argument(
        "input_file", help="Path to generated parquet file"
    )
    parser.add_argument(
        "--llm-eval",
        action="store_true",
        help="Run LLM-based evaluation on a sample",
    )
    parser.add_argument(
        "--llm-eval-model",
        default="claude-haiku-4-5-20251001",
        help="Model for LLM evaluation",
    )
    parser.add_argument(
        "--llm-eval-sample",
        type=int,
        default=10,
        help="Number of rows to LLM-evaluate",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_file)
    print(f"Loaded {len(df)} rows from {args.input_file}")

    metrics_df = compute_metrics(df)

    llm_results = None
    if args.llm_eval:
        print("\nRunning LLM-based evaluation...")
        client = anthropic.Anthropic()
        llm_results = llm_evaluate_sample(
            client,
            args.llm_eval_model,
            df,
            sample_size=args.llm_eval_sample,
        )

    print_report(df, metrics_df, llm_results)


if __name__ == "__main__":
    main()
