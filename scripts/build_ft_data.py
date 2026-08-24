#!/usr/bin/env python3
"""Build messages-format JSONL training files for closed-task QLoRA fine-tuning.

Reads:
  - data/final-train-00000-of-00001_with_variants_with_reasoning.parquet
  - data/final-validation-00000-of-00001_with_variants_with_reasoning.parquet

Writes 4 files into ft_data/:
  - closed_DAUTH_train.jsonl   closed_DAUTH_val.jsonl
  - closed_DA_train.jsonl      closed_DA_val.jsonl

Conditions:
  D-AUTH = source rows only (~3,918 train / ~500 val), all gold=ambiguous, canned reasoning.
  D-A    = source + masculine + feminine (~11,754 train / ~1,500 val), gold per variant, canned reasoning.

Each JSONL line is messages-format:
  {"messages": [
    {"role": "system",    "content": <SYSTEM_TASK from classify_gender.py>},
    {"role": "user",      "content": "Sentence: <s>\nReferent: <r>"},
    {"role": "assistant", "content": <D-AUTH | D-A target>}
  ]}

Hygiene check: aborts if any train/val sentence appears as a substring in the test set
(or vice-versa) after normalisation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "ft_data"

TRAIN_PQ = DATA / "final-train-00000-of-00001_with_variants_with_reasoning.parquet"
VAL_PQ   = DATA / "final-validation-00000-of-00001_with_variants_with_reasoning.parquet"
TEST_PQ  = DATA / "final_reviewed_test-00000-of-00001_with_variants.parquet"

# Verbatim from classify_gender.py:31-42
SYSTEM_TASK = """You classify the gender of a named REFERENT in an English sentence.

Output one of three labels:
- masculine  - the referent is unambiguously male
- feminine   - the referent is unambiguously female
- ambiguous  - the sentence contains no signal that reveals the referent's gender

Respond with ONLY a JSON object:
{"gender": "masculine|feminine|ambiguous", "confidence": N, "reasoning": "..."}

confidence is an integer 1-5 (5 = certain).
reasoning is one short sentence citing the textual evidence (the pronoun, the title, the gendered noun, etc.)."""

# Verbatim from classify_gender_local.py:68-72
_CANNED_REASON = {
    "masculine": "the sentence contains a male-marking signal (title, pronoun, or gendered noun) bound to the referent",
    "feminine":  "the sentence contains a female-marking signal (title, pronoun, or gendered noun) bound to the referent",
    "ambiguous": "no gendered title, pronoun, or noun in the sentence reveals the referent's gender",
}

VARIANTS = [
    # (variant_label, sentence_col, referent_col, gold_label)
    ("source",    "EN_source_sentence",    "referent",      "ambiguous"),
    ("masculine", "EN_masculine_sentence", "masc_referent", "masculine"),
    ("feminine",  "EN_feminine_sentence",  "fem_referent",  "feminine"),
]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _filter_contaminated(df: pd.DataFrame, test_sents: set[str], tag: str) -> pd.DataFrame:
    """Drop train/val rows whose any-variant sentence appears in the test set.
    Returns the filtered df and prints a count."""
    keep_mask = ~(
        df["EN_source_sentence"].map(_norm).isin(test_sents)
        | df["EN_masculine_sentence"].map(_norm).isin(test_sents)
        | df["EN_feminine_sentence"].map(_norm).isin(test_sents)
    )
    n_drop = int((~keep_mask).sum())
    if n_drop:
        pct = n_drop / len(df) * 100
        print(f"  [hygiene] {tag}: dropped {n_drop}/{len(df)} ({pct:.2f}%) rows whose any-variant sentence is in test")
    else:
        print(f"  [hygiene] {tag}: no overlap with test")
    return df.loc[keep_mask].reset_index(drop=True)


def _collect_test_sents(test_df: pd.DataFrame) -> set[str]:
    s: set[str] = set()
    for col in ["EN_source_sentence", "EN_masculine_sentence", "EN_feminine_sentence"]:
        if col in test_df.columns:
            s.update(_norm(x) for x in test_df[col].dropna() if str(x).strip())
    return s


def _user_message(sentence: str, referent: str) -> str:
    return f"Sentence: {sentence}\nReferent: {referent}"


def _ja(gender: str, reasoning: str) -> str:
    """JSON assistant target with a fixed shape."""
    return json.dumps({"gender": gender, "confidence": 5, "reasoning": reasoning},
                      ensure_ascii=False)


def _make_record(sentence: str, referent: str, assistant_content: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYSTEM_TASK},
        {"role": "user",      "content": _user_message(sentence, referent)},
        {"role": "assistant", "content": assistant_content},
    ]}


def build_dauth(df: pd.DataFrame) -> list[dict]:
    """Source rows only (authentic GAND data; all gold=ambiguous). Canned reasoning."""
    out = []
    canned = _ja("ambiguous", _CANNED_REASON["ambiguous"])
    for _, row in df.iterrows():
        s = row.get("EN_source_sentence")
        r = row.get("referent")
        if not s or not r:
            continue
        out.append(_make_record(str(s).strip(), str(r).strip(), canned))
    return out


def build_da(df: pd.DataFrame) -> list[dict]:
    """All 3 variants. Canned per-class reasoning."""
    out = []
    for _, row in df.iterrows():
        for _vlabel, sent_col, ref_col, gold in VARIANTS:
            s = row.get(sent_col)
            r = row.get(ref_col)
            if not s or not r:
                continue
            asst = _ja(gold, _CANNED_REASON[gold])
            out.append(_make_record(str(s).strip(), str(r).strip(), asst))
    return out


def write_jsonl(recs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}  ({len(recs)} records)")


def main() -> None:
    print(f"Reading {TRAIN_PQ.relative_to(ROOT)}")
    train = pd.read_parquet(TRAIN_PQ)
    print(f"  train: {len(train)} source rows, {len(train.columns)} cols")
    print(f"Reading {VAL_PQ.relative_to(ROOT)}")
    val = pd.read_parquet(VAL_PQ)
    print(f"  val:   {len(val)} source rows, {len(val.columns)} cols")
    print(f"Reading {TEST_PQ.relative_to(ROOT)}  (for hygiene check)")
    test = pd.read_parquet(TEST_PQ)
    print(f"  test:  {len(test)} source rows")

    # Filter out any train/val rows whose any-variant sentence is in the test set.
    test_sents = _collect_test_sents(test)
    print(f"  test sentences (any variant, normalised): {len(test_sents)}")
    train = _filter_contaminated(train, test_sents, "train")
    val   = _filter_contaminated(val,   test_sents, "val")

    OUT.mkdir(parents=True, exist_ok=True)

    # D-AUTH
    print("\n=== D-AUTH (source-only, gold=ambiguous, canned reasoning) ===")
    write_jsonl(build_dauth(train), OUT / "closed_DAUTH_train.jsonl")
    write_jsonl(build_dauth(val),   OUT / "closed_DAUTH_val.jsonl")

    # D-A
    print("\n=== D-A (full 2/2/2, canned reasoning) ===")
    write_jsonl(build_da(train), OUT / "closed_DA_train.jsonl")
    write_jsonl(build_da(val),   OUT / "closed_DA_val.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
