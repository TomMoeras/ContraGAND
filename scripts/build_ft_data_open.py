#!/usr/bin/env python3
"""Build messages-format JSONL training files for OPEN-task QLoRA fine-tuning.

System prompt: SYSTEM_TASK_IDENTIFY (model sees only `Sentence: ...`, must list all
referents with their genders).

Reads:
  - data/final-train-00000-of-00001_with_variants_with_reasoning.parquet
  - data/final-validation-00000-of-00001_with_variants_with_reasoning.parquet

Writes 4 files into ft_data/:
  - open_DAUTH_train.jsonl   open_DAUTH_val.jsonl
  - open_DA_train.jsonl      open_DA_val.jsonl

Conditions for now:
  D-AUTH = source rows only, single-referent assistant target with gold=ambiguous.
  D-A    = source + masculine + feminine, single-referent assistant target with gold per variant.

Each JSONL line:
  {"messages": [
    {"role": "system",    "content": <SYSTEM_TASK_IDENTIFY from classify_gender.py>},
    {"role": "user",      "content": "Sentence: <s>"},                      # no Referent line
    {"role": "assistant", "content": "{\"referents\": [{...}]}"}
  ]}

Hygiene check: drop train/val rows whose any-variant sentence is in the test set.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "ft_data"

TRAIN_PQ = DATA / "final-train-00000-of-00001_with_variants_with_reasoning.parquet"
VAL_PQ   = DATA / "final-validation-00000-of-00001_with_variants_with_reasoning.parquet"
TEST_PQ  = DATA / "final_reviewed_test-00000-of-00001_with_variants.parquet"

# Verbatim from classify_gender.py:45-58
SYSTEM_TASK_IDENTIFY = """You read an English sentence, identify every role, occupation, or relational REFERENT in it (e.g. doctor, teacher, brother, neighbor, captain), and classify each one's gender.

For every referent you find, classify it as one of:
- masculine  - unambiguously male
- feminine   - unambiguously female
- ambiguous  - no signal in the sentence reveals this referent's gender

Use the surface form as it appears in the sentence (e.g. "doctor", not "the doctor"). List the referents in the order they appear.

Respond with ONLY a JSON object:
{"referents": [{"referent": "...", "gender": "masculine|feminine|ambiguous", "confidence": N, "reasoning": "..."}]}

confidence is an integer 1-5 (5 = certain).
reasoning is one short sentence per referent citing the textual evidence."""

_CANNED_REASON = {
    "masculine": "the sentence contains a male-marking signal (title, pronoun, or gendered noun) bound to the referent",
    "feminine":  "the sentence contains a female-marking signal (title, pronoun, or gendered noun) bound to the referent",
    "ambiguous": "no gendered title, pronoun, or noun in the sentence reveals the referent's gender",
}

VARIANTS = [
    ("source",    "EN_source_sentence",    "referent",      "ambiguous"),
    ("masculine", "EN_masculine_sentence", "masc_referent", "masculine"),
    ("feminine",  "EN_feminine_sentence",  "fem_referent",  "feminine"),
]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _filter_contaminated(df: pd.DataFrame, test_sents: set[str], tag: str) -> pd.DataFrame:
    keep_mask = ~(
        df["EN_source_sentence"].map(_norm).isin(test_sents)
        | df["EN_masculine_sentence"].map(_norm).isin(test_sents)
        | df["EN_feminine_sentence"].map(_norm).isin(test_sents)
    )
    n_drop = int((~keep_mask).sum())
    if n_drop:
        print(f"  [hygiene] {tag}: dropped {n_drop}/{len(df)} ({n_drop/len(df)*100:.2f}%) contaminated rows")
    else:
        print(f"  [hygiene] {tag}: clean")
    return df.loc[keep_mask].reset_index(drop=True)


def _collect_test_sents(test_df: pd.DataFrame) -> set[str]:
    s: set[str] = set()
    for col in ["EN_source_sentence", "EN_masculine_sentence", "EN_feminine_sentence"]:
        if col in test_df.columns:
            s.update(_norm(x) for x in test_df[col].dropna() if str(x).strip())
    return s


def _user_message(sentence: str) -> str:
    return f"Sentence: {sentence}"


def _ja(referent: str, gender: str, reasoning: str) -> str:
    """Open-task assistant target: {referents:[{referent, gender, confidence, reasoning}]}."""
    return json.dumps(
        {"referents": [{
            "referent": referent,
            "gender": gender,
            "confidence": 5,
            "reasoning": reasoning,
        }]},
        ensure_ascii=False,
    )


def _make_record(sentence: str, assistant_content: str) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYSTEM_TASK_IDENTIFY},
        {"role": "user",      "content": _user_message(sentence)},
        {"role": "assistant", "content": assistant_content},
    ]}


def build_dauth(df: pd.DataFrame) -> list[dict]:
    """Source rows only; single referent labelled ambiguous."""
    out = []
    for _, row in df.iterrows():
        s = row.get("EN_source_sentence")
        r = row.get("referent")
        if not s or not r:
            continue
        asst = _ja(str(r).strip(), "ambiguous", _CANNED_REASON["ambiguous"])
        out.append(_make_record(str(s).strip(), asst))
    return out


def build_da(df: pd.DataFrame) -> list[dict]:
    """All 3 variants; single referent per record with gold per variant."""
    out = []
    for _, row in df.iterrows():
        for _vlabel, sent_col, ref_col, gold in VARIANTS:
            s = row.get(sent_col)
            r = row.get(ref_col)
            if not s or not r:
                continue
            asst = _ja(str(r).strip(), gold, _CANNED_REASON[gold])
            out.append(_make_record(str(s).strip(), asst))
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
    print(f"Reading {VAL_PQ.relative_to(ROOT)}")
    val = pd.read_parquet(VAL_PQ)
    print(f"Reading {TEST_PQ.relative_to(ROOT)}  (for hygiene)")
    test = pd.read_parquet(TEST_PQ)

    test_sents = _collect_test_sents(test)
    train = _filter_contaminated(train, test_sents, "train")
    val   = _filter_contaminated(val,   test_sents, "val")

    OUT.mkdir(parents=True, exist_ok=True)

    print("\n=== D-AUTH-open (source-only) ===")
    write_jsonl(build_dauth(train), OUT / "open_DAUTH_train.jsonl")
    write_jsonl(build_dauth(val),   OUT / "open_DAUTH_val.jsonl")

    print("\n=== D-A-open (full 2/2/2) ===")
    write_jsonl(build_da(train), OUT / "open_DA_train.jsonl")
    write_jsonl(build_da(val),   OUT / "open_DA_val.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
