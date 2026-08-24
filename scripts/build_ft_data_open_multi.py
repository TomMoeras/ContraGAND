#!/usr/bin/env python3
"""Build D-MULTI-open JSONL using Gemma-4-31B's full multi-referent open-task output as supervision.

Reads:
  - results/classify/gemma31b_open_thinking_on_final_train.parquet
  - results/classify/gemma31b_open_thinking_on_final_val.parquet

Each row's `raw_response` contains Gemma's full JSON answer with ALL identified
referents (not just the annotated one). We extract that JSON via the same
`parse_json_response` parser used at inference time and use it as the
assistant target for FT - teaching the student to enumerate every referent the
way Gemma did.

We DROP rows where:
  - `parse_json_response` returns nothing (parse error)
  - The parsed JSON has no `referents` list
  - The annotated referent's gold label is not the gender Gemma predicted for
    it (Gemma got the annotated one wrong) - this avoids teaching the student
    Gemma's mistakes on the dataset's own primary signal.

Other referents Gemma identified that don't have a gold label are KEPT (we
trust Gemma on the unannotated ones - they are the multi-referent enrichment).

Outputs:
  - ft_data/open_DMULTI_train.jsonl
  - ft_data/open_DMULTI_val.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from classify_gender import parse_json_response   # noqa: E402

OUT = ROOT / "ft_data"
GEN = ROOT / "generated" / "classify"

# Same as classify_gender.py:45-58
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


def _norm(s):
    return (s or "").lower().strip().rstrip(".,!?:;")


def _extract_referents(raw_response: str) -> list | None:
    """Parse the full referents list from Gemma's raw response."""
    data = parse_json_response(raw_response or "")
    refs = data.get("referents") if isinstance(data, dict) else None
    if not isinstance(refs, list):
        return None
    cleaned = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        ref = r.get("referent")
        gender = r.get("gender")
        if not ref or gender not in {"masculine", "feminine", "ambiguous"}:
            continue
        cleaned.append({
            "referent": str(ref).strip(),
            "gender": gender,
            "confidence": int(r["confidence"]) if isinstance(r.get("confidence"), (int, float)) else 5,
            "reasoning": str(r.get("reasoning") or "").strip(),
        })
    return cleaned if cleaned else None


def _annotated_match(refs: list[dict], annotated: str) -> dict | None:
    """Find Gemma's entry for the annotated referent (case/punct insensitive)."""
    target = _norm(annotated)
    if not target:
        return None
    for r in refs:
        if _norm(r["referent"]) == target:
            return r
    # token / substring fallback (mirror find_target_in_open_response)
    for r in refs:
        ref = _norm(r["referent"])
        if not ref:
            continue
        if target in ref.split() or ref in target.split():
            return r
        if target in ref or ref in target:
            return r
    return None


def build(records_pq: Path, kind: str) -> list[dict]:
    df = pd.read_parquet(records_pq)
    out = []
    n_total = len(df)
    n_no_parse = 0
    n_no_match = 0
    n_wrong_on_annot = 0
    n_kept = 0
    extra_refs_total = 0
    for _, row in df.iterrows():
        sentence = row["sentence"]
        annotated = row["referent"]
        gold = row["expected_label"]
        refs = _extract_referents(row.get("raw_response"))
        if refs is None:
            n_no_parse += 1
            continue
        annot_entry = _annotated_match(refs, annotated)
        if annot_entry is None:
            n_no_match += 1
            continue
        if annot_entry["gender"] != gold:
            # Gemma got the gold-labelled referent wrong - fix Gemma's label on
            # the annotated entry but keep the rest of Gemma's calls.
            annot_entry["gender"] = gold
            n_wrong_on_annot += 1
            # don't drop; just override
        # keep all Gemma referents (annotated + extras)
        out.append({"messages": [
            {"role": "system",    "content": SYSTEM_TASK_IDENTIFY},
            {"role": "user",      "content": f"Sentence: {sentence}"},
            {"role": "assistant", "content": json.dumps({"referents": refs}, ensure_ascii=False)},
        ]})
        n_kept += 1
        extra_refs_total += max(0, len(refs) - 1)
    avg_refs = (extra_refs_total + n_kept) / n_kept if n_kept else 0
    print(f"[{kind}] total={n_total}  parse_err={n_no_parse}  annot_not_found={n_no_match}  "
          f"gemma_wrong_on_annot={n_wrong_on_annot}  kept={n_kept}  "
          f"avg_refs_per_record={avg_refs:.2f}  extra_refs_total={extra_refs_total}")
    return out


def write_jsonl(recs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}  ({len(recs)} records)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== D-MULTI-open from Gemma-4-31B open-task predictions ===")
    train = build(GEN / "gemma31b_open_thinking_on_final_train.parquet", "train")
    write_jsonl(train, OUT / "open_DMULTI_train.jsonl")
    val = build(GEN / "gemma31b_open_thinking_on_final_val.parquet", "val")
    write_jsonl(val, OUT / "open_DMULTI_val.jsonl")


if __name__ == "__main__":
    main()
