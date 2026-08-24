#!/usr/bin/env python3
"""Harvest concrete sentence-level examples for results_discussion.md.

Pulls examples from the 8 parquets matching specific failure / regression
patterns; writes them to figures/data/examples.json for inline citation in
the discussion document.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "generated" / "classify"
OUT = ROOT / "figures" / "data" / "examples.json"

QWEN_CL    = pd.read_parquet(GEN / "test_Qwen_Qwen3.6-27B_zero_shot.parquet")
QWEN_CL_T  = pd.read_parquet(GEN / "test_Qwen_Qwen3.6-27B_zero_shot_thinking.parquet")
QWEN_OP    = pd.read_parquet(GEN / "test_Qwen_Qwen3.6-27B_zero_shot_identify.parquet")
QWEN_OP_T  = pd.read_parquet(GEN / "test_Qwen_Qwen3.6-27B_zero_shot_identify_thinking.parquet")
GEMMA_CL   = pd.read_parquet(GEN / "test_google_gemma-4-31B-it_zero_shot.parquet")
GEMMA_CL_T = pd.read_parquet(GEN / "test_google_gemma-4-31B-it_zero_shot_thinking.parquet")
GEMMA_OP   = pd.read_parquet(GEN / "test_google_gemma-4-31B-it_zero_shot_identify.parquet")
GEMMA_OP_T = pd.read_parquet(GEN / "test_google_gemma-4-31B-it_zero_shot_identify_thinking.parquet")


def lookup(df: pd.DataFrame, row_id: int, variant: str) -> dict:
    sub = df[(df.row_id == row_id) & (df.variant == variant)]
    if len(sub) != 1:
        return {}
    r = sub.iloc[0]
    return {
        "predicted_label": str(r.predicted_label),
        "expected_label":  str(r.expected_label),
        "confidence":      None if pd.isna(r.confidence) else int(r.confidence),
        "reasoning":       None if pd.isna(r.reasoning) else str(r.reasoning),
        "raw_response":    str(r.raw_response),
        "correct":         bool(r.correct),
    }


def example(qwen_op_row: pd.Series) -> dict:
    rid, var = int(qwen_op_row.row_id), str(qwen_op_row.variant)
    return {
        "row_id":          rid,
        "variant":         var,
        "sentence":        str(qwen_op_row.sentence),
        "referent":        str(qwen_op_row.referent),
        "expected_label":  str(qwen_op_row.expected_label),
        "strategy":        str(qwen_op_row.strategy) if "strategy" in qwen_op_row.index else None,
        "qwen_open":       lookup(QWEN_OP,   rid, var),
        "qwen_open_think": lookup(QWEN_OP_T, rid, var),
        "qwen_closed":     lookup(QWEN_CL,   rid, var),
        "qwen_closed_think": lookup(QWEN_CL_T, rid, var),
        "gemma_open":      lookup(GEMMA_OP,  rid, var),
        "gemma_open_think": lookup(GEMMA_OP_T, rid, var),
        "gemma_closed":    lookup(GEMMA_CL,  rid, var),
    }


def attach_strategy(df: pd.DataFrame) -> pd.DataFrame:
    test = pd.read_parquet(ROOT / "data" / "test-00000-of-00001_with_variants.parquet")
    test = test.reset_index(drop=False).rename(columns={"index": "row_id"})
    if "row_id" not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    strat = test[["row_id", "disambiguation_strategy"]].rename(
        columns={"disambiguation_strategy": "strategy"}
    )
    return df.merge(strat, on="row_id", how="left")


def main() -> None:
    qwen_op = attach_strategy(QWEN_OP)
    qwen_op_t = attach_strategy(QWEN_OP_T)

    bundle: dict = {}

    print("\n=== 1.1 Qwen open pronoun-binding failures ===")
    binding_fail = qwen_op[
        (qwen_op.variant.isin(["masculine", "feminine"])) &
        (qwen_op.predicted_label == "ambiguous")
    ]
    seen_strategies = set()
    picks_4a: list[dict] = []
    preferred = ["pronoun_insertion", "sir_maam", "relational_insertion"]
    for strat in preferred:
        cands = binding_fail[binding_fail.strategy == strat]
        if len(cands) == 0:
            continue
        cands = cands.copy()
        cands["slen"] = cands["sentence"].str.len()
        row = cands.sort_values("slen").iloc[0]
        picks_4a.append(example(row))
        seen_strategies.add(strat)
        print(f"  {strat:22s} row_id={row.row_id} variant={row.variant} ref={row.referent}")
        print(f"    sentence: {row.sentence[:100]}")
    bundle["coreference_blind_examples"] = picks_4a

    print("\n=== 1.2 Qwen not_found cases ===")
    nf = qwen_op[qwen_op.predicted_label == "not_found"].copy()
    nf["slen"] = nf["sentence"].str.len()
    nf = nf.sort_values("slen")
    picks_4c: list[dict] = []
    for _, row in nf.head(50).iterrows():
        ex = example(row)
        if not any(p["row_id"] == ex["row_id"] for p in picks_4c):
            picks_4c.append(ex)
        if len(picks_4c) >= 2:
            break
    for p in picks_4c:
        print(f"  row_id={p['row_id']} variant={p['variant']} ref={p['referent']}")
        print(f"    sentence: {p['sentence'][:100]}")
    bundle["not_found_examples"] = picks_4c

    print("\n=== 1.3 Qwen closed-task thinking flips (False -> True) ===")
    qcl = QWEN_CL[["row_id", "variant", "correct", "predicted_label",
                    "expected_label", "sentence", "referent"]].rename(
        columns={"correct": "cl_correct", "predicted_label": "cl_pred"}
    )
    qclt = QWEN_CL_T[["row_id", "variant", "correct", "predicted_label"]].rename(
        columns={"correct": "clt_correct", "predicted_label": "clt_pred"}
    )
    j = qcl.merge(qclt, on=["row_id", "variant"], how="inner")
    flips = j[(~j.cl_correct) & (j.clt_correct)]
    print(f"  total flips: {len(flips)}")
    flips = attach_strategy(flips)
    picks_5a: list[dict] = []
    for strat in ["referent_swap", "sir_maam"]:
        cands = flips[(flips.strategy == strat) & (flips.variant == "masculine")]
        if len(cands) == 0:
            cands = flips[flips.strategy == strat]
        if len(cands) == 0:
            continue
        cands = cands.copy()
        cands["slen"] = cands["sentence"].str.len()
        row = cands.sort_values("slen").iloc[0]
        picks_5a.append(example(row))
        print(f"  {strat:22s} row_id={row.row_id} variant={row.variant} ref={row.referent}")
        print(f"    sentence: {row.sentence[:100]}")
    bundle["closed_thinking_flips"] = picks_5a

    print("\n=== 1.4 Qwen open-task thinking regressions (True -> False, feminine) ===")
    qop = QWEN_OP[["row_id", "variant", "correct", "predicted_label",
                    "expected_label", "sentence", "referent"]].rename(
        columns={"correct": "op_correct", "predicted_label": "op_pred"}
    )
    qopt = QWEN_OP_T[["row_id", "variant", "correct", "predicted_label"]].rename(
        columns={"correct": "opt_correct", "predicted_label": "opt_pred"}
    )
    jo = qop.merge(qopt, on=["row_id", "variant"], how="inner")
    regr = jo[(jo.variant == "feminine") & (jo.op_correct) & (~jo.opt_correct) &
              (jo.opt_pred == "ambiguous")]
    print(f"  total feminine regressions to ambiguous: {len(regr)}")
    regr = regr.copy()
    regr["slen"] = regr["sentence"].str.len()
    regr = regr.sort_values("slen")
    picks_5b: list[dict] = []
    for _, row in regr.head(20).iterrows():
        ex = example(row)
        rt = ex["qwen_open_think"]["raw_response"].lower()
        if "gender-neutral" in rt or "neutral" in rt or "no gendered" in rt:
            picks_5b.append(ex)
            if len(picks_5b) >= 2:
                break
    if len(picks_5b) < 2:
        for _, row in regr.head(20).iterrows():
            ex = example(row)
            if not any(p["row_id"] == ex["row_id"] for p in picks_5b):
                picks_5b.append(ex)
            if len(picks_5b) >= 2:
                break
    for p in picks_5b:
        print(f"  row_id={p['row_id']} ref={p['referent']}")
        print(f"    sentence: {p['sentence'][:100]}")
    bundle["open_thinking_regressions"] = picks_5b

    print("\n=== 1.5 Gemma thinking format example ===")
    g = GEMMA_CL_T[GEMMA_CL_T.variant == "source"].head(20).copy()
    target = None
    for _, r in g.iterrows():
        if r.raw_response.startswith("thought") or "thought" in r.raw_response[:30].lower():
            target = r
            break
    if target is None:
        target = g.iloc[0]
    qwen_same = QWEN_CL_T[(QWEN_CL_T.row_id == int(target.row_id)) &
                          (QWEN_CL_T.variant == target.variant)].iloc[0]
    bundle["gemma_thinking_example"] = {
        "row_id":          int(target.row_id),
        "variant":         str(target.variant),
        "sentence":        str(target.sentence),
        "referent":        str(target.referent),
        "expected_label":  str(target.expected_label),
        "gemma_response":  str(target.raw_response),
        "qwen_response":   str(qwen_same.raw_response),
    }
    print(f"  row_id={target.row_id} ref={target.referent}")

    print("\n=== 1.6 Aggregate numbers ===")
    summary = {
        "qwen": {
            "closed":       0.9142,
            "closed_think": 0.9545,
            "open":         0.7333,
            "open_think":   0.7366,
        },
        "gemma": {
            "closed":       0.9630,
            "closed_think": 0.9716,
            "open":         0.9347,
            "open_think":   0.9426,
        },
    }
    bundle["headline"] = summary

    cm_qw = pd.read_csv(ROOT / "results/eval" / "qwen3_6_27b_step2" /
                         "confusion_zero_shot_identify.csv", index_col=0)
    cm_gm = pd.read_csv(ROOT / "results/eval" / "gemma_4_31b_step2" /
                         "confusion_zero_shot_identify.csv", index_col=0)
    bundle["error_budget"] = {
        "qwen_open_confusion":  cm_qw.to_dict(),
        "gemma_open_confusion": cm_gm.to_dict(),
    }

    high_conf_qwen_op  = pd.read_csv(ROOT / "results/eval" / "qwen3_6_27b_step2" /
                                       "misclassified_high_conf_zero_shot_identify.csv")
    high_conf_gemma_op = pd.read_csv(ROOT / "results/eval" / "gemma_4_31b_step2" /
                                       "misclassified_high_conf_zero_shot_identify.csv")
    bundle["high_conf_open"] = {
        "qwen":  high_conf_qwen_op.head(5).to_dict("records"),
        "gemma": high_conf_gemma_op.head(5).to_dict("records"),
    }

    bias_qwen_op  = pd.read_csv(ROOT / "results/eval" / "qwen3_6_27b_step2" /
                                 "ambiguity_bias_zero_shot_identify.csv")
    bias_gemma_op = pd.read_csv(ROOT / "results/eval" / "gemma_4_31b_step2" /
                                 "ambiguity_bias_zero_shot_identify.csv")
    bundle["stereotype_bias"] = {
        "qwen_open":  bias_qwen_op.head(15).to_dict("records"),
        "gemma_open": bias_gemma_op.head(15).to_dict("records"),
    }

    print("\n=== 1.7 Cross-model bias overlap ===")
    bias_qwen_cl  = pd.read_csv(ROOT / "results/eval" / "qwen3_6_27b_step2" /
                                  "ambiguity_bias_zero_shot.csv")
    bias_gemma_cl = pd.read_csv(ROOT / "results/eval" / "gemma_4_31b_step2" /
                                  "ambiguity_bias_zero_shot.csv")
    qw_set_cl = set(bias_qwen_cl["referent"].astype(str).str.lower())
    gm_set_cl = set(bias_gemma_cl["referent"].astype(str).str.lower())
    qw_set_op = set(bias_qwen_op["referent"].astype(str).str.lower())
    gm_set_op = set(bias_gemma_op["referent"].astype(str).str.lower())
    overlap_cl = sorted(qw_set_cl & gm_set_cl)
    overlap_op = sorted(qw_set_op & gm_set_op)
    print(f"  closed-task overlap: {len(overlap_cl)} referents - {overlap_cl}")
    print(f"  open-task overlap:   {len(overlap_op)} referents - {overlap_op}")
    bundle["bias_overlap"] = {
        "closed_task": overlap_cl,
        "open_task":   overlap_op,
        "qwen_closed_count":  len(qw_set_cl),
        "gemma_closed_count": len(gm_set_cl),
        "qwen_open_count":    len(qw_set_op),
        "gemma_open_count":   len(gm_set_op),
    }
    bundle["stereotype_bias"]["qwen_closed"]  = bias_qwen_cl.head(15).to_dict("records")
    bundle["stereotype_bias"]["gemma_closed"] = bias_gemma_cl.head(15).to_dict("records")

    OUT.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
