# Gemma-4-E4B LoRA fine-tuning (reviewer-requested experiment)

Fine-tune of Gemma-4-E4B (the zero-format-failure small model) on the contrastive training data, run for the ACL ARR discussion period. Because Gemma-4-E4B emits valid JSON in 100% of open-task responses both before and after fine-tuning, the within-model before/after contrast isolates the binding deficit from output formatting.

| Gemma-4-E4B, open task (D-A) | Before FT | After FT |
|---|---:|---:|
| Accuracy | 0.585 | 0.966 |
| Macro-F1 | 0.580 | 0.980 |
| Parse errors | 0 | 0 |
| Masculine recall | 0.336 | 0.974 |
| Feminine recall | 0.443 | 0.974 |

Artifacts:

- Per-row predictions: `results/classify/test_google_gemma-4-E4B-it_zero_shot_identify_DA_open_ft_thinking_final.parquet` (plus D-A closed and D-MULTI open variants).
- Aggregate metrics + report: `results/eval/gemma_4_E4B_ft_DA_open_thinking_final/` (`FT_REPORT.md`, `prf_*.csv`, `confusion_*.csv`, per-strategy breakdowns), same layout for `_DA_closed_` and `_DMULTI_open_`.
- Training provenance: `train_provenance/` in each eval directory (`adapter_config.json`, `trainer_state.json`).
- Configs: `ft_configs/ft_gemma_E4B_*_h200.yml`. Gemma-4's vision/audio towers contain `Gemma4ClippableLinear` modules that break PEFT target-module resolution; the fix (LoRA module exclusions plus a defensive patch) is implemented in `patches/`.
- Baseline (before-FT) per-row predictions: `results/classify/test_google_gemma-4-E4B-it_zero_shot_identify_thinking_final.parquet`.

Stretch results: D-A closed accuracy 0.993, D-MULTI open accuracy 0.978 (see the corresponding eval directories).

## FT-GAND (authentic ambiguous-only) runs

Completing all conditions for the second student, the same recipe trained on the
ambiguous-only D-AUTH data reproduces the single-class collapse observed on Qwen3.5-4B:

| Gemma-4-E4B FT-GAND | Acc | macro-F1 | rec(masc) | rec(fem) | rec(amb) | parse-err |
|---|---:|---:|---:|---:|---:|---:|
| open (D-AUTH)   | 0.326 | 0.179 | 0.000 | 0.019 | 0.959 | 0 |
| closed (D-AUTH) | 0.386 | 0.268 | 0.019 | 0.140 | 1.000 | 0 |

Artifacts: `results/classify/test_google_gemma-4-E4B-it_*DAUTH*_ft_thinking_final.parquet`
and `results/eval/gemma_4_E4B_ft_DAUTH_{open,closed}_thinking_final/` (metrics, reports,
`train_provenance/`). Both runs have zero parse errors, so the collapse is a property of
the ambiguous-only supervision, consistent across both students.
