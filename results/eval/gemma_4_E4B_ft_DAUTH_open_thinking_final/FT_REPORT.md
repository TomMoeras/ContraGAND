# Gemma-4-E4B FT-GAND (DAUTH) — open + closed results

Per `jobs/ORDER_ft_gemma_dauth.md` (EMNLP 2026 camera-ready). Fills the missing FT-GAND cells
for \gemmaS{} in `tab:ft-open`, `tab:ft-closed`, and Figure 5. Same recipe as the DA runs
(hyperparameters identical by design; nothing tuned). Test set: `final_reviewed_test` (465 src ×
3 variants = 1,395 rows). 0 parse errors on both runs.

## Headline

| task | condition | Acc | macro-F1 | rec(masc) | rec(fem) | rec(amb) | parse-err | not-found |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| open   | zero_shot_identify | 0.326 | 0.179 | 0.000 | 0.019 | 0.959 | 0 | 63 |
| closed | zero_shot          | 0.386 | 0.268 | 0.019 | 0.140 | 1.000 | 0 | 0  |

Per-class F1 — open: masc 0.000 / fem 0.038 / amb 0.499. closed: masc 0.038 / fem 0.245 / amb 0.521.

## Verdict: single-class collapse, matching the Qwen-4B FT-GAND pattern

Both conditions collapse toward predicting **ambiguous**, exactly as the order predicted and
mirroring Qwen-4B FT-GAND:
- Open acc 0.326 (~0.33), ambiguous recall 0.959 (>0.95), masc/fem recall 0.000/0.019 (<0.05). ✓
- Closed acc 0.386 (in the 0.33–0.40 range), ambiguous recall 1.000 (~1.0). ✓

The confusion matrices show the mechanism: masculine and feminine test items are almost entirely
predicted ambiguous (open: 443/465 masc → amb; closed: 456/465 masc → amb, 400/465 fem → amb).
This is a property of the DAUTH training signal, which is authentic **ambiguous-only** GAND
sources (3,902 train / 485 val): the student memorizes that distribution (final eval loss open
0.0021, closed 4.5e-7) and learns to answer "ambiguous" almost unconditionally. It is a genuine
finding about FT-GAND supervision, consistent across both students — not a training defect.

## Provenance / run details

- Base: google/gemma-4-E4B-it; LoRA (Gemma4ClippableLinear PEFT exclusions + sitecustomize patch;
  `attn_implementation: eager`). Adapters NOT merged into the committed artifacts.
- Early stopping: open stopped at step 700 (2.87 epochs), closed at 900 (3.69 epochs).
- Wall-times (H200): open train 33m32s + eval 6m38s; closed train 34m36s + eval 5m30s.
- adapter_config.json + trainer_state.json under `train_provenance/` in each eval dir.
