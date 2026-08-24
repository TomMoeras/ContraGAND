# Gemma-4-E4B-it — LoRA fine-tune on open-task D-A (H200 GPU node)

Run date: 2026-07-09. Cluster: HPC cluster, 1× NVIDIA H200. For the ACL ARR rebuttal.

## Headline

| Metric                  | Gemma-4-E4B baseline (zs, open) | **Gemma-4-E4B FT D-A-open** | Qwen3.5-4B FT D-A-open (ref) |
| ----------------------- | ------------------------------- | --------------------------- | ---------------------------- |
| Accuracy                | 0.585                           | **0.966** (1347/1395)       | 0.958                        |
| macro-F1                | 0.580                           | **0.980**                   | 0.973                        |
| parse_error count / %   | 0 / 0.0%                        | **0 / 0.0%**                | —                            |
| not_found count         | 48                              | **41**                      | 42                           |
| masc / fem / amb recall | 0.336 / 0.443 / 0.976           | **0.974 / 0.974 / 0.948**   | 0.970 / 0.970 / 0.936        |

(Baseline recomputed here: accuracy 0.5849, parse_error 0, not_found 48 — matches the
paper's 0.585. Baseline over-predicts *ambiguous* 982/1395 times, so it keeps high
ambiguous recall but collapses masc/fem recall.)

## Why this answers the reviewer

The reviewer argues our Qwen3.5-4B open-task gain conflates **output-format learning**
with **coreference-binding** improvement, because Qwen's zero-shot open failures are
69% parse errors — so fine-tuning could merely be teaching JSON.

Gemma-4-E4B **has 0 parse errors at zero-shot** (and 0 after FT). Its open-task
failure is therefore *not* a formatting problem: at baseline it emits perfectly valid
JSON but mislabels — it calls masculine/feminine referents "ambiguous"
(masc recall 0.336, fem recall 0.443). Fine-tuning on D-A-open raises those recalls to
**0.974 / 0.974** with **no format to fix**. The +38.1 pp accuracy gain (0.585 → 0.966)
is thus attributable to **binding/enumeration**, not formatting — the confound the
reviewer raises does not apply to this model, and the effect survives.

It also closes the closed→open gap: Gemma's closed-task baseline is 0.953 (open baseline
0.585 → 36.9 pp gap); FT on open brings open to 0.966, i.e. up to closed-task level.

## Training

- Config: `ft_configs/ft_gemma_E4B_DA_open_h200.yml` (plain LoRA, bf16 base;
  r=64, α=128, dropout 0.05; targets q/k/v/o/gate/up/down_proj (text decoder only);
  lr 2e-5 cosine, warmup 100, micro-bsz 8 × grad-accum 2, seq 1024, sample_packing off,
  train_on_inputs false, 5 epochs, early_stopping_patience 3, eval/save every 100).
- Data: `ft_data/open_DA_train.jsonl` (11,706) / `open_DA_val.jsonl` (1,455).
- **Epochs completed:** 1.5 (early stopping fired at step 1400; best = step 1100).
- **Final losses:** best eval_loss **0.00337** (step 1100); final train loss ~0.0015.
  eval_loss trajectory: 2.4225 → 0.0086 (100) → 0.0045 (300) → 0.0039 (700) → **0.0034 (1100)**.
- **Wall-clock:** full train 25 min 54 s on 1× H200.
- **GPU-hours (all single-GPU):** train 0.43 + smokes 0.14 + failed-smoke tokenization
  0.36 + merge/infer/eval 0.11 ≈ **~1.0 GPU-h** of H200 time.
- Trainable params: 139,526,144 / 8,080,626,976 = **1.73%**.

## The Gemma-4 enablement (three blockers, all fixed) — see patches/

1. **PEFT can't wrap `Gemma4ClippableLinear`.** It is an `nn.Module` wrapping an
   `nn.Linear` (not a subclass), living **only in the vision_tower + audio_tower**.
   PEFT's name-suffix matching grabbed those towers' `q_proj`/etc. Fix (pure config):
   `lora_exclude_modules: '.*(vision_tower|audio_tower|multi_modal_projector)\..*'`
   (axolotl → PEFT `exclude_modules`). Text decoder proj layers are plain `nn.Linear`.
2. **axolotl's Gemma-4 fused-attention monkeypatch crashes** (`AttributeError:
   'Gemma4TextAttention' has no attribute 'kv_shared_layer_index'` under transformers
   5.9.0). It is applied unconditionally for gemma4. Fix: `patches/sitecustomize.py`
   (injected via `PYTHONPATH=$REPO/patches`) no-ops `patch_gemma4_fused_attn` → stock
   attention forward.
3. **sdpa produces NaN gradients** on Gemma-4 (global layers head_dim=512): forward is
   finite but grad_norm is NaN from step 1. Fix: `attn_implementation: eager` (stable,
   verified in smoke: loss 2.46→0.10 over 25 steps with finite grad).
   Also: `pip install torchvision` (multimodal AutoProcessor needs Gemma4VideoProcessor).

Merged for inference (`ft_outputs/ft_gemma_E4B_DA_open/merged`, 15 GB, bf16) so the
custom architecture never has to be served as a live LoRA; inference via
`--backend transformers` (`AutoModelForCausalLM` resolves to
`Gemma4ForConditionalGeneration`, text-only generate works). Merged tokenizer carries
the identical training chat template (`chat_template.jinja`, llama3 fallback) → prompt
format matches training.

## Three sample raw responses (one per variant, referent "colleague")

- **source (expected ambiguous → predicted ambiguous):**
  `I haven't called a colleague today, but the one whom I love.`
  → `{"referents":[{"referent":"colleague","gender":"ambiguous","confidence":5,"reasoning":"no gendered title, pronoun, or noun in the sentence reveals the referent's gender"}]}`
- **masculine (expected masculine → predicted masculine):**
  `I haven't called a male colleague today, but the one whom I love.`
  → `{"referents":[{"referent":"colleague","gender":"masculine","confidence":5,"reasoning":"the sentence contains a male-marking signal (title, pronoun, or gendered noun) bound to the referent"}]}`
- **feminine (expected feminine → predicted feminine):**
  `I haven't called a female colleague today, but the one whom I love.`
  → `{"referents":[{"referent":"colleague","gender":"feminine","confidence":5,"reasoning":"the sentence contains a female-marking signal (title, pronoun, or gendered noun) bound to the referent"}]}`

## Stretch goals (both completed cleanly)

### Stretch 1 — D-A **closed** FT (single-referent classify; `--condition zero_shot`)

| Metric | closed baseline (ref) | **Gemma-4-E4B FT D-A-closed** |
| ------ | --------------------- | ----------------------------- |
| Accuracy | 0.953 | **0.993** (1385/1395) |
| macro-F1 | — | **0.993** |
| parse_error | — | **0 / 0.0%** |
| not_found | — | **0** |
| masc / fem / amb recall | — | **0.991 / 0.991 / 0.996** |

Best eval_loss 0.00088 (step 1200, epoch 1.64). Config
`ft_configs/ft_gemma_E4B_DA_closed_h200.yml`; eval
`results_local/gemma_4_E4B_ft_DA_closed_thinking_final/`. FT lifts the already-strong
closed task 0.953 → 0.993 (near-ceiling).

### Stretch 2 — D-MULTI **open** FT (multi-referent enumerate; `--condition zero_shot_identify`)

| Metric | Qwen3.5-4B FT D-MULTI-open (ref) | **Gemma-4-E4B FT D-MULTI-open** |
| ------ | -------------------------------- | ------------------------------- |
| Accuracy | 0.976 | **0.978** (1364/1395) |
| macro-F1 | — | **0.982** |
| parse_error | — | 9 / 0.6% |
| not_found | — | 2 |
| masc / fem / amb recall | — | **0.966 / 0.979 / 0.989** |

Best eval_loss 0.08306 (step 1400, epoch 1.92; harder multi-referent objective than D-A).
Config `ft_configs/ft_gemma_E4B_DMULTI_open_h200.yml`; eval
`results_local/gemma_4_E4B_ft_DMULTI_open_thinking_final/`. Matches Qwen's 0.976 on the
richer D-MULTI open task (the 9 parse errors, 0.6%, are the only non-zero parse count in
the whole set — the multi-referent training occasionally over-enumerates on single-referent
test rows).

### Session GPU budget (all single-GPU H200)
D-A-open train 25:54 + smokes 8:30 + failed-smoke tokenization 20:01 + D-A-open MIE 6:44
+ D-MULTI train 55:23 + D-A-closed train 44:49 + D-MULTI MIE 12:23 + D-A-closed MIE 6:08
≈ **~3.3 GPU-hours** total across all three conditions.

## Artifacts
- Predictions: `results/classify/test_gemma-4-E4B-it_zero_shot_identify_DA_open_ft_HPC.parquet` (+ .csv)
- Full eval: `results_local/gemma_4_E4B_ft_DA_open_thinking_final/` (haiku_test_report.md, prf/confusion CSVs)
- Adapter: `ft_outputs/ft_gemma_E4B_DA_open/adapter_model.safetensors` (not committed; 620 MB)
