"""Auto-imported (via PYTHONPATH) in every Python process of the training job,
including the accelerate-launched ``axolotl.cli.train`` subprocess.

Purpose: neutralize axolotl 0.17.0's Gemma-4 fused-attention monkeypatch.

axolotl/loaders/patch_manager.py unconditionally applies
``patch_gemma4_fused_attn`` for any ``model_config_type in {gemma4, gemma4_text}``
(unlike the Qwen fused kernels, which require ``fused_attn_kernel: true``). Its
fused forward does:

    key_states, value_states = shared_kv_states[self.kv_shared_layer_index]

but transformers 5.9.0's ``Gemma4TextAttention`` does not expose a
``kv_shared_layer_index`` attribute (KV-sharing bookkeeping moved into
``past_key_values.shared_layers``), so training dies on the first forward:

    AttributeError: 'Gemma4TextAttention' object has no attribute 'kv_shared_layer_index'

We turn the patch installer into a no-op, so ``Gemma4TextAttention.forward``
keeps its stock (correct) transformers implementation. The only thing lost is a
fused RMSNorm+RoPE speedup — irrelevant for a 4B model at sequence_len 1024 on
an H200. patch_manager imports the symbol *inside* the function at call time, so
replacing the module attribute here (before model load) is sufficient.
"""
import sys


def _neutralize_gemma4_fused_attn():
    try:
        import axolotl.monkeypatch.models.gemma4.fused_attn as _fa
    except Exception:
        return  # axolotl not present in this process; nothing to do

    def _noop(*args, **kwargs):
        print("[sitecustomize] gemma4 fused_attn patch neutralized "
              "(using stock transformers attention forward)", file=sys.stderr)
        return None

    _fa.patch_gemma4_fused_attn = _noop


_neutralize_gemma4_fused_attn()
