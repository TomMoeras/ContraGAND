"""Diagnose Gemma4ClippableLinear so we can pick the lightest PEFT fix.
Run on CPU (no GPU needed). Prints MRO, __init__ signature, key attrs,
the class source, and whether plain PEFT LoRA injection already works.
"""
import os, inspect, textwrap
import torch
from transformers import AutoModelForCausalLM, AutoConfig

MODEL = "google/gemma-4-E4B-it"

cfg = AutoConfig.from_pretrained(MODEL)
print("=== top-level architecture ===")
print(type(cfg).__name__, "->", cfg.architectures)

print("\n=== loading model on CPU (bf16) ===")
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype="bfloat16", device_map="cpu")
print("model class:", type(m).__name__)

# Find a Gemma4ClippableLinear instance
clip = [(n, mod) for n, mod in m.named_modules()
        if type(mod).__name__ == "Gemma4ClippableLinear"]
print(f"\n#Gemma4ClippableLinear instances: {len(clip)}")
if clip:
    name0, lin = clip[0]
    print("example name:", name0)
    print("MRO:", [c.__name__ for c in type(lin).__mro__])
    print("is nn.Linear subclass:", isinstance(lin, torch.nn.Linear))
    print("__init__ sig:", inspect.signature(type(lin).__init__))
    for k in ("in_features", "out_features", "weight", "bias"):
        v = getattr(lin, k, "<MISSING>")
        if isinstance(v, torch.Tensor):
            v = f"Tensor{tuple(v.shape)} {v.dtype}"
        print(f"  {k}: {v}")
    print("\n=== class source ===")
    try:
        print(textwrap.indent(inspect.getsource(type(lin)), "  "))
    except Exception as e:
        print("  (no source)", e)
    print("=== forward source ===")
    try:
        print(textwrap.indent(inspect.getsource(type(lin).forward), "  "))
    except Exception as e:
        print("  (no forward source)", e)

# Which of these are in the *text* decoder (our LoRA targets)?
text_targets = [n for n, mod in clip
                if any(t in n for t in ("q_proj","k_proj","v_proj","o_proj",
                                         "gate_proj","up_proj","down_proj"))
                and (".vision" not in n and ".audio" not in n and "vision_tower" not in n and "audio_tower" not in n)]
print(f"\n#clippable in text decoder proj layers: {len(text_targets)}")
print("sample:", text_targets[:4])

# Try plain PEFT injection
print("\n=== try plain get_peft_model (q/k/v/o/gate/up/down) ===")
from peft import LoraConfig, get_peft_model
import peft
print("peft version:", peft.__version__)
lc = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                "gate_proj","up_proj","down_proj"],
                task_type="CAUSAL_LM")
try:
    pm = get_peft_model(m, lc)
    pm.print_trainable_parameters()
    print("PLAIN_INJECTION_OK")
except Exception as e:
    import traceback
    print("PLAIN_INJECTION_FAILED:", type(e).__name__)
    traceback.print_exc()
