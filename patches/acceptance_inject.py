"""Acceptance (injection stage): prove LoRA injects cleanly once the vision/audio
towers are excluded. CPU-only; forward/backward is exercised by the GPU smoke run.
"""
import re, collections, torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "google/gemma-4-E4B-it"
EXCLUDE = r".*(vision_tower|audio_tower|multi_modal_projector)\..*"
TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype="bfloat16", device_map="cpu")
print("model class:", type(m).__name__)

# Prefixes that hold Gemma4ClippableLinear (must all be excluded)
pref = collections.Counter()
for n, mod in m.named_modules():
    if type(mod).__name__ == "Gemma4ClippableLinear":
        pref[n.split(".layers.")[0]] += 1
print("Gemma4ClippableLinear parents:", dict(pref))

# text-decoder proj layers: confirm plain nn.Linear + capture their path prefix
text_proj = [(n, type(mod).__name__) for n, mod in m.named_modules()
             if n.endswith(tuple("."+t for t in TARGETS))
             and not re.match(EXCLUDE, n)]
print(f"#targeted (post-exclude) proj layers: {len(text_proj)}")
print("  sample:", text_proj[:3])
print("  all plain nn.Linear:", all(t=="Linear" for _, t in text_proj))

lc = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                target_modules=TARGETS, exclude_modules=EXCLUDE,
                task_type="CAUSAL_LM")
pm = get_peft_model(m, lc)
pm.print_trainable_parameters()

# Verify no Gemma4ClippableLinear got a LoRA adapter
wrapped_clip = [n for n, mod in pm.named_modules()
                if type(mod).__name__ == "Gemma4ClippableLinear"
                and hasattr(mod, "lora_A")]
n_lora = sum(1 for n, _ in pm.named_modules() if n.endswith(".lora_A.default"))
print("LoRA-wrapped Gemma4ClippableLinear (must be 0):", len(wrapped_clip))
print("total lora_A modules:", n_lora)
print("INJECTION_OK" if not wrapped_clip and n_lora > 0 else "INJECTION_PROBLEM")
