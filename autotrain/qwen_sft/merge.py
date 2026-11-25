"""Merge the QLoRA adapter into the 7B base and export a full bf16 model.

GGUF stores complete weights, not LoRA deltas, so this merge is a REQUIRED
step before convert_hf_to_gguf.py. Runs on CPU: loading the 14 GB base in
bf16 needs plenty of RAM but avoids competing with the desktop for VRAM.

    python autotrain/qwen_sft/merge.py
    # then copy vocab.json / merges.txt into runs/qwen-merged/ (BPE assets)
"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "runs/qwen-base/7b"
ADAPTER = "runs/qwen-sft/run7b/adapter"
OUT = "runs/qwen-merged"


def main() -> None:
    print("loading base in bf16 on CPU ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
    )
    print("attaching adapter ...", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER)
    print("merging ...", flush=True)
    model = model.merge_and_unload()
    model.config.use_cache = True
    print("saving merged model ...", flush=True)
    model.save_pretrained(OUT, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
