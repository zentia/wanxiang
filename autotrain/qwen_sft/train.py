"""4-bit QLoRA fine-tuning of a Qwen2.5-Coder base on local code SFT data.

Deliberately conservative for a 16 GB card with a WDDM desktop sharing VRAM:
batch 1 x grad-accum 8, gradient checkpointing, bf16, paged 8-bit optimizer.
A short smoke run verifies the whole chain before any real training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class _Rows(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.rows[i]


def _collate(features: list[dict[str, Any]], pad_id: int) -> dict[str, torch.Tensor]:
    maxlen = max(len(f["input_ids"]) for f in features)
    input_ids, labels, attention = [], [], []
    for f in features:
        n = len(f["input_ids"])
        pad = maxlen - n
        input_ids.append(f["input_ids"] + [pad_id] * pad)
        labels.append(f["labels"] + [-100] * pad)
        attention.append([1] * n + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention, dtype=torch.long),
    }


def run(
    base_dir: str | Path,
    data_dir: str | Path,
    out_dir: str | Path,
    *,
    max_steps: int = 20,
    max_samples: int | None = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> dict[str, Any]:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    data = Path(data_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    rows = _load_rows(data / "train.jsonl")
    if max_samples is not None:
        rows = rows[:max_samples]

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_dir),
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    args = TrainingArguments(
        output_dir=str(out),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=max_steps,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=min(10, max(1, max_steps // 20)),
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_Rows(rows),
        data_collator=lambda f: _collate(f, pad_id),
    )
    result = trainer.train()
    adapter = out / "adapter"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    return {
        "status": "OK",
        "base": str(base_dir),
        "samples": len(rows),
        "trainable_params": trainable,
        "steps": max_steps,
        "final_loss": float(result.training_loss),
        "adapter": str(adapter),
    }
