#!/usr/bin/env python3
"""Train the Qwen2.5-7B-Instruct LoRA adapter reported in the paper.

This is an optional GPU-intensive regeneration path. It intentionally uses
bfloat16 LoRA without 4-bit quantisation; the reported system is not QLoRA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

SYSTEM = (
    "You are a FrameNet expert for Turkish. Given a Turkish verb expression "
    "(SYNSET) and its definition (DEFINITION), output the single most "
    "appropriate FrameNet frame name. Output ONLY the frame name, nothing else."
)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def build_text(tokenizer: AutoTokenizer, row: pd.Series) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"SYNSET: {row['SYNSET']}\nDEFINITION: {row['DEFINITION']}",
        },
        {"role": "assistant", "content": str(row["FRAME1"])},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    set_seed(42)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_df = pd.read_csv(args.train)
    required = {"SYNSET", "DEFINITION", "FRAME1"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"Training file is missing columns: {sorted(missing)}")
    train_df = train_df.copy()
    train_df["SYNSET"] = train_df["SYNSET"].fillna("")
    train_df["DEFINITION"] = train_df["DEFINITION"].fillna("")
    train_df["text"] = train_df.apply(lambda row: build_text(tokenizer, row), axis=1)
    dataset = Dataset.from_pandas(train_df[["text"]], preserve_index=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    training_args = SFTConfig(
        output_dir=str(out),
        dataset_text_field="text",
        max_length=512,
        packing=False,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=50,
        optim="paged_adamw_8bit",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(out / "final_adapter")
    tokenizer.save_pretrained(out / "final_adapter")

    metadata = {
        "base_model": args.base_model,
        "training_rows": len(train_df),
        "epochs": 3,
        "per_device_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 32,
        "learning_rate": 2e-4,
        "warmup_steps": 50,
        "max_sequence_length": 512,
        "packing": False,
        "precision": "bfloat16",
        "optimizer": "paged_adamw_8bit",
        "seed": 42,
        "quantisation": None,
        "lora_rank": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.05,
        "target_modules": TARGET_MODULES,
        "checkpoint_selection": "final epoch; no validation-based selection",
    }
    with open(out / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
