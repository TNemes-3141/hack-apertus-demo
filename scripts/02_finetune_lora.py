#!/usr/bin/env python3
"""
02_finetune_lora.py
===================
LoRA supervised fine-tuning of Apertus 8B for multilingual translation.
Built on top of:
  - HuggingFace PEFT  (LoRA adapter)
  - TRL SFTTrainer    (supervised fine-tuning loop)
  - Apertus instruct  (swiss-ai/Apertus-8B-Instruct-2509)

Mirrors the approach in the official Apertus fine-tuning recipes:
  https://github.com/swiss-ai/apertus-finetuning-recipes
  https://apertvs.ai/docs/tech/fine-tuning/

Usage (GPU — recommended, requires ~20GB VRAM with bf16):
    python scripts/02_finetune_lora.py --config configs/sft_lora.yaml

Usage (GPU + 4-bit quantization — fits ~10GB VRAM):
    python scripts/02_finetune_lora.py --config configs/sft_lora.yaml --load_in_4bit

Usage (CPU — very slow, demo/smoke-test only):
    python scripts/02_finetune_lora.py --config configs/sft_lora.yaml --cpu --max_steps 5

After training, the LoRA adapter lands in output/apertus_lora/
See 03_evaluate.py to benchmark chrF and BLEU.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml


# ---------------------------------------------------------------------------
# Dependency check — give actionable error messages
# ---------------------------------------------------------------------------
def check_deps():
    missing = []
    for pkg in ["transformers", "peft", "trl", "datasets", "accelerate"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("❌  Missing dependencies:", ", ".join(missing))
        print("    Install with:")
        print("    pip install transformers>=4.56.0 peft trl datasets accelerate")
        sys.exit(1)


check_deps()

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------
def load_chat_dataset(jsonl_path: str) -> Dataset:
    """Load chat-format JSONL (output of 01_prepare_dataset.py)."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"📂  Loaded {len(records)} training examples from {jsonl_path}")
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Apertus 8B for translation")
    parser.add_argument("--config",       default="configs/sft_lora.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Quantize to 4-bit (bitsandbytes) — use on smaller GPUs")
    parser.add_argument("--cpu",          action="store_true",
                        help="Force CPU inference (very slow; demo only)")
    parser.add_argument("--max_steps",    type=int, default=None,
                        help="Override max training steps (e.g. 5 for a smoke-test)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Device setup -------------------------------------------------------
    if args.cpu:
        device     = "cpu"
        dtype      = torch.float32
        use_bf16   = False
        use_fp16   = False
        print("⚠️   CPU mode: training will be very slow. Use --max_steps 5 for a smoke-test.")
    else:
        if not torch.cuda.is_available():
            print("⚠️   No CUDA GPU detected. Falling back to CPU.")
            print("    Tip: add --cpu and --max_steps 5 to run a quick smoke-test.")
            sys.exit(1)
        device   = "cuda"
        dtype    = torch.bfloat16
        use_bf16 = True
        use_fp16 = False
        print(f"🚀  GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Tokenizer ----------------------------------------------------------
    model_path = cfg["model_path"]
    print(f"\n📥  Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # Required for SFT with causal LMs

    # ---- Model loading ------------------------------------------------------
    print(f"📥  Loading model ({model_path}) ...")

    if args.load_in_4bit and device == "cuda":
        print("    → 4-bit quantization enabled (bitsandbytes NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    elif device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    print(f"    ✅ Model loaded. Parameters: {model.num_parameters() / 1e9:.2f}B")

    # ---- LoRA config --------------------------------------------------------
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"🔧  LoRA adapter applied.")
    print(f"    Trainable params: {trainable:,} ({100 * trainable / total:.2f}% of total)")

    # ---- Dataset ------------------------------------------------------------
    dataset_path = cfg.get("dataset_path", "data/train_chat.jsonl")
    dataset = load_chat_dataset(dataset_path)

    # ---- Training arguments -------------------------------------------------
    output_dir = cfg.get("output_dir", "output/apertus_lora")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 2),
        report_to=cfg.get("report_to", "none"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True) and (device != "cpu"),
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_pin_memory=(device == "cuda"),
    )

    if args.max_steps is not None:
        train_kwargs["max_steps"] = args.max_steps
        print(f"⚡  Overriding to max_steps={args.max_steps} (smoke-test mode)")

    sft_config = SFTConfig(
        max_seq_length=cfg.get("max_seq_length", 512),
        **train_kwargs,
    )

    # ---- Trainer ------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        # SFTTrainer applies the chat template automatically when 'messages' key is present
    )

    print(f"\n🏋️  Starting LoRA fine-tuning ...")
    print(f"    Output directory: {output_dir}")
    print(f"    Epochs: {train_kwargs.get('num_train_epochs', 3)}")
    print(f"    Effective batch size: "
          f"{train_kwargs['per_device_train_batch_size'] * train_kwargs['gradient_accumulation_steps']}")

    trainer.train()

    # ---- Save ---------------------------------------------------------------
    print(f"\n💾  Saving LoRA adapter to {output_dir} ...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"""
✅  Fine-tuning complete!

Adapter saved to: {output_dir}/
  adapter_model.safetensors  ← LoRA delta weights
  adapter_config.json        ← LoRA configuration
  tokenizer.*                ← Tokenizer files

Next steps:
  1. Evaluate:  python scripts/03_evaluate.py --adapter {output_dir}
  2. Serve:     docker-compose up  (see docker-compose.yml)

To load the adapter later:
  from transformers import AutoModelForCausalLM
  from peft import PeftModel

  base  = AutoModelForCausalLM.from_pretrained("swiss-ai/Apertus-8B-Instruct-2509")
  model = PeftModel.from_pretrained(base, "{output_dir}")
  model = model.merge_and_unload()  # optional: bake into weights
""")


if __name__ == "__main__":
    main()