"""
LoRA Training Script for Countdown Task using GRPO

Trains Qwen2.5-1.5B-Instruct with LoRA adapters using Group Relative
Policy Optimization (GRPO) from TRL. The model learns to solve countdown
math puzzles through reward-based reinforcement learning.
"""

import json
import argparse
import time
from pathlib import Path

import yaml
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOConfig, GRPOTrainer

from reward import format_reward_func, correctness_reward_func


def load_config(config_path: str | None = None) -> dict:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to the YAML config. Defaults to configs/lora_config.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "lora_config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_training_data(data_dir: str | None = None) -> Dataset:
    """Load processed training data from JSON.

    Args:
        data_dir: Path to directory containing train.json.

    Returns:
        HuggingFace Dataset object.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    train_path = data_dir / "train.json"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run 'python part1_tinyzero/data/countdown.py' first."
        )

    with open(train_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Dataset.from_list(data)


def print_trainable_params(model):
    """Print the number of trainable vs total parameters.

    Args:
        model: The model (with LoRA adapters applied).
    """
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    pct = 100 * trainable / total if total > 0 else 0
    print(f"\nTrainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)\n")


def main():
    """Run GRPO training with LoRA adapters."""
    parser = argparse.ArgumentParser(description="Train LoRA with GRPO")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only 20 steps on CPU for local testing",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output directory for checkpoints and logs",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    model_name = config["model_name"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]

    # Override settings for dry-run
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE - 20 steps on CPU")
        print("=" * 60)
        train_cfg["num_train_epochs"] = 1
        train_cfg["per_device_train_batch_size"] = 2
        train_cfg["gradient_accumulation_steps"] = 1
        train_cfg["num_generations"] = 2
        train_cfg["max_completion_length"] = 64
        train_cfg["save_steps"] = 999999
        train_cfg["logging_steps"] = 5

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "lora_checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {model_name}")
    device_map = "cpu" if args.dry_run else "auto"
    dtype = torch.float32 if args.dry_run else torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    print_trainable_params(model)

    # Load data
    print("Loading training data...")
    dataset = load_training_data(args.data_dir)
    if args.dry_run:
        dataset = dataset.select(range(min(100, len(dataset))))
    print(f"Training examples: {len(dataset)}")

    # Configure GRPO training
    max_steps = 20 if args.dry_run else train_cfg.get("max_steps", -1)
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_completion_length=train_cfg["max_completion_length"],
        num_generations=train_cfg["num_generations"],
        temperature=train_cfg["temperature"],
        save_steps=train_cfg["save_steps"],
        logging_steps=train_cfg["logging_steps"],
        max_steps=max_steps,
        report_to="none",
        log_level="info",
        bf16=not args.dry_run,
        use_vllm=False,
    )

    # Set up reward functions
    reward_funcs = [format_reward_func, correctness_reward_func]

    # Initialize trainer
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

    # Resume from checkpoint if specified
    resume_from = args.checkpoint if args.checkpoint else None

    # Train
    print("\nStarting training...")
    start_time = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from)
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save adapter weights
    adapter_path = output_dir / "final_adapter"
    print(f"Saving adapter weights to {adapter_path}")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # Save training log
    log_data = {
        "model_name": model_name,
        "lora_config": lora_cfg,
        "training_config": train_cfg,
        "dry_run": args.dry_run,
        "training_time_seconds": elapsed,
        "train_result": {
            k: v for k, v in train_result.metrics.items()
        },
    }
    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"Training log saved to {log_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
