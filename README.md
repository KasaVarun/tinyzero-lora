# TinyZero-LoRA: Reproducing TinyZero Countdown with LoRA + GRPO

This project reproduces the [TinyZero](https://github.com/Jiayi-Pan/TinyZero) countdown task using **LoRA** (Low-Rank Adaptation) instead of full fine-tuning, trained with **GRPO** (Group Relative Policy Optimization).

The countdown task: given a set of numbers, create an equation using +, -, *, / that equals a target number, with each number used at most once.

## Project Structure

```
tinyzero-lora/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── lora_config.yaml          # All hyperparameters
├── part1_tinyzero/
│   ├── data/
│   │   └── countdown.py          # Data preprocessing
│   ├── reward.py                 # Reward functions for GRPO
│   ├── train_lora.py             # Main training script
│   ├── evaluate.py               # Evaluation script
│   └── modal_train.py            # Modal GPU cloud wrapper
└── part2_agentflow/              # Scaffold for Part 2
    ├── agents/
    ├── benchmarks/
    └── modal_inference.py
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Modal Setup (for cloud GPU training)

```bash
pip install modal
modal setup
# Optional: create a HuggingFace secret
modal secret create huggingface-secret HF_TOKEN=<your_token>
```

## Usage

### Step 1: Preprocess Data

```bash
python part1_tinyzero/data/countdown.py
```

This downloads the `Jiayi-Pan/Countdown-Tasks-3to4` dataset from HuggingFace, formats prompts, and saves train/test splits to `part1_tinyzero/data/processed/`.

### Step 2: Local Dry-Run Test

Test that everything works on CPU (no GPU required):

```bash
python part1_tinyzero/train_lora.py --dry-run
```

This runs 20 training steps with reduced batch size to verify the pipeline.

### Step 3: Full Training on Modal

```bash
modal run part1_tinyzero/modal_train.py --action train
```

This launches training on an A10G GPU (24GB VRAM) with a 2-hour timeout. Checkpoints are saved to a persistent Modal Volume.

### Step 4: Evaluate

Run evaluation locally (if you have a GPU):

```bash
python part1_tinyzero/evaluate.py --checkpoint_path outputs/lora_checkpoint/final_adapter --num_samples 200
```

Or on Modal:

```bash
modal run part1_tinyzero/modal_train.py --action evaluate --num-samples 200
```

### Step 5: Download Results

```bash
modal run part1_tinyzero/modal_train.py --action download
```

## Configuration

All hyperparameters are in `configs/lora_config.yaml`:

- **Model**: Qwen/Qwen2.5-1.5B-Instruct
- **LoRA**: r=16, alpha=64, targeting attention projections
- **Training**: GRPO with lr=1e-4, batch_size=4, 3 epochs, 8 generations per prompt

## Expected Results

| Metric | Before Training | After Training |
|--------|----------------|----------------|
| Accuracy | ~5% | ~40%+ |
| Format Compliance | ~10% | ~90%+ |

The model learns to:
1. Use the `<think>...</think>` and `<answer>...</answer>` format
2. Construct valid arithmetic equations
3. Reach the target number using the given numbers

## Technical Details

- **LoRA** reduces trainable parameters from ~1.5B to ~4M (~0.3%)
- **GRPO** uses group-relative rewards instead of a separate value model
- Reward = correctness (1.0) + format compliance (0.1)
- Safe equation evaluation via AST parsing (no `eval()` on raw strings)
