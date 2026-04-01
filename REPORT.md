# Reproducing TinyZero Countdown Task with LoRA Adapters and GRPO Training

**Student:** Varun Kasa
**Course:** Self-Improving AI, Northeastern University
**Due:** April 6, 2025
**Part:** 1 of 2

---

## 1. Introduction

The TinyZero project demonstrated that small language models can develop reasoning capabilities through reinforcement learning, specifically by training on a "countdown" math task. In this task, the model receives a set of numbers and a target value, then must construct an arithmetic equation using those numbers (each at most once) that evaluates to the target. For example, given numbers `[1, 2, 3, 4]` and target `10`, a valid answer is `1 + 2 + 3 + 4`.

The goal of this project is to reproduce TinyZero's countdown task training pipeline, but with one key modification: instead of full fine-tuning, we use **LoRA (Low-Rank Adaptation)** adapters. LoRA freezes the base model weights and injects small trainable rank-decomposition matrices into the attention layers. This reduces the number of trainable parameters by over 99%, making training feasible on a single GPU with significantly less memory. The training algorithm remains the same: **GRPO (Group Relative Policy Optimization)**, a reinforcement learning method that uses reward signals to guide the model toward producing correct, well-formatted answers.

## 2. Technical Approach

### Base Model

We use **Qwen2.5-1.5B-Instruct**, a 1.5 billion parameter instruction-tuned model from Alibaba. This model provides a strong foundation for instruction-following while being small enough to train on a single GPU.

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha (lora_alpha) | 64 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Dropout | 0.05 |
| Bias | none |
| **Trainable parameters** | **4,358,144 / 1,548,072,448 (0.28%)** |

By targeting all four attention projection matrices with rank-16 adapters, we achieve sufficient expressiveness while training only 0.28% of the total model parameters.

### Training Algorithm

GRPO (Group Relative Policy Optimization) generates multiple candidate completions per prompt, scores them with reward functions, and updates the policy to favor higher-reward outputs. We use TRL's `GRPOTrainer` with two reward functions:

- **Format reward (0.1):** Awarded when the output contains both `</think>` and `<answer>...</answer>` tags.
- **Correctness reward (1.0):** Awarded when the equation inside `<answer>` tags evaluates to the target and uses only the available numbers, each at most once. Evaluation uses a safe AST-based parser that only permits `+`, `-`, `*`, `/` operators.

### Dataset

We use the **Jiayi-Pan/Countdown-Tasks-3to4** dataset from HuggingFace, containing **441,327 training examples** (90/10 train/test split). Each example provides 3-4 numbers and a target value, formatted with the exact TinyZero prompt template:

```
A conversation between User and Assistant. The user asks a question,
and the Assistant solves it. The assistant first thinks about the reasoning
process in the mind and then provides the user with the answer.
User: Using the numbers [1, 2, 3, 4], create an equation that equals 10.
You can use +, -, *, / and each number at most once.
Assistant: <think>
```

### Training Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 5e-6 |
| Batch size | 4 per device |
| Gradient accumulation steps | 4 |
| Max completion length | 256 tokens |
| Generations per prompt | 4 |
| Temperature | 1.0 |
| Max steps | 250 |
| Precision | bfloat16 |

### Libraries

| Library | Version |
|---|---|
| PEFT | 0.18.1 |
| TRL | 1.0.0 |
| Transformers | 5.4.0 |
| PyTorch | 2.2+ |

## 3. Infrastructure and Setup

**Local development** was done on a Windows 11 laptop using Cursor IDE. Local dry-run mode (`--dry-run`) runs 20 steps on CPU to verify the pipeline before submitting GPU jobs.

**Cloud GPU training** used **Modal.com** with an **NVIDIA A100 GPU (40GB VRAM)**. Modal was chosen for its Python-native SDK, persistent volumes for checkpoint storage, and support for long-running detached jobs. The training script, data preprocessing, and evaluation all run inside a Modal container with dependencies installed via `pip_install` in the image definition.

## 4. Challenges and Solutions

### Challenge 1: TRL 1.0 API Breaking Changes

TRL 1.0 introduced significant API changes from prior versions. `GRPOConfig` no longer accepts `max_prompt_length`, and reward functions now require the signature `(prompts, completions, completion_ids, **kwargs)` where dataset columns are passed as keyword arguments.

**Solution:** Inspected TRL source code to identify the new API surface. Updated all reward function signatures and removed deprecated parameters from the training config.

### Challenge 2: Modal API Changes

Modal 1.4 removed `modal.Mount.from_local_dir()`, which was used in reference implementations.

**Solution:** Switched to `Image.add_local_dir()`, which bakes local project code directly into the container image at build time.

### Challenge 3: Training Not Saving Before Timeout

The first two training runs hit Modal's timeout (2 hours, then 3 hours). The training script wrote checkpoints to the container's local disk and only copied them to the persistent volume after training completed. When the container was killed at timeout, all progress was lost.

**Solution:** Changed `output_dir` to write directly to `/root/outputs/`, which is mounted to the Modal Volume. Every `save_steps` checkpoint now persists immediately to durable storage during training, not after.

### Challenge 4: GPU Preemption

Modal's A10G spot instance was preempted at step 88/100 during an early training run, killing the process mid-training.

**Solution:** Three mitigations: (1) switched to A100 GPU for more stable allocation, (2) reduced `save_steps` to 25 so checkpoints are frequent, and (3) added automatic resume logic in the Modal wrapper that detects the latest `checkpoint-*` directory and passes `--checkpoint` to resume training from where it stopped.

### Challenge 5: Determining Number of Training Steps

Full training on 441K examples for 3 epochs would require 661,989 steps, estimated at ~10,000 GPU-hours. Even 500 steps takes ~8 hours on an A10G.

**Solution:** Set `max_steps=250`, which completes in approximately 4 hours on A100. This is sufficient to observe the reward signal improving and demonstrate that the LoRA training pipeline works end-to-end. TinyZero used many more steps, but 250 is adequate for a proof-of-concept within budget and time constraints.

### Challenge 6: Python Environment Conflicts on Windows

Two Python installations (Anaconda and standalone Python 3.12) caused `pip install` to target one Python while the shell invoked the other at runtime.

**Solution:** Used the full Anaconda Python path (`/c/Users/varun2002/anaconda3/python.exe`) for all local runs. Modal handles the Linux environment cleanly inside the container, avoiding this issue entirely for GPU runs.

## 5. Training Results

Training completed 250 steps across 2 sessions with automatic checkpoint resume.

### Reward Progression

| Step | Total Reward | Format Reward | Correctness Reward |
|---|---|---|---|
| 10 | 0.07 | 0.07 | 0.00 |
| 50 | 0.10 | 0.10 | 0.00 |
| 100 | 0.13 | 0.10 | 0.03 |
| 200 | 0.15 | 0.10 | 0.05 |
| 250 | 0.16 | 0.10 | 0.06 |

### Evaluation Results (200 test samples)

| Metric | Value |
|---|---|
| Accuracy | 0.5% (1/200) |
| Format compliance | 100% (200/200) |
| Avg response length | 262 characters |

The model achieved perfect format compliance, always producing `<think>...</think>` and `<answer>...</answer>` tags. The correctness reward grew from 0.0 to 0.06, showing the model was beginning to learn arithmetic reasoning.

## 6. Why Accuracy Is Low and Why That Is Okay

At 250 steps with a batch size of 4 and 4 gradient accumulation steps, the model has seen roughly 4,000 examples — less than **0.1% of one epoch** of the full dataset. TinyZero's original experiments used thousands of steps to achieve meaningful accuracy on this task.

The goal of Part 1 is not to achieve a specific accuracy target. It is to **reproduce the training process with LoRA adapters** and demonstrate that:

1. The reward signal is clearly improving over training.
2. The model learns the output format completely (100% compliance).
3. The correctness reward shows a positive trend (0.0 to 0.06).
4. The full pipeline — data preprocessing, GRPO training, checkpoint saving, evaluation, and inference — works end-to-end.

With additional compute budget and more training steps, accuracy would continue to improve following the established trend.

## 7. Inference Demo

We built two inference scripts for live testing:

- **`inference.py`**: Interactive local inference. Accepts comma-separated numbers and a target, generates a response, and validates correctness using the same reward functions from training.
- **`modal_inference.py`**: Modal GPU wrapper that runs single-query inference on an A100.

Example interaction:

```
Enter numbers (comma separated) and target, e.g. '1,2,3,4 target=10': 1,2,3 target=5

Numbers: [1, 2, 3], Target: 5
Generating...

--- Model Output ---
<think>I need to find an equation using 1, 2, and 3 that equals 5.
Let me try: 2 + 3 = 5, and I haven't used 1.
What about 1 * (2 + 3) = 5? Yes!</think>
<answer>1 * (2 + 3)</answer>
--- End Output ---

Result: CORRECT  |  1 * (2 + 3) = 5
```

## 8. Project Structure

```
tinyzero-lora/
├── configs/
│   └── lora_config.yaml          # LoRA and training hyperparameters
├── part1_tinyzero/
│   ├── data/
│   │   └── countdown.py          # Dataset download and preprocessing
│   ├── reward.py                 # Format and correctness reward functions
│   ├── train_lora.py             # Main LoRA + GRPO training script
│   ├── evaluate.py               # Test set evaluation script
│   ├── inference.py              # Interactive local inference
│   ├── modal_train.py            # Modal wrapper for GPU training
│   └── modal_inference.py        # Modal wrapper for GPU inference
├── outputs/                      # Checkpoints and evaluation results
└── REPORT.md                     # This report
```

## 9. How to Reproduce

**Prerequisites:** Python 3.11+, a Modal account (for GPU training).

```bash
# 1. Install dependencies
pip install peft trl transformers accelerate datasets torch pyyaml

# 2. Preprocess the dataset
python part1_tinyzero/data/countdown.py

# 3. Local dry-run (CPU, 20 steps, verifies pipeline)
python part1_tinyzero/train_lora.py --dry-run

# 4. Full training on Modal GPU
pip install modal
modal setup
modal run part1_tinyzero/modal_train.py

# 5. Download trained checkpoint
modal run part1_tinyzero/modal_train.py --action download

# 6. Evaluate
python part1_tinyzero/evaluate.py --checkpoint_path outputs/lora_checkpoint/final_adapter

# 7. Interactive inference
python part1_tinyzero/inference.py --checkpoint_path outputs/lora_checkpoint/final_adapter

# 8. Modal inference (single query)
modal run part1_tinyzero/modal_inference.py --numbers "1,2,3,4" --target 10
```

## 10. Conclusion

We successfully reproduced the TinyZero countdown task training pipeline using LoRA adapters and GRPO. The key findings:

- **LoRA reduces trainable parameters by 99.72%** (4.4M out of 1.5B) while still enabling the model to learn from the GRPO reward signal.
- **Format compliance reached 100%**, demonstrating the model fully learned the expected output structure.
- **The correctness reward improved from 0.0 to 0.06** over 250 steps, confirming the model was beginning to learn arithmetic reasoning.
- **Infrastructure challenges** — API migrations, timeout management, checkpoint persistence, and GPU preemption — consumed more development time than the ML implementation itself.

With more training steps and compute budget, the model would continue improving in accuracy. Part 2 of this project will explore extended training and further optimizations.
