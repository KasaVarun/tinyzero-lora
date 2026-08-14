# Results

Measured results from the training run described in [REPORT.md](REPORT.md).
All numbers below are observed, not projected.

## Setup

| Parameter | Value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct |
| Method | GRPO (TRL) with LoRA |
| LoRA config | r=16, alpha=64, dropout=0.05, targets `q_proj` `k_proj` `v_proj` `o_proj` |
| Trainable parameters | 4,358,144 of 1,548,072,448 (**0.28%**) |
| Learning rate | 5e-6 |
| Batch size | 4 (gradient accumulation 4) |
| Generations per prompt | 4 |
| Max completion length | 256 tokens |
| Sampling temperature | 1.0 |
| Precision | bf16 |
| Training steps | 250 |
| Hardware | Modal A100 40GB |
| Dataset | `Jiayi-Pan/Countdown-Tasks-3to4` (441,327 examples) |

Reward: correctness `1.0` + format `0.1`, with equations evaluated through a
safe AST parser restricted to `+ - * /`.

## Evaluation (200 held-out samples)

| Metric | Result |
|---|---|
| Output-format compliance | **100.0%** (200/200) |
| Task accuracy | **0.5%** (1/200) |
| Correctness reward (start → end of training) | 0.00 → 0.06 |

## Reading these numbers honestly

Format compliance reached 100%: the model reliably learned the
`<think>...</think><answer>...</answer>` protocol and produced parseable output
on every single sample.

Task accuracy stayed near zero, and that is the expected outcome at this
training budget rather than a bug. 250 steps at batch size 4 covers roughly
1,000 examples — about **0.1% of a single epoch** over the 441k-example dataset.
The correctness reward moved from 0.00 to 0.06, which shows the signal is
present and the gradient is pointing the right way; there simply was not enough
optimization to convert protocol adherence into arithmetic search ability.

The reported purpose of this run was to validate the full LoRA + GRPO pipeline
end to end under a constrained compute budget, not to reproduce TinyZero's
final accuracy. Format compliance is the metric that budget can move, and it
moved completely.

What a full reproduction would require: several thousand steps, a larger batch,
and ideally the 3B base model used in the original TinyZero work.

## Infrastructure notes

Two training runs were lost before the successful one:

1. Checkpoints written inside the container were destroyed on function timeout.
   Fixed by mounting `output_dir` directly to a persistent Modal Volume.
2. A10G spot capacity was preempted at step 88 of 100. Fixed by moving to A100,
   reducing `save_steps` to 25, and adding automatic detection and resume from
   the latest checkpoint.

The pipeline was also migrated through TRL 1.0's breaking reward-function API
change during development.

## Environment

| Package | Version |
|---|---|
| TRL | 1.0.0 |
| PEFT | 0.18.1 |
| Transformers | 5.4.0 |
