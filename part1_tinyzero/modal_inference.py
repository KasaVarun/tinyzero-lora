"""
Modal GPU Inference Wrapper for Countdown Task

Runs inference on Modal with an A100 GPU. Accepts numbers and target
as arguments and returns the model's response with correctness check.

Usage:
    modal run part1_tinyzero/modal_inference.py --numbers "1,2,3,4" --target 10
"""

import modal

app = modal.App("tinyzero-lora-inference")

volume = modal.Volume.from_name("tinyzero-lora-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "peft>=0.10.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "torch>=2.2.0",
    )
    .add_local_dir(".", remote_path="/root/tinyzero-lora", copy=True)
)


@app.function(
    image=image,
    gpu="a100",
    timeout=600,
    volumes={"/root/outputs": volume},
)
def solve(numbers: list[float], target: float, base_model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> dict:
    """Run inference on Modal to solve a single countdown puzzle.

    Args:
        numbers: List of available numbers.
        target: The target value to reach.
        base_model: HuggingFace model ID for the base model.

    Returns:
        Dict with keys: 'response', 'equation', 'correct', 'numbers', 'target'.
    """
    import sys
    sys.path.insert(0, "/root/tinyzero-lora/part1_tinyzero")

    from pathlib import Path
    from inference import load_model, generate_response, check_answer, PROMPT_TEMPLATE

    checkpoint_path = Path("/root/outputs/lora_checkpoint/final_adapter")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run training first."
        )

    model, tokenizer = load_model(checkpoint_path, base_model)

    numbers_str = str([int(n) if n == int(n) else n for n in numbers])
    prompt = PROMPT_TEMPLATE.format(numbers=numbers_str, target=int(target))

    response = generate_response(model, tokenizer, prompt)
    full_output = "<think>" + response
    equation, correct = check_answer(response, target, numbers)

    return {
        "response": full_output,
        "equation": equation,
        "correct": correct,
        "numbers": numbers,
        "target": target,
    }


@app.local_entrypoint()
def main(numbers: str = "1,2,3,4", target: int = 10):
    """Local entrypoint for Modal CLI.

    Args:
        numbers: Comma-separated list of numbers (e.g. "1,2,3,4").
        target: The target value to reach.
    """
    nums = [float(n.strip()) for n in numbers.split(",")]
    print(f"Numbers: {nums}, Target: {target}")
    print("Running inference on Modal...\n")

    result = solve.remote(numbers=nums, target=float(target))

    print("--- Model Output ---")
    print(result["response"])
    print("--- End Output ---\n")

    if result["equation"] is None:
        print("Result: No <answer> tags found in response.")
    elif result["correct"]:
        print(f"Result: CORRECT  |  {result['equation']} = {target}")
    else:
        print(f"Result: INCORRECT  |  equation: {result['equation']}")
