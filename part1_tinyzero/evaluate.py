"""
Evaluation Script for Countdown Task

Loads a trained LoRA adapter, runs inference on the test set,
and computes accuracy, format compliance, and response length metrics.
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from reward import (
    extract_answer,
    safe_eval,
    parse_numbers_from_prompt,
    check_numbers_used,
)


def load_test_data(data_dir: str | None = None) -> list[dict]:
    """Load processed test data from JSON.

    Args:
        data_dir: Path to directory containing test.json.

    Returns:
        List of test examples.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data" / "processed"
    else:
        data_dir = Path(data_dir)

    test_path = data_dir / "test.json"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Run 'python part1_tinyzero/data/countdown.py' first."
        )

    with open(test_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 1024) -> str:
    """Generate a response from the model for a given prompt.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: The input prompt string.
        max_new_tokens: Maximum tokens to generate.

    Returns:
        The generated text (completion only, not including prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    # Decode only the generated tokens (exclude prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def evaluate_response(response: str, target: float, numbers: list) -> dict:
    """Evaluate a single model response.

    Args:
        response: The model's generated text.
        target: The target number to reach.
        numbers: The available numbers.

    Returns:
        Dict with 'correct', 'has_format', and 'response_length' fields.
    """
    import re

    has_think = bool(re.search(r"</think>", response))
    has_answer = bool(re.search(r"<answer>.*?</answer>", response, re.DOTALL))
    has_format = has_think and has_answer

    correct = False
    equation = extract_answer(response)
    if equation is not None:
        try:
            if check_numbers_used(equation, numbers):
                result = safe_eval(equation)
                if abs(result - target) < 1e-6:
                    correct = True
        except Exception:
            pass

    return {
        "correct": correct,
        "has_format": has_format,
        "response_length": len(response),
    }


def main():
    """Run evaluation on the test set."""
    parser = argparse.ArgumentParser(description="Evaluate LoRA model on countdown task")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of test samples to evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load test data
    print("Loading test data...")
    test_data = load_test_data(args.data_dir)
    test_data = test_data[:args.num_samples]
    print(f"Evaluating on {len(test_data)} samples")

    # Load base model + LoRA adapter
    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    # Run evaluation
    results = []
    correct_count = 0
    format_count = 0
    total_length = 0

    for i, example in enumerate(test_data):
        prompt = example["prompt"]
        target = float(example["target"])
        numbers = parse_numbers_from_prompt(example["numbers"])

        response = generate_response(model, tokenizer, prompt)
        eval_result = evaluate_response(response, target, numbers)

        results.append({
            "prompt": prompt,
            "response": response,
            "target": target,
            "numbers": example["numbers"],
            **eval_result,
        })

        correct_count += int(eval_result["correct"])
        format_count += int(eval_result["has_format"])
        total_length += eval_result["response_length"]

        if (i + 1) % 20 == 0:
            acc = correct_count / (i + 1) * 100
            fmt = format_count / (i + 1) * 100
            print(f"[{i+1}/{len(test_data)}] Accuracy: {acc:.1f}%, Format: {fmt:.1f}%")

    # Compute final metrics
    n = len(test_data)
    accuracy = correct_count / n * 100
    format_compliance = format_count / n * 100
    avg_length = total_length / n

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:          {accuracy:.1f}% ({correct_count}/{n})")
    print(f"Format compliance: {format_compliance:.1f}% ({format_count}/{n})")
    print(f"Avg response len:  {avg_length:.0f} chars")
    print("=" * 60)

    # Save results
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output = {
        "accuracy": accuracy,
        "format_compliance": format_compliance,
        "avg_response_length": avg_length,
        "num_samples": n,
        "checkpoint_path": str(checkpoint_path),
        "details": results,
    }
    eval_path = output_dir / "eval_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {eval_path}")


if __name__ == "__main__":
    main()
