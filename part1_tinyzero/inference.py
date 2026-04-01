"""
Interactive Inference for Countdown Task

Loads a trained LoRA adapter and runs interactive inference,
letting the user input numbers and a target to solve countdown puzzles.
"""

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


PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer.\n"
    "User: Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use +, -, *, / and each number at most once. \n"
    "Assistant: <think>"
)


def load_model(checkpoint_path: Path, base_model: str):
    """Load the base model with LoRA adapter merged in.

    Args:
        checkpoint_path: Path to the saved LoRA adapter directory.
        base_model: HuggingFace model ID for the base model.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a model response for a given prompt.

    Args:
        model: The language model with LoRA adapter.
        tokenizer: The tokenizer.
        prompt: The formatted input prompt.

    Returns:
        The generated text (completion only, excluding the prompt).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def check_answer(response: str, target: float, numbers: list[float]) -> tuple[str | None, bool]:
    """Check whether the model's answer is correct.

    Args:
        response: The model's generated text.
        target: The target number.
        numbers: The available numbers.

    Returns:
        Tuple of (extracted equation or None, whether it is correct).
    """
    equation = extract_answer(response)
    if equation is None:
        return None, False

    try:
        if not check_numbers_used(equation, numbers):
            return equation, False
        result = safe_eval(equation)
        return equation, abs(result - target) < 1e-6
    except Exception:
        return equation, False


def parse_input(user_input: str) -> tuple[list[float], float]:
    """Parse user input like '1,2,3,4 target=10' into numbers and target.

    Args:
        user_input: Raw input string from the user.

    Returns:
        Tuple of (list of numbers, target value).

    Raises:
        ValueError: If the input cannot be parsed.
    """
    parts = user_input.split("target=")
    if len(parts) != 2:
        raise ValueError("Expected format: '1,2,3,4 target=10'")

    target = float(parts[1].strip())
    numbers = [float(n.strip()) for n in parts[0].strip().rstrip(",").split(",")]
    return numbers, target


def main():
    """Run interactive inference with the trained LoRA model."""
    parser = argparse.ArgumentParser(description="Interactive countdown inference")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="outputs/lora_checkpoint/final_adapter",
        help="Path to the LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent.parent / checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, tokenizer = load_model(checkpoint_path, args.base_model)
    print("\nModel loaded. Type 'quit' to exit.\n")

    while True:
        user_input = input(
            "Enter numbers (comma separated) and target, e.g. '1,2,3,4 target=10': "
        )
        if user_input.strip().lower() == "quit":
            print("Goodbye!")
            break

        try:
            numbers, target = parse_input(user_input)
        except ValueError as e:
            print(f"Parse error: {e}\n")
            continue

        numbers_str = str([int(n) if n == int(n) else n for n in numbers])
        prompt = PROMPT_TEMPLATE.format(numbers=numbers_str, target=int(target))

        print(f"\nNumbers: {numbers_str}, Target: {int(target)}")
        print("Generating...\n")

        response = generate_response(model, tokenizer, prompt)
        full_output = "<think>" + response

        print("--- Model Output ---")
        print(full_output)
        print("--- End Output ---\n")

        equation, correct = check_answer(response, target, numbers)
        if equation is None:
            print("Result: No <answer> tags found in response.")
        elif correct:
            print(f"Result: CORRECT  |  {equation} = {int(target)}")
        else:
            print(f"Result: INCORRECT  |  equation: {equation}")
        print()


if __name__ == "__main__":
    main()
