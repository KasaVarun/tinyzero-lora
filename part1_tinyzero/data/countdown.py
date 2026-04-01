"""
Countdown Task Data Preprocessing

Loads the Jiayi-Pan/Countdown-Tasks-3to4 dataset from HuggingFace,
formats each example with the TinyZero prompt template, and saves
train/test splits as JSON files.
"""

import json
import argparse
from pathlib import Path
from datasets import load_dataset


PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer.\n"
    "User: Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use +, -, *, / and each number at most once. \n"
    "Assistant: <think>"
)


def format_example(example):
    """Format a single countdown example into the TinyZero prompt format.

    Args:
        example: A dataset row with 'nums' and 'target' fields.

    Returns:
        Dict with 'prompt', 'numbers', and 'target' fields.
    """
    nums = example["nums"]
    target = example["target"]

    # Format numbers as a list string, e.g. "[1, 2, 3, 4]"
    if isinstance(nums, str):
        numbers_str = nums
    else:
        numbers_str = str(nums)

    prompt = PROMPT_TEMPLATE.format(numbers=numbers_str, target=target)

    return {
        "prompt": prompt,
        "numbers": numbers_str,
        "target": target,
    }


def main():
    """Load dataset, format examples, split, and save to disk."""
    parser = argparse.ArgumentParser(description="Preprocess Countdown dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for test split",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset: Jiayi-Pan/Countdown-Tasks-3to4")
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")

    # The dataset may have a single 'train' split; we split it ourselves
    if "train" in dataset:
        full_data = dataset["train"]
    else:
        # Use the first available split
        split_name = list(dataset.keys())[0]
        full_data = dataset[split_name]

    print(f"Total examples: {len(full_data)}")

    # Format all examples
    formatted = [format_example(example) for example in full_data]

    # Split into train/test
    split_idx = int(len(formatted) * (1 - args.test_ratio))
    train_data = formatted[:split_idx]
    test_data = formatted[split_idx:]

    print(f"Train examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")

    # Save to JSON
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Saved train data to {train_path}")
    print(f"Saved test data to {test_path}")

    # Print a sample
    print("\n--- Sample prompt ---")
    print(formatted[0]["prompt"])
    print("--- End sample ---")


if __name__ == "__main__":
    main()
