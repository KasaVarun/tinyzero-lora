"""
Reward Functions for Countdown Task

Implements reward computation for GRPO training:
- Correctness reward: 1.0 if the equation evaluates to the target
- Format reward: 0.1 if output uses <think>...</think> and <answer>...</answer> tags
"""

import re
import ast
import operator


# Allowed operators for safe eval
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expr: str) -> float:
    """Safely evaluate a math expression string.

    Only allows +, -, *, / and numeric literals. No function calls,
    imports, or attribute access.

    Args:
        expr: A string math expression like "3 + 4 * 2".

    Returns:
        The numeric result of the expression.

    Raises:
        ValueError: If the expression contains disallowed operations.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ValueError(f"Invalid expression: {expr}")

    def _eval(node):
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed operator: {op_type.__name__}")
            left = _eval(node.left)
            right = _eval(node.right)
            return ALLOWED_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Disallowed operator: {op_type.__name__}")
            return ALLOWED_OPERATORS[op_type](_eval(node.operand))
        elif isinstance(node, ast.Num):
            # Fallback for older Python versions
            return node.n
        else:
            raise ValueError(f"Disallowed node type: {type(node).__name__}")

    return _eval(tree)


def extract_answer(text: str) -> str | None:
    """Extract the equation from <answer>...</answer> tags.

    Args:
        text: The model's full output text.

    Returns:
        The equation string, or None if no answer tags found.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_numbers_from_equation(equation: str) -> list[float]:
    """Extract all numeric literals from an equation string.

    Args:
        equation: A math expression string.

    Returns:
        List of numbers found in the equation.
    """
    # Match integers and floats
    numbers = re.findall(r"\d+\.?\d*", equation)
    return [float(n) for n in numbers]


def check_numbers_used(equation: str, available_numbers: list) -> bool:
    """Verify that only the provided numbers are used, each at most once.

    Args:
        equation: The equation string to check.
        available_numbers: The list of numbers that can be used.

    Returns:
        True if the equation only uses available numbers (each at most once).
    """
    used_numbers = extract_numbers_from_equation(equation)
    available = [float(n) for n in available_numbers]

    # Check each used number is available
    remaining = available.copy()
    for num in used_numbers:
        if num in remaining:
            remaining.remove(num)
        else:
            return False
    return True


def parse_numbers_from_prompt(prompt_numbers: str) -> list:
    """Parse the numbers string from the prompt (e.g., '[1, 2, 3, 4]').

    Args:
        prompt_numbers: String representation of the numbers list.

    Returns:
        List of numbers.
    """
    try:
        # Handle string like "[1, 2, 3, 4]"
        nums = ast.literal_eval(prompt_numbers)
        if isinstance(nums, (list, tuple)):
            return [float(n) for n in nums]
        return [float(nums)]
    except (ValueError, SyntaxError):
        # Try extracting numbers directly
        numbers = re.findall(r"\d+\.?\d*", prompt_numbers)
        return [float(n) for n in numbers]


def format_reward_func(prompts, completions, completion_ids=None, **kwargs) -> list[float]:
    """Reward function that checks if the output has correct format.

    Awards 0.1 if the completion contains both <think>...</think> and
    <answer>...</answer> tags.

    Args:
        prompts: List of prompt strings (from GRPOTrainer).
        completions: List of model completion strings (from GRPOTrainer).
        completion_ids: List of token ID lists (unused).
        **kwargs: Extra dataset columns passed by GRPOTrainer.

    Returns:
        List of reward floats (0.0 or 0.1).
    """
    rewards = []
    for completion in completions:
        text = completion if isinstance(completion, str) else str(completion)
        has_think = bool(re.search(r"</think>", text))
        has_answer = bool(
            re.search(r"<answer>.*?</answer>", text, re.DOTALL)
        )
        rewards.append(0.1 if (has_think and has_answer) else 0.0)
    return rewards


def correctness_reward_func(prompts, completions, completion_ids=None, **kwargs) -> list[float]:
    """Reward function that checks if the equation is correct.

    Awards 1.0 if the equation evaluates to the target and uses only
    the available numbers (each at most once).

    Args:
        prompts: List of prompt strings (from GRPOTrainer).
        completions: List of model completion strings (from GRPOTrainer).
        completion_ids: List of token ID lists (unused).
        **kwargs: Must contain 'target' and 'numbers' lists from the dataset.

    Returns:
        List of reward floats (0.0 or 1.0).
    """
    target = kwargs.get("target", [])
    numbers = kwargs.get("numbers", [])
    rewards = []
    for i, completion in enumerate(completions):
        try:
            text = completion if isinstance(completion, str) else str(completion)
            equation = extract_answer(text)
            if equation is None:
                rewards.append(0.0)
                continue

            # Get target and available numbers for this example
            tgt = float(target[i])
            avail = parse_numbers_from_prompt(str(numbers[i]))

            # Check numbers constraint
            if not check_numbers_used(equation, avail):
                rewards.append(0.0)
                continue

            # Evaluate the equation
            result = safe_eval(equation)

            # Check if result matches target (with small tolerance for floats)
            if abs(result - tgt) < 1e-6:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


if __name__ == "__main__":
    # Quick test
    test_prompts = ["prompt1", "prompt2", "prompt3"]
    test_completions = [
        "Let me think... </think>\n<answer>1 + 2 + 3</answer>",
        "No tags here, just 1 + 2 + 3 = 6",
        "</think>\n<answer>invalid</answer>",
    ]
    test_targets = [6, 6, 6]
    test_numbers = ["[1, 2, 3]", "[1, 2, 3]", "[1, 2, 3]"]

    print("Format rewards:", format_reward_func(test_prompts, test_completions))
    print(
        "Correctness rewards:",
        correctness_reward_func(
            test_prompts, test_completions, target=test_targets, numbers=test_numbers
        ),
    )
