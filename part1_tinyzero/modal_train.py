"""
Modal GPU Training Wrapper

Runs the LoRA GRPO training and evaluation on Modal cloud GPUs.
Persists checkpoints to a Modal Volume for later download.

Usage:
    modal run part1_tinyzero/modal_train.py                          # train
    modal run part1_tinyzero/modal_train.py --action evaluate        # evaluate
    modal run part1_tinyzero/modal_train.py --action download        # download outputs
"""

import modal

# Define the Modal app
app = modal.App("tinyzero-lora")

# Create a persistent volume for outputs
volume = modal.Volume.from_name("tinyzero-lora-outputs", create_if_missing=True)

# Build the container image with deps + local project code baked in
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "peft>=0.10.0",
        "trl>=0.8.0",
        "transformers>=4.40.0",
        "accelerate>=0.29.0",
        "datasets>=2.19.0",
        "torch>=2.2.0",
        "numpy",
        "pandas",
        "pyyaml",
    )
    .add_local_dir(".", remote_path="/root/tinyzero-lora", copy=True)
)


def _run(cmd: list[str], cwd: str):
    """Run a subprocess, inheriting the full environment.

    Args:
        cmd: Command and arguments to run.
        cwd: Working directory.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code.
    """
    import subprocess
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root/tinyzero-lora/part1_tinyzero"

    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}")


@app.function(
    image=image,
    gpu="a100",
    timeout=10800,
    volumes={"/root/outputs": volume},
    # Uncomment if you have a HuggingFace secret configured in Modal:
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    """Run GRPO training with LoRA on a Modal GPU.

    Preprocesses data if needed, then runs the full training pipeline.
    Saves checkpoints to the persistent Modal Volume.
    """
    from pathlib import Path

    project_dir = "/root/tinyzero-lora"
    data_dir = Path(project_dir) / "part1_tinyzero" / "data" / "processed"
    volume_output = "/root/outputs/lora_checkpoint"

    # Step 1: Preprocess data if needed
    if not (data_dir / "train.json").exists():
        print("=" * 60)
        print("PREPROCESSING DATA")
        print("=" * 60)
        _run(
            ["python", "part1_tinyzero/data/countdown.py"],
            cwd=project_dir,
        )

    # Step 2: Check for existing checkpoint to resume from
    volume_output_path = Path(volume_output)
    train_cmd = [
        "python", "part1_tinyzero/train_lora.py",
        "--data_dir", str(data_dir),
        "--config", f"{project_dir}/configs/lora_config.yaml",
        "--output_dir", volume_output,
    ]

    checkpoints = sorted(
        [d for d in volume_output_path.glob("checkpoint-*") if d.is_dir()],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if checkpoints:
        latest = checkpoints[-1]
        print(f"Resuming from checkpoint: {latest}")
        train_cmd += ["--checkpoint", str(latest)]
    else:
        print("No existing checkpoint found, starting fresh.")

    # Step 3: Run training (writes directly to the Modal Volume)
    print("=" * 60)
    print("STARTING GRPO TRAINING WITH LoRA")
    print("=" * 60)
    _run(train_cmd, cwd=project_dir)

    volume.commit()
    print("Training complete! Outputs persisted to volume.")


@app.function(
    image=image,
    gpu="a100",
    timeout=3600,
    volumes={"/root/outputs": volume},
    # Uncomment if you have a HuggingFace secret configured in Modal:
    # secrets=[modal.Secret.from_name("huggingface-secret")],
)
def evaluate(num_samples: int = 200):
    """Run evaluation on the test set using the trained adapter.

    Args:
        num_samples: Number of test samples to evaluate.
    """
    import shutil
    from pathlib import Path

    project_dir = "/root/tinyzero-lora"
    data_dir = Path(project_dir) / "part1_tinyzero" / "data" / "processed"
    checkpoint_path = Path("/root/outputs/lora_checkpoint/final_adapter")

    # Preprocess data if needed
    if not (data_dir / "test.json").exists():
        print("Preprocessing data...")
        _run(
            ["python", "part1_tinyzero/data/countdown.py"],
            cwd=project_dir,
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run training first."
        )

    print(f"Running evaluation with {num_samples} samples...")
    _run(
        [
            "python", "part1_tinyzero/evaluate.py",
            "--checkpoint_path", str(checkpoint_path),
            "--num_samples", str(num_samples),
            "--data_dir", str(data_dir),
        ],
        cwd=project_dir,
    )

    # Copy eval results to volume
    eval_src = Path(project_dir) / "outputs" / "eval_results.json"
    eval_dst = Path("/root/outputs/eval_results.json")
    if eval_src.exists():
        shutil.copy2(eval_src, eval_dst)
        volume.commit()
        print("Evaluation results saved to volume.")


@app.function(
    image=image,
    volumes={"/root/outputs": volume},
)
def download():
    """Download the trained checkpoint from Modal Volume.

    Returns:
        Dict mapping file paths to file contents (bytes).
    """
    from pathlib import Path

    output_dir = Path("/root/outputs")
    files = {}

    for f in output_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(output_dir)
            files[str(rel_path)] = f.read_bytes()

    print(f"Prepared {len(files)} files for download")
    return files


@app.local_entrypoint()
def main(action: str = "train", num_samples: int = 200, download_outputs: bool = False):
    """Local entrypoint for Modal CLI.

    Args:
        action: One of 'train', 'evaluate', or 'download'.
        num_samples: Number of samples for evaluation.
        download_outputs: Whether to download outputs after training.
    """
    from pathlib import Path

    if action == "train":
        print("Launching training on Modal...")
        train.remote()
        print("Training complete!")

        if download_outputs:
            print("Downloading outputs...")
            files = download.remote()
            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            for rel_path, content in files.items():
                out_path = output_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(content)
            print(f"Downloaded {len(files)} files to {output_dir}")

    elif action == "evaluate":
        print(f"Launching evaluation on Modal ({num_samples} samples)...")
        evaluate.remote(num_samples=num_samples)
        print("Evaluation complete!")

    elif action == "download":
        print("Downloading outputs from Modal volume...")
        files = download.remote()
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, content in files.items():
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)
        print(f"Downloaded {len(files)} files to {output_dir}")

    else:
        print(f"Unknown action: {action}. Use 'train', 'evaluate', or 'download'.")
