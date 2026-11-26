"""Fine-tune Flan-T5 on the weak-label medication data from prepare_t5_data.py.

Runnable as: python -m lablens.train_t5 [--data data/t5] [--model google/flan-t5-base]
    [--out models/flan-t5] [--max-input 768] [--max-target 256] [--epochs 3] [--batch 8]
    [--grad-accum 2] [--lr 3e-4] [--warmup 0.06] [--seed 42] [--limit N] [--max-steps N]
    [--fp16] [--no-gradient-checkpointing] [--git-commit SHA]

Trains and evaluates only on the local machine/GPU; no document text or data leaves it.

Trains in fp32 with the Adafactor optimizer by default: fp16 made T5 loss diverge to NaN
in practice, so it's opt-in via --fp16. Gradient checkpointing is on by default to keep
fp32 training within a 16 GB T4's memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Flan-T5 on weak-label medication extraction data."
    )
    parser.add_argument("--data", default="data/t5", help="Directory with train.jsonl/val.jsonl.")
    parser.add_argument("--model", default="google/flan-t5-base", help="Base model to fine-tune.")
    parser.add_argument("--out", default="models/flan-t5", help="Output directory for the model.")
    parser.add_argument("--max-input", type=int, default=768, help="Max input tokens.")
    parser.add_argument("--max-target", type=int, default=256, help="Max target tokens.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Per-device train/eval batch size.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.06, help="Warmup ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Use only N training examples (smoke test)."
    )
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Stop after N steps (smoke test)."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Train in fp16 (opt-in: this diverged for T5 in practice). Ignored on CPU.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade compute for memory (default: on).",
    )
    parser.add_argument(
        "--git-commit",
        default=None,
        help="Override the detected git commit (e.g. when running outside the repo, like Kaggle).",
    )
    return parser.parse_args(argv if argv is not None else sys.argv[1:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_commit() -> str | None:
    """Return the current git commit hash, or None if git isn't available / this isn't a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    return result.stdout.strip()


def build_run_info(
    args: argparse.Namespace,
    train_path: Path,
    val_path: Path,
    device: str,
    fp16: bool,
    git_commit: str | None,
    train_loss: float | None,
    val_loss: float | None,
    training_seconds: float,
) -> dict[str, Any]:
    return {
        "settings": {
            "data": str(args.data),
            "model": args.model,
            "out": str(args.out),
            "max_input": args.max_input,
            "max_target": args.max_target,
            "epochs": args.epochs,
            "batch": args.batch,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "warmup": args.warmup,
            "seed": args.seed,
            "limit": args.limit,
            "max_steps": args.max_steps,
            "fp16": fp16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "git_commit_override": args.git_commit,
        },
        "data_sha256": {
            "train.jsonl": sha256_file(train_path),
            "val.jsonl": sha256_file(val_path),
        },
        "git_commit": git_commit,
        "device": device,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "training_seconds": training_seconds,
    }


def write_run_info(path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = args.fp16 and device == "cuda"

    data_dir = Path(args.data)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    train_records = _load_jsonl(train_path)
    val_records = _load_jsonl(val_path)
    if args.limit is not None:
        train_records = train_records[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[int]]:
        model_inputs = tokenizer(batch["input"], max_length=args.max_input, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=args.max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_list(train_records).map(
        tokenize, batched=True, remove_columns=["input", "target"]
    )
    val_dataset = Dataset.from_list(val_records).map(
        tokenize, batched=True, remove_columns=["input", "target"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    out_dir = Path(args.out)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # This transformers version folds warmup_ratio into warmup_steps: a value < 1 is
        # treated as a ratio of total steps, >= 1 as an absolute step count.
        warmup_steps=args.warmup,
        fp16=fp16,
        optim="adafactor",
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=args.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    start = time.monotonic()
    train_result = trainer.train()
    training_seconds = time.monotonic() - start

    eval_metrics = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    git_commit = args.git_commit or get_git_commit()
    info = build_run_info(
        args,
        train_path,
        val_path,
        device,
        fp16,
        git_commit,
        train_result.training_loss,
        eval_metrics.get("eval_loss"),
        training_seconds,
    )
    write_run_info(out_dir / "run_info.json", info)

    print(f"Device: {device} (fp16={fp16})")
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Val loss: {eval_metrics.get('eval_loss'):.4f}")
    print(f"Training time: {training_seconds:.1f}s")
    print(f"Saved model + tokenizer + run_info.json to {out_dir}")


if __name__ == "__main__":
    main()
