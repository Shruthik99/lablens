"""Build weak-label train/val data for fine-tuning Flan-T5 on medication extraction.

Pulls notes and GPT-generated structured summaries from AGBonnet/augmented-clinical-notes,
turning each summary's "treatments" entries into a linearized target (see t5_format.py)
the model should learn to produce from the (truncated) note text. Notes whose index is in
HELD_OUT_NOTE_INDEXES are always skipped, since they're reserved for evaluation.

Runnable as: python -m lablens.prepare_t5_data [--n 500] [--seed 42] [--val-frac 0.1] [--out data/t5]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from lablens.grounding import GroundingStats, ground_value, is_drug_grounded
from lablens.splits import HELD_OUT_NOTE_INDEXES
from lablens.t5_format import linearize_medications, parse_medications

_FLAN_T5_TOKENIZER = "google/flan-t5-base"
_MAX_INPUT_TOKENS = 768

_TRAILING_PARENTHETICAL_RE = re.compile(r"\(([^()]+)\)\s*$")


def _trailing_parenthetical(name: str) -> str | None:
    """Return the text inside a trailing "(...)" in name, or None if there isn't one."""
    match = _TRAILING_PARENTHETICAL_RE.search(name)
    return match.group(1).strip() if match else None


def _loose_json(value: Any) -> Any:
    """Parse value if it's a JSON string; pass dicts/lists through; return None on failure."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value, strict=False)
    except json.JSONDecodeError:
        return None


def _clean(value: Any) -> str | None:
    """Treat None, "None", and empty/whitespace strings as null; otherwise strip."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() == "none":
        return None
    return value


def build_target_medications(summary_raw: Any) -> list[dict[str, str | None]]:
    """Build the weak-label medication list from a (possibly string-encoded) summary."""
    summary = _loose_json(summary_raw)
    if not isinstance(summary, dict):
        return []

    treatments = summary.get("treatments")
    treatments = _loose_json(treatments) if isinstance(treatments, str) else treatments
    if not isinstance(treatments, list):
        return []

    medications = []
    for item in treatments:
        if not isinstance(item, dict):
            continue
        drug = _clean(item.get("name"))
        if drug is None:
            continue
        medications.append(
            {
                "drug": drug,
                "strength": _clean(item.get("dosage")),
                "frequency": _clean(item.get("frequency")),
                "duration": _clean(item.get("duration")),
            }
        )
    return medications


def ground_target_medications(
    medications: list[dict[str, str | None]],
    input_text: str,
    stats: GroundingStats | None = None,
) -> list[dict[str, str | None]]:
    """Keep a medication only if its drug name is grounded in input_text; null out any
    strength/frequency/duration value that isn't (reuses the extract_llm grounding checks,
    so weak labels are held to the same evidence standard as LLM extractions).

    Weak labels sometimes wrap a real drug name in a class/procedure description, e.g.
    "Corticosteroids (prednisone)". If the full name doesn't ground, retry on the text
    inside a trailing "(...)" and keep the medication under that name if it grounds.
    """
    grounded = []
    for med in medications:
        drug = med["drug"]
        if not is_drug_grounded(drug, input_text):
            inner = _trailing_parenthetical(drug)
            if inner and is_drug_grounded(inner, input_text):
                drug = inner
                if stats is not None:
                    stats.rescued_drugs += 1
            else:
                if stats is not None:
                    stats.ungrounded_drugs += 1
                continue
        grounded.append(
            {
                "drug": drug,
                "strength": ground_value(med.get("strength"), input_text, stats),
                "frequency": ground_value(med.get("frequency"), input_text, stats),
                "duration": ground_value(med.get("duration"), input_text, stats),
            }
        )
    return grounded


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens, as a safety net for longer notes."""
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _iter_examples(n: int, seed: int, stats: GroundingStats | None = None) -> list[dict[str, str]]:
    """Load the dataset, skip held-out notes, truncate and ground, and collect n examples."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_FLAN_T5_TOKENIZER)
    dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train")
    dataset = dataset.shuffle(seed=seed)

    examples = []
    for row in dataset:
        if row["idx"] in HELD_OUT_NOTE_INDEXES:
            continue
        medications = build_target_medications(row["summary"])
        if not medications:
            continue

        input_text = _truncate_to_tokens(
            tokenizer, "extract medications: " + row["note"], _MAX_INPUT_TOKENS
        )
        medications = ground_target_medications(medications, input_text, stats)
        if not medications:
            continue

        examples.append({"input": input_text, "target": linearize_medications(medications)})
        if len(examples) >= n:
            break
    return examples


def _report(examples: list[dict[str, str]], stats: GroundingStats) -> None:
    total_medications = sum(len(parse_medications(ex["target"]) or []) for ex in examples)
    avg_medications = total_medications / len(examples) if examples else 0.0

    print(f"Examples: {len(examples)}")
    print(f"Avg medications per example: {avg_medications:.2f}")
    print(f"Medications dropped (drug name ungrounded): {stats.ungrounded_drugs}")
    print(f"Medications rescued (grounded via trailing parenthetical): {stats.rescued_drugs}")
    print(f"Fields dropped (value ungrounded): {stats.ungrounded_values}")


def _write_jsonl(path: Path, examples: list[dict[str, str]]) -> None:
    """Write LF-only, UTF-8 JSONL so the same input hashes identically on every OS
    (an unspecified newline mode lets Windows translate "\\n" to "\\r\\n" on write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(example) + "\n" for example in examples)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare weak-label train/val data for fine-tuning Flan-T5."
    )
    parser.add_argument("--n", type=int, default=500, help="Number of examples (default: 500).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of examples held out for validation (default: 0.1).",
    )
    parser.add_argument(
        "--out", default="data/t5", help="Output directory for train.jsonl/val.jsonl."
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    stats = GroundingStats()
    examples = _iter_examples(args.n, args.seed, stats)
    _report(examples, stats)

    n_val = round(len(examples) * args.val_frac)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    out_dir = Path(args.out)
    _write_jsonl(out_dir / "train.jsonl", train_examples)
    _write_jsonl(out_dir / "val.jsonl", val_examples)
    print(f"Wrote {len(train_examples)} train / {len(val_examples)} val examples to {out_dir}")


if __name__ == "__main__":
    main()
