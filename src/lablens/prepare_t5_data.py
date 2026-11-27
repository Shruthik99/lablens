"""Build weak-label train/val data for fine-tuning Flan-T5 on medication extraction.

Pulls notes and GPT-generated structured summaries from AGBonnet/augmented-clinical-notes,
turning each summary's "treatments" entries into a linearized target (see t5_format.py)
the model should learn to produce from the (truncated) note text. Notes whose index is in
HELD_OUT_NOTE_INDEXES are always skipped, since they're reserved for evaluation.

The weak label combines two sources over the same truncated note text: GPT-4's
"treatments" (source A, mapped/grounded below) and lablens.extract_rules.extract in its
default lexicon mode (source B). See merge_medications for how they're combined.

Runnable as: python -m lablens.prepare_t5_data [--n 500] [--seed 42] [--val-frac 0.1] [--out data/t5]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lablens import extract_rules
from lablens.grounding import GroundingStats, ground_value, is_drug_grounded, normalize
from lablens.schema import Medication
from lablens.splits import HELD_OUT_NOTE_INDEXES
from lablens.t5_format import linearize_medications, parse_medications

_FLAN_T5_TOKENIZER = "google/flan-t5-base"
_MAX_INPUT_TOKENS = 768

_TRAILING_PARENTHETICAL_RE = re.compile(r"\(([^()]+)\)\s*$")

# Source B runs in the same lexicon mode the rest of the project treats as its default.
_MERGE_LEXICON_MODE = extract_rules.DEFAULT_LEXICON_MODE

# Cap on how many rejected-drug examples _report prints.
_MAX_REJECTED_EXAMPLES = 15


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


@dataclass
class MergeStats:
    """Counts of how source A (GPT-4 treatments) and source B (rule extractor) combine."""

    gpt_only: int = 0
    rules_only: int = 0
    both: int = 0
    rejected_not_in_rxnorm: int = 0
    rejected_examples: list[str] = field(default_factory=list)
    with_route: int = 0
    # A rules-only medication (GPT-4 never mentioned it) kept only if a dose, route, or
    # medication verb appears nearby -- see Medication.has_context. This count is how many
    # were dropped for lacking that support; it does not affect extract_rules.extract()
    # itself, only which rules-only medications become training labels.
    rules_only_dropped_no_context: int = 0


def _dedupe_rule_medications(rule_meds: list[Medication]) -> list[Medication]:
    """Collapse repeated mentions of the same drug (one per line it appears on, e.g. in
    both history and discharge meds) to the first mention, by normalized name."""
    seen: set[str] = set()
    deduped = []
    for med in rule_meds:
        key = normalize(med.drug)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(med)
    return deduped


def _merged_from_both(
    gpt_med: dict[str, str | None], rule_med: Medication
) -> dict[str, str | None]:
    return {
        "drug": rule_med.drug,
        "strength": rule_med.strength if rule_med.strength is not None else gpt_med.get("strength"),
        "dosage": rule_med.dosage,
        "route": rule_med.route,
        "frequency": (
            rule_med.frequency if rule_med.frequency is not None else gpt_med.get("frequency")
        ),
        "duration": rule_med.duration if rule_med.duration is not None else gpt_med.get("duration"),
        "sources": ("gpt", "rules"),
    }


def _from_gpt_only(gpt_med: dict[str, str | None]) -> dict[str, str | None]:
    return {
        "drug": gpt_med["drug"],
        "strength": gpt_med.get("strength"),
        "dosage": None,
        "route": None,
        "frequency": gpt_med.get("frequency"),
        "duration": gpt_med.get("duration"),
        "sources": ("gpt",),
    }


def _from_rules_only(rule_med: Medication) -> dict[str, str | None] | None:
    """Build a training-label record for a rules-only medication, or None to drop it.

    GPT-4 never mentioned this drug, so unlike a rules match that's merged with or fills
    in for a GPT-4 mention, there's no second source vouching for it: it's kept as a
    training label only if a dose, route, or medication verb appears nearby (see
    Medication.has_context). This is stricter than extract_rules.extract() itself, which
    keeps it regardless -- that's deliberate here, training labels only.
    """
    if not rule_med.has_context:
        return None
    return {
        "drug": rule_med.drug,
        "strength": rule_med.strength,
        "dosage": rule_med.dosage,
        "route": rule_med.route,
        "frequency": rule_med.frequency,
        "duration": rule_med.duration,
        "sources": ("rules",),
    }


def merge_medications(
    gpt_meds: list[dict[str, str | None]],
    rule_meds: list[Medication],
    stats: MergeStats | None = None,
) -> list[dict[str, str | None]]:
    """Combine source A (GPT-4 treatments, already grounded) and source B (rule extractor
    output, default lexicon mode) into one medication list.

    - in both (normalized name or same RxCUI): keep source B's attributes; fill a field
      from A only where B has none (A's values are already grounded by the caller)
    - only in A: keep only if its name is in the RxNorm lexicon and not ambiguous-only
    - only in B: keep it as is
    Each merged record carries "sources" (not part of the model target, see
    _project_for_target) for reporting which source(s) it came from.
    """
    rule_meds = _dedupe_rule_medications(rule_meds)
    by_name: dict[str, Medication] = {normalize(med.drug): med for med in rule_meds}
    by_rxcui: dict[str, str] = {}
    for key, med in by_name.items():
        if med.rxcui is not None:
            by_rxcui.setdefault(med.rxcui, key)

    matched_keys: set[str] = set()
    merged: list[dict[str, str | None]] = []

    for gpt_med in gpt_meds:
        gpt_key = normalize(gpt_med["drug"])
        entry = extract_rules.lookup(gpt_med["drug"], mode=_MERGE_LEXICON_MODE)

        rule_match = by_name.get(gpt_key)
        match_key = gpt_key
        if rule_match is None and entry is not None and entry.rxcui is not None:
            candidate_key = by_rxcui.get(entry.rxcui)
            if candidate_key is not None:
                match_key = candidate_key
                rule_match = by_name.get(candidate_key)

        if rule_match is not None and match_key in matched_keys:
            # Already claimed by an earlier A drug (e.g. two A names sharing one rxcui);
            # treat this one as only-in-A rather than double-merging into one B record.
            rule_match = None

        if rule_match is not None:
            matched_keys.add(match_key)
            merged.append(_merged_from_both(gpt_med, rule_match))
            if stats is not None:
                stats.both += 1
            continue

        if entry is not None and not entry.ambiguous:
            merged.append(_from_gpt_only(gpt_med))
            if stats is not None:
                stats.gpt_only += 1
        else:
            if stats is not None:
                stats.rejected_not_in_rxnorm += 1
                if len(stats.rejected_examples) < _MAX_REJECTED_EXAMPLES:
                    stats.rejected_examples.append(gpt_med["drug"])

    for key, rule_med in by_name.items():
        if key in matched_keys:
            continue
        rules_only_med = _from_rules_only(rule_med)
        if rules_only_med is None:
            if stats is not None:
                stats.rules_only_dropped_no_context += 1
            continue
        merged.append(rules_only_med)
        if stats is not None:
            stats.rules_only += 1

    if stats is not None:
        stats.with_route += sum(1 for med in merged if med["route"] is not None)

    return merged


def _project_for_target(medications: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    """Drop "sources" (reporting-only, not part of the model target)."""
    return [
        {
            "drug": med["drug"],
            "strength": med["strength"],
            "dosage": med["dosage"],
            "route": med["route"],
            "frequency": med["frequency"],
            "duration": med["duration"],
        }
        for med in medications
    ]


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens, as a safety net for longer notes."""
    ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def _fix_literal_newlines(text: str) -> str:
    """Some notes store line breaks as a literal two-character "\\n" rather than an actual
    newline; convert them so line-based extraction (lablens.extract_rules) sees real line
    boundaries instead of treating the whole note as one line."""
    return text.replace("\\n", "\n")


def _iter_examples(
    n: int,
    seed: int,
    stats: GroundingStats | None = None,
    merge_stats: MergeStats | None = None,
) -> list[dict[str, str]]:
    """Load the dataset, skip held-out notes, truncate, ground/merge, and collect n examples."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_FLAN_T5_TOKENIZER)
    dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train")
    dataset = dataset.shuffle(seed=seed)

    examples = []
    for row in dataset:
        if row["idx"] in HELD_OUT_NOTE_INDEXES:
            continue

        gpt_meds = build_target_medications(row["summary"])

        note_text = _fix_literal_newlines(row["note"])
        input_text = _truncate_to_tokens(
            tokenizer, "extract medications: " + note_text, _MAX_INPUT_TOKENS
        )
        gpt_meds = ground_target_medications(gpt_meds, input_text, stats)

        # Rules extractor runs on the note alone, without the T5 prefix: the prefix isn't
        # part of the clinical text and only this call's line/sentence boundaries matter.
        rules_text = _truncate_to_tokens(tokenizer, note_text, _MAX_INPUT_TOKENS)
        rule_meds = extract_rules.extract(rules_text, mode=_MERGE_LEXICON_MODE)

        medications = merge_medications(gpt_meds, rule_meds, merge_stats)
        if not medications:
            continue

        target = linearize_medications(_project_for_target(medications))
        examples.append({"input": input_text, "target": target})
        if len(examples) >= n:
            break
    return examples


def _report(examples: list[dict[str, str]], stats: GroundingStats, merge_stats: MergeStats) -> None:
    total_medications = sum(len(parse_medications(ex["target"]) or []) for ex in examples)
    avg_medications = total_medications / len(examples) if examples else 0.0

    print(f"Examples: {len(examples)}")
    print(f"Avg medications per example: {avg_medications:.2f}")
    print(f"Medications dropped (drug name ungrounded): {stats.ungrounded_drugs}")
    print(f"Medications rescued (grounded via trailing parenthetical): {stats.rescued_drugs}")
    print(f"Fields dropped (value ungrounded): {stats.ungrounded_values}")

    total_unique = merge_stats.gpt_only + merge_stats.rules_only + merge_stats.both
    gpt_total = merge_stats.gpt_only + merge_stats.both + merge_stats.rejected_not_in_rxnorm

    print(
        f"Medications per source — GPT-4 only: {merge_stats.gpt_only}, "
        f"rules only: {merge_stats.rules_only}, both: {merge_stats.both}"
    )
    print(
        "Rules-only medications dropped (no supporting dose/route/verb context): "
        f"{merge_stats.rules_only_dropped_no_context}"
    )
    if total_unique:
        print(
            f"Overlap rate (both / all unique medications): {merge_stats.both / total_unique:.2%}"
        )
    if gpt_total:
        print(
            "Agreement rate (both / grounded source-A medications): "
            f"{merge_stats.both / gpt_total:.2%}"
        )
    print(
        "Drugs from source A rejected (not in RxNorm lexicon or ambiguous-only): "
        f"{merge_stats.rejected_not_in_rxnorm}"
    )
    for name in merge_stats.rejected_examples[:_MAX_REJECTED_EXAMPLES]:
        print(f"  - {name}")
    print(f"Medications with a route: {merge_stats.with_route} / {total_unique}")


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
    merge_stats = MergeStats()
    examples = _iter_examples(args.n, args.seed, stats, merge_stats)
    _report(examples, stats, merge_stats)

    n_val = round(len(examples) * args.val_frac)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    out_dir = Path(args.out)
    _write_jsonl(out_dir / "train.jsonl", train_examples)
    _write_jsonl(out_dir / "val.jsonl", val_examples)
    print(f"Wrote {len(train_examples)} train / {len(val_examples)} val examples to {out_dir}")


if __name__ == "__main__":
    main()
