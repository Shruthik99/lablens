"""Evaluate the rule-based medication extractor against a labeled fixture.

Runnable as: python -m lablens.evaluate <path-to-fixture.json>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

from lablens.extract_rules import extract
from lablens.schema import Medication

FIELDS = ("strength", "dosage", "route", "frequency", "duration")

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(value: str | None) -> str | None:
    if value is None:
        return None
    return _WHITESPACE_RE.sub(" ", value.strip()).lower()


def _fields_match(expected_value: str | None, predicted_value: str | None) -> bool:
    return _normalize(expected_value) == _normalize(predicted_value)


def match_drugs(
    expected: list[dict[str, Any]], predicted: list[Medication]
) -> tuple[list[tuple[dict[str, Any], Medication]], list[dict[str, Any]], list[Medication]]:
    """Greedily pair expected medications to predicted ones by drug name (case-insensitive).

    Returns (matches, missed expected entries, extra predicted entries).
    """
    remaining = list(predicted)
    matches = []
    missed = []
    for exp in expected:
        idx = next(
            (i for i, med in enumerate(remaining) if med.drug.lower() == exp["drug"].lower()),
            None,
        )
        if idx is None:
            missed.append(exp)
        else:
            matches.append((exp, remaining.pop(idx)))
    return matches, missed, remaining


def score_case(
    text: str, expected: list[dict[str, Any]], predicted: list[Medication]
) -> dict[str, Any]:
    """Score one case's predicted medications against its expected medications."""
    matches, missed, extra = match_drugs(expected, predicted)

    field_correct = {field: 0 for field in FIELDS}
    field_total = {field: 0 for field in FIELDS}
    mismatches = []

    for exp, pred in matches:
        for field in FIELDS:
            exp_val = exp.get(field)
            pred_val = getattr(pred, field)
            field_total[field] += 1
            if _fields_match(exp_val, pred_val):
                field_correct[field] += 1
            else:
                mismatches.append(
                    {
                        "text": text,
                        "drug": exp["drug"],
                        "field": field,
                        "expected": exp_val,
                        "predicted": pred_val,
                    }
                )

    return {
        "true_positives": len(matches),
        "false_negatives": len(missed),
        "false_positives": len(extra),
        "field_correct": field_correct,
        "field_total": field_total,
        "mismatches": mismatches,
        "missed_drugs": [{"text": text, "drug": exp["drug"]} for exp in missed],
        "extra_drugs": [{"text": text, "drug": pred.drug} for pred in extra],
    }


def summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case scores into overall metrics."""
    tp = sum(r["true_positives"] for r in case_results)
    fn = sum(r["false_negatives"] for r in case_results)
    fp = sum(r["false_positives"] for r in case_results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    field_correct = {field: 0 for field in FIELDS}
    field_total = {field: 0 for field in FIELDS}
    mismatches: list[dict[str, Any]] = []
    missed_drugs: list[dict[str, Any]] = []
    extra_drugs: list[dict[str, Any]] = []

    for r in case_results:
        for field in FIELDS:
            field_correct[field] += r["field_correct"][field]
            field_total[field] += r["field_total"][field]
        mismatches.extend(r["mismatches"])
        missed_drugs.extend(r["missed_drugs"])
        extra_drugs.extend(r["extra_drugs"])

    field_accuracy = {
        field: (field_correct[field] / field_total[field] if field_total[field] else 1.0)
        for field in FIELDS
    }

    return {
        "num_cases": len(case_results),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "field_accuracy": field_accuracy,
        "mismatches": mismatches,
        "missed_drugs": missed_drugs,
        "extra_drugs": extra_drugs,
    }


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run extract() over each case and compute metrics against the expected medications."""
    results = [score_case(case["text"], case["expected"], extract(case["text"])) for case in cases]
    return summarize(results)


def format_report(metrics: dict[str, Any]) -> str:
    lines = [
        f"Cases evaluated: {metrics['num_cases']}",
        (
            f"Drug detection: precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
            f"(tp={metrics['true_positives']} fp={metrics['false_positives']} "
            f"fn={metrics['false_negatives']})"
        ),
        "Field accuracy (matched drugs only):",
    ]
    lines.extend(f"  {field}: {metrics['field_accuracy'][field]:.3f}" for field in FIELDS)

    if metrics["mismatches"]:
        lines.append("")
        lines.append("Field mismatches:")
        lines.extend(
            f"  [{m['field']}] drug={m['drug']!r} text={m['text']!r} "
            f"expected={m['expected']!r} predicted={m['predicted']!r}"
            for m in metrics["mismatches"]
        )

    if metrics["missed_drugs"]:
        lines.append("")
        lines.append("Missed drugs (expected but not predicted):")
        lines.extend(f"  drug={m['drug']!r} text={m['text']!r}" for m in metrics["missed_drugs"])

    if metrics["extra_drugs"]:
        lines.append("")
        lines.append("Extra drugs (predicted but not expected):")
        lines.extend(f"  drug={e['drug']!r} text={e['text']!r}" for e in metrics["extra_drugs"])

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate the rule-based medication extractor.")
    parser.add_argument("fixture", help="Path to a JSON file of {text, expected} cases.")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    with open(args.fixture, encoding="utf-8") as f:
        cases = json.load(f)

    metrics = evaluate_cases(cases)
    print(format_report(metrics))
    return metrics


if __name__ == "__main__":
    main()
