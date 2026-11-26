"""Compare every lexicon mode on identical data.

Runnable as: python -m lablens.compare_lexicons FIXTURE [FIXTURE ...] [--track]

Expanding a medication lexicon trades recall for precision, so the point of this script
is to show the trade rather than assert it: each mode runs over exactly the same cases
and the table reports what it found, what it invented, and what it cost.

Field accuracy is reported alongside precision and recall because a spurious match also
truncates the previous drug's attribute window, so attributes degrade with lexicon size
even when the extra match is discarded later.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

from lablens import extract_rules
from lablens.evaluate import FIELDS, evaluate_cases, log_mlflow_run

BRAND_MODES = ("with_brands", "with_brands_context")


def load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            cases.extend(json.load(f))
    return cases


def run_mode(mode: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate one lexicon mode, keeping only one index in memory at a time."""
    extract_rules.clear_lexicon_cache()
    extract_rules.warm_lexicon_cache(mode)
    metrics = evaluate_cases(cases, partial(extract_rules.extract, mode=mode))
    metrics["lexicon_size"] = extract_rules.lexicon_size(mode)
    metrics["found_drugs"] = _found_drugs(mode, cases)
    return metrics


def _found_drugs(mode: str, cases: list[dict[str, Any]]) -> set[str]:
    """Every distinct drug name a mode produced, for the brand-coverage diff."""
    return {med.drug for case in cases for med in extract_rules.extract(case["text"], mode=mode)}


def brand_only_drugs(results: dict[str, dict[str, Any]]) -> set[str]:
    """Drugs that only the brand-enabled modes found: the incremental brand coverage."""
    ingredient_drugs: set[str] = set()
    brand_drugs: set[str] = set()
    for mode, metrics in results.items():
        if mode in BRAND_MODES:
            brand_drugs |= metrics["found_drugs"]
        else:
            ingredient_drugs |= metrics["found_drugs"]
    return brand_drugs - ingredient_drugs


def format_table(results: dict[str, dict[str, Any]]) -> str:
    header = (
        f"{'mode':<22}{'terms':>8}{'tp':>5}{'fp':>5}{'fn':>5}"
        f"{'precision':>11}{'recall':>9}{'f1':>8}{'fields':>9}"
    )
    lines = [header, "-" * len(header)]

    for mode, m in results.items():
        predicted_any = m["true_positives"] + m["false_positives"] > 0
        precision = f"{m['precision']:.3f}" if predicted_any else "n/a"

        scored = [m["field_accuracy"][f] for f in FIELDS if m["field_accuracy"][f] is not None]
        fields = f"{sum(scored) / len(scored):.3f}" if scored else "n/a"

        lines.append(
            f"{mode:<22}{m['lexicon_size']:>8}{m['true_positives']:>5}"
            f"{m['false_positives']:>5}{m['false_negatives']:>5}"
            f"{precision:>11}{m['recall']:>9.3f}{m['f1']:>8.3f}{fields:>9}"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, dict[str, Any]]:
    parser = argparse.ArgumentParser(description="Compare lexicon modes on identical data.")
    parser.add_argument(
        "fixtures",
        nargs="+",
        type=Path,
        help="One or more JSON fixtures of {text, expected} cases, evaluated together.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log one MLflow run per mode (sqlite:///mlflow.db).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    cases = load_cases(args.fixtures)
    print(f"Cases: {len(cases)} from {', '.join(str(p) for p in args.fixtures)}")
    print(f"Expected medications: {sum(len(c['expected']) for c in cases)}\n")

    results = {mode: run_mode(mode, cases) for mode in extract_rules.LEXICON_MODES}
    print(format_table(results))

    brand_only = brand_only_drugs(results)
    print(f"\nIncremental brand coverage: {len(brand_only)} drug(s) found only with brand names")
    for drug in sorted(brand_only):
        print(f"  {drug}")

    false_positives = {mode: [e["drug"] for e in m["extra_drugs"]] for mode, m in results.items()}
    print("\nFalse positives by mode:")
    for mode, drugs in false_positives.items():
        print(f"  {mode:<22}{len(drugs):>3}  {sorted(set(drugs))}")

    if args.track:
        for mode, metrics in results.items():
            log_mlflow_run(
                str(args.fixtures[0]),
                metrics,
                format_table({mode: metrics}),
                extractor="rules",
                lexicon_mode=mode,
            )

    return results


if __name__ == "__main__":
    main()
