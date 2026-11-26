"""Evaluate a medication extractor against a labeled fixture.

Runnable as: python -m lablens.evaluate <path-to-fixture.json> [--extractor {rules,llm,t5}]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from lablens import extract_llm, extract_rules, extract_t5
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

    # None (not 1.0) when a field was never compared, i.e. there were zero matched drugs:
    # a ratio here would silently read as "perfect" when nothing was actually checked.
    field_accuracy = {
        field: (field_correct[field] / field_total[field] if field_total[field] else None)
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


def evaluate_cases(
    cases: list[dict[str, Any]],
    extract_fn: Callable[[str], list[Medication]] = extract,
) -> dict[str, Any]:
    """Run extract_fn over each case, computing metrics and average time per case."""
    results = []
    durations = []
    for case in cases:
        started = time.perf_counter()
        predicted = extract_fn(case["text"])
        durations.append(time.perf_counter() - started)
        results.append(score_case(case["text"], case["expected"], predicted))

    metrics = summarize(results)
    metrics["avg_seconds_per_case"] = sum(durations) / len(durations) if durations else 0.0
    return metrics


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
    lines.extend(
        (
            f"  {field}: n/a"
            if metrics["field_accuracy"][field] is None
            else f"  {field}: {metrics['field_accuracy'][field]:.3f}"
        )
        for field in FIELDS
    )

    if "avg_seconds_per_case" in metrics:
        lines.append(f"Avg seconds/case: {metrics['avg_seconds_per_case']:.3f}")

    if "ungrounded_values" in metrics:
        lines.append(f"Ungrounded values dropped: {metrics['ungrounded_values']}")

    if "ungrounded_drugs" in metrics:
        lines.append(f"Ungrounded drugs dropped: {metrics['ungrounded_drugs']}")

    if "parse_failures" in metrics:
        lines.append(f"Parse failures: {metrics['parse_failures']}")

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


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _load_run_info(model_dir: str) -> dict[str, Any] | None:
    """Load models/flan-t5/run_info.json (written by train_t5.py), or None if absent."""
    path = Path(model_dir) / "run_info.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_run_data(
    fixture_path: str,
    metrics: dict[str, Any],
    *,
    lexicon_size: int | None = None,
    lexicon_mode: str | None = None,
    lexicon_sha256: str | None = None,
    rxnorm_version: str | None = None,
    fixture_sha256: str,
    git_commit: str | None,
    extractor: str = "rules",
    model: str | None = None,
    prompt_version: str | None = None,
    model_dir: str | None = None,
    t5_run_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the MLflow params/metrics payload for one evaluation run."""
    params: dict[str, Any] = {
        "extractor": extractor,
        "fixture_path": fixture_path,
        "fixture_sha256": fixture_sha256,
    }
    if lexicon_size is not None:
        params["lexicon_size"] = lexicon_size
    if lexicon_mode is not None:
        params["lexicon_mode"] = lexicon_mode
    if lexicon_sha256 is not None:
        params["lexicon_sha256"] = lexicon_sha256
    if rxnorm_version is not None:
        params["rxnorm_version"] = rxnorm_version
    if model is not None:
        params["model"] = model
    if prompt_version is not None:
        params["prompt_version"] = prompt_version
    if extractor == "llm":
        params["grounding"] = "on"
    if git_commit:
        params["git_commit"] = git_commit
    if model_dir is not None:
        params["model_dir"] = model_dir
    if t5_run_info is not None:
        if t5_run_info.get("git_commit"):
            params["model_git_commit"] = t5_run_info["git_commit"]
        data_sha256 = t5_run_info.get("data_sha256", {})
        if "train.jsonl" in data_sha256:
            params["model_train_sha256"] = data_sha256["train.jsonl"]
        if "val.jsonl" in data_sha256:
            params["model_val_sha256"] = data_sha256["val.jsonl"]

    run_metrics = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }
    run_metrics.update(
        {
            f"field_accuracy_{field}": metrics["field_accuracy"][field]
            for field in FIELDS
            if metrics["field_accuracy"][field] is not None
        }
    )
    if "avg_seconds_per_case" in metrics:
        run_metrics["avg_seconds_per_case"] = metrics["avg_seconds_per_case"]
    if "ungrounded_values" in metrics:
        run_metrics["ungrounded_values"] = metrics["ungrounded_values"]
    if "ungrounded_drugs" in metrics:
        run_metrics["ungrounded_drugs"] = metrics["ungrounded_drugs"]
    if "parse_failures" in metrics:
        run_metrics["parse_failures"] = metrics["parse_failures"]

    return {"params": params, "metrics": run_metrics}


def log_mlflow_run(
    fixture_path: str,
    metrics: dict[str, Any],
    report: str,
    *,
    extractor: str = "rules",
    model: str | None = None,
    model_dir: str | None = None,
    lexicon_mode: str | None = None,
) -> None:
    """Log one evaluation run to MLflow. Imports mlflow lazily since it's an optional extra."""
    import mlflow

    t5_run_info = _load_run_info(model_dir) if extractor == "t5" and model_dir else None
    is_rules = extractor == "rules"
    mode = lexicon_mode or extract_rules.DEFAULT_LEXICON_MODE

    run_data = build_run_data(
        fixture_path,
        metrics,
        lexicon_size=extract_rules.lexicon_size(mode) if is_rules else None,
        lexicon_mode=mode if is_rules else None,
        lexicon_sha256=extract_rules.lexicon_sha256(mode) if is_rules else None,
        rxnorm_version=extract_rules.rxnorm_version(mode) if is_rules else None,
        fixture_sha256=_sha256_file(fixture_path),
        git_commit=_git_commit(),
        extractor=extractor,
        model=model,
        prompt_version=extract_llm.PROMPT_VERSION if extractor == "llm" else None,
        model_dir=model_dir if extractor == "t5" else None,
        t5_run_info=t5_run_info,
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("medication-extraction")
    with mlflow.start_run():
        mlflow.log_params(run_data["params"])
        mlflow.log_metrics(run_data["metrics"])
        mlflow.log_text(report, "report.txt")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Evaluate a medication extractor.")
    parser.add_argument("fixture", help="Path to a JSON file of {text, expected} cases.")
    parser.add_argument(
        "--extractor",
        choices=["rules", "llm", "t5"],
        default="rules",
        help="Which extractor to evaluate (default: rules).",
    )
    parser.add_argument(
        "--model",
        default="qwen3:4b",
        help="Ollama model to use when --extractor=llm (default: qwen3:4b).",
    )
    parser.add_argument(
        "--model-dir",
        default=extract_t5.DEFAULT_MODEL_DIR,
        help=f"Fine-tuned model directory to use when --extractor=t5 (default: {extract_t5.DEFAULT_MODEL_DIR}).",
    )
    parser.add_argument(
        "--lexicon-mode",
        choices=list(extract_rules.LEXICON_MODES),
        default=extract_rules.DEFAULT_LEXICON_MODE,
        help=(
            "Which lexicon the rules extractor matches on "
            f"(default: {extract_rules.DEFAULT_LEXICON_MODE})."
        ),
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log this run to MLflow (sqlite:///mlflow.db, experiment 'medication-extraction').",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    with open(args.fixture, encoding="utf-8") as f:
        cases = json.load(f)

    stats = None
    if args.extractor == "llm":
        stats = extract_llm.GroundingStats()
        extract_fn = partial(extract_llm.extract, model=args.model, stats=stats)
    elif args.extractor == "t5":
        stats = extract_t5.GroundingStats()
        extract_fn = partial(extract_t5.extract, model_dir=args.model_dir, stats=stats)
    else:
        # Build the index before timing starts, or the first case is billed for it.
        extract_rules.warm_lexicon_cache(args.lexicon_mode)
        extract_fn = partial(extract, mode=args.lexicon_mode)

    metrics = evaluate_cases(cases, extract_fn)
    if stats is not None:
        metrics["ungrounded_values"] = stats.ungrounded_values
        metrics["ungrounded_drugs"] = stats.ungrounded_drugs
        metrics["parse_failures"] = stats.parse_failures

    report = format_report(metrics)
    print(report)

    if args.track:
        log_mlflow_run(
            args.fixture,
            metrics,
            report,
            extractor=args.extractor,
            model=args.model if args.extractor == "llm" else None,
            model_dir=args.model_dir if args.extractor == "t5" else None,
            lexicon_mode=args.lexicon_mode if args.extractor == "rules" else None,
        )

    return metrics


if __name__ == "__main__":
    main()
