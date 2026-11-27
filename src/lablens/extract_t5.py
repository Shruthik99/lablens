"""Flan-T5-based medication extractor using a locally fine-tuned checkpoint.

Runs entirely on the local machine: the model directory is loaded once per process and
cached from then on; no document text or data ever leaves it.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from lablens.grounding import GroundingStats, find_evidence_line, ground_value, normalize
from lablens.schema import Medication
from lablens.t5_format import parse_medications

DEFAULT_MODEL_DIR = "models/flan-t5"
_MAX_NEW_TOKENS = 256
# Matches the input cap the model was trained with (see prepare_t5_data._MAX_INPUT_TOKENS).
_MAX_INPUT_TOKENS = 768


@cache
def _load_model(model_dir: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def _generate(text: str, model_dir: str) -> str:
    tokenizer, model = _load_model(model_dir)

    inputs = tokenizer(
        "extract medications: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_INPUT_TOKENS,
    )
    output_ids = model.generate(
        **inputs,
        max_new_tokens=_MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


_ATTRIBUTE_FIELDS = ("strength", "dosage", "route", "frequency", "duration")


def _dominates(a: dict[str, str | None], b: dict[str, str | None]) -> bool:
    """True if a is at least as complete as b and doesn't conflict with it: every field b
    sets is also set in a, with an equal value. (Two identical entries dominate each
    other; a bare entry with no fields at all is trivially dominated by anything.)"""
    return all(b[field] is None or a[field] == b[field] for field in _ATTRIBUTE_FIELDS)


def _reduce_group(
    indexed_items: list[tuple[int, dict[str, str | None]]],
) -> list[tuple[int, dict[str, str | None]]]:
    """Keep only the entries not dominated by another entry for the same drug -- drop a
    less-complete copy once a more-complete (or identical) one is known. Two entries with
    conflicting field values dominate neither one another and are both kept: they're never
    merged. Ties (identical entries) keep the first occurrence."""
    survivors: list[tuple[int, dict[str, str | None]]] = []
    for idx, item in indexed_items:
        if any(_dominates(s_item, item) for _s_idx, s_item in survivors):
            continue
        survivors[:] = [
            (s_idx, s_item) for s_idx, s_item in survivors if not _dominates(item, s_item)
        ]
        survivors.append((idx, item))
    return survivors


def _dedupe(
    parsed: list[dict[str, str | None]], stats: GroundingStats | None
) -> list[dict[str, str | None]]:
    """For the same normalized drug, drop an entry that's a less-complete copy of another
    (all its non-null fields equal the corresponding fields of another entry) -- a
    generation loop can otherwise repeat one medication, sometimes with extra fields on
    one copy, many times. Entries with conflicting values are never merged."""
    groups: dict[str, list[tuple[int, dict[str, str | None]]]] = {}
    for idx, item in enumerate(parsed):
        groups.setdefault(normalize(item["drug"]), []).append((idx, item))

    surviving_indices = {idx for group in groups.values() for idx, _item in _reduce_group(group)}
    deduped = [item for idx, item in enumerate(parsed) if idx in surviving_indices]
    if stats is not None:
        stats.duplicates_removed += len(parsed) - len(deduped)
    return deduped


def extract(
    text: str,
    model_dir: str = DEFAULT_MODEL_DIR,
    stats: GroundingStats | None = None,
) -> list[Medication]:
    generated = _generate(text, model_dir)

    parsed = parse_medications(generated)
    if parsed is None:
        if stats is not None:
            stats.parse_failures += 1
        return []

    parsed = _dedupe(parsed, stats)

    medications: list[Medication] = []
    for item in parsed:
        grounded = find_evidence_line(text, item["drug"])
        if grounded is None:
            if stats is not None:
                stats.ungrounded_drugs += 1
            continue
        line, start, end = grounded

        medications.append(
            Medication(
                drug=item["drug"],
                strength=ground_value(item.get("strength"), line, stats),
                dosage=ground_value(item.get("dosage"), line, stats),
                route=ground_value(item.get("route"), line, stats),
                frequency=ground_value(item.get("frequency"), line, stats),
                duration=ground_value(item.get("duration"), line, stats),
                evidence=line,
                start=start,
                end=end,
            )
        )
    return medications
