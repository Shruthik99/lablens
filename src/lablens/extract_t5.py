"""Flan-T5-based medication extractor using a locally fine-tuned checkpoint.

Runs entirely on the local machine: the model directory is loaded once per process and
cached from then on; no document text or data ever leaves it.
"""

from __future__ import annotations

from functools import cache
from typing import Any

from lablens.grounding import GroundingStats, find_evidence_line, ground_value
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
