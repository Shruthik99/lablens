"""Shared grounding helpers: keep only drug names and field values that are literally
present in a piece of source text, to filter out hallucinated or weak-label content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class GroundingStats:
    """Counts of medications/fields dropped or rescued by the grounding checks."""

    ungrounded_drugs: int = 0
    ungrounded_values: int = 0
    rescued_drugs: int = 0
    parse_failures: int = 0


def normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def is_drug_grounded(drug: str, evidence: str) -> bool:
    """True if drug appears in evidence as a whole word/phrase, not inside another word
    (e.g. "iron" must not match inside "environment")."""
    pattern = r"\b" + re.escape(normalize(drug)) + r"\b"
    return re.search(pattern, normalize(evidence)) is not None


def find_evidence_line(text: str, drug: str) -> tuple[str, int, int] | None:
    """Return (line, start, end) of the first line containing drug as a whole word/phrase
    (case-insensitive), or None."""
    offset = 0
    for raw_line in text.split("\n"):
        line = raw_line.removesuffix("\r")
        if is_drug_grounded(drug, line):
            return line, offset, offset + len(line)
        offset += len(raw_line) + 1
    return None


def ground_value(
    value: str | None, evidence: str, stats: GroundingStats | None = None
) -> str | None:
    """Keep value only if it appears in evidence as a standalone token, not e.g. "5 mg" in "25 mg"."""
    if value is None:
        return None
    pattern = r"(?<![0-9.])" + re.escape(normalize(value)) + r"(?!\d)"
    if re.search(pattern, normalize(evidence)):
        return value
    if stats is not None:
        stats.ungrounded_values += 1
    return None
