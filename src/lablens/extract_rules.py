"""Rule-based medication extractor: finds known drug names and line-local attributes."""

import re
from importlib import resources

from lablens.schema import Medication

_STRENGTH_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:ml|mg))?\b",
    re.IGNORECASE,
)

_ROUTE_RE = re.compile(
    r"\b(?:by mouth|subcutaneous|topical|inhaled|oral|po|iv|sc)\b",
    re.IGNORECASE,
)

_FREQUENCY_RE = re.compile(
    r"\b(?:once daily|twice daily|every \d+ hours?|as needed|q\d+h|bid|tid|qid|prn|daily)\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"\b(?:for \d+ days?|x\s?\d+\s?d(?:ays?)?)\b",
    re.IGNORECASE,
)


def _load_lexicon() -> list[str]:
    data = resources.files("lablens").joinpath("resources/drug_lexicon.txt").read_text()
    return [line.strip() for line in data.splitlines() if line.strip()]


def _build_drug_pattern(lexicon: list[str]) -> re.Pattern[str]:
    # Longest terms first so multi-word names win over any shorter alternative
    # that happens to start at the same position.
    terms = sorted(lexicon, key=len, reverse=True)
    alternation = "|".join(re.escape(term) for term in terms)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


_LEXICON = _load_lexicon()
_DRUG_RE = _build_drug_pattern(_LEXICON)


def _first_match(pattern: re.Pattern[str], line: str) -> str | None:
    match = pattern.search(line)
    return match.group(0) if match else None


def extract(text: str) -> list[Medication]:
    medications: list[Medication] = []
    offset = 0

    for raw_line in text.split("\n"):
        line_start = offset
        offset = line_start + len(raw_line) + 1
        line = raw_line.removesuffix("\r")
        line_end = line_start + len(line)

        drug_matches = list(_DRUG_RE.finditer(line))
        for i, drug_match in enumerate(drug_matches):
            segment_start = drug_match.start()
            segment_end = drug_matches[i + 1].start() if i + 1 < len(drug_matches) else len(line)
            segment = line[segment_start:segment_end]

            medications.append(
                Medication(
                    drug=drug_match.group(0).lower(),
                    strength=_first_match(_STRENGTH_RE, segment),
                    route=_first_match(_ROUTE_RE, segment),
                    frequency=_first_match(_FREQUENCY_RE, segment),
                    duration=_first_match(_DURATION_RE, segment),
                    evidence=line,
                    start=line_start,
                    end=line_end,
                )
            )

    return medications
