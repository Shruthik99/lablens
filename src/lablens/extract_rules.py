"""Rule-based medication extractor: finds known drug names and line-local attributes."""

import logging
import re
from importlib import resources

from lablens.schema import Medication

logger = logging.getLogger(__name__)

_STRENGTH_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:ml|mg))?\b",
    re.IGNORECASE,
)

_ROUTE_RE = re.compile(
    r"\b(?:by mouth|subcutaneous|subcut|subq|intramuscular|intra-articular|"
    r"inhalation|inhaler|inhaled|injection|topical|oral|po|iv|im|sc)\b",
    re.IGNORECASE,
)

_BASE_FREQUENCY = (
    r"once daily|twice daily|weekly|nightly|at bedtime|qhs|every morning|"
    r"every evening|every \d+ hours?|q\d+h|bid|tid|qid|daily"
)

_FREQUENCY_RE = re.compile(
    rf"\b(?:(?:{_BASE_FREQUENCY})\s+(?:prn|as needed)|{_BASE_FREQUENCY}|as needed|prn)\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"\b(?:for \d+ days?|x\s?\d+\s?d(?:ays?)?)\b",
    re.IGNORECASE,
)

_DOSAGE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:puffs?|tablets?|capsules?|ml|drops?)\b",
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

    for line_number, raw_line in enumerate(text.split("\n"), start=1):
        line_start = offset
        offset = line_start + len(raw_line) + 1
        line = raw_line.removesuffix("\r")
        line_end = line_start + len(line)

        logger.debug("line %d: %r", line_number, line)

        drug_matches = list(_DRUG_RE.finditer(line))
        for i, drug_match in enumerate(drug_matches):
            segment_start = drug_match.start()
            segment_end = drug_matches[i + 1].start() if i + 1 < len(drug_matches) else len(line)
            segment = line[segment_start:segment_end]

            logger.debug(
                "drug match: %r at [%d, %d)", drug_match.group(0), segment_start, segment_end
            )
            logger.debug("attribute window: %r", segment)

            strength = _first_match(_STRENGTH_RE, segment)
            logger.debug("strength: %s", strength or "not found")

            dosage = _first_match(_DOSAGE_RE, segment)
            logger.debug("dosage: %s", dosage or "not found")

            route = _first_match(_ROUTE_RE, segment)
            logger.debug("route: %s", route or "not found")

            frequency = _first_match(_FREQUENCY_RE, segment)
            logger.debug("frequency: %s", frequency or "not found")

            duration = _first_match(_DURATION_RE, segment)
            logger.debug("duration: %s", duration or "not found")

            medications.append(
                Medication(
                    drug=drug_match.group(0).lower(),
                    strength=strength,
                    dosage=dosage,
                    route=route,
                    frequency=frequency,
                    duration=duration,
                    evidence=line,
                    start=line_start,
                    end=line_end,
                )
            )

    return medications
