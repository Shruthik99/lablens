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


# Hyphens and apostrophes are separators, not part of tokens. That keeps "metformin"
# findable inside "metformin-containing" (which the old \b alternation also matched) and
# makes hyphenated lexicon names like "alpha-tocopherol" match however the text spells them.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _load_lexicon() -> list[str]:
    data = (
        resources.files("lablens")
        .joinpath("resources/drug_lexicon.txt")
        .read_text(encoding="utf-8")
    )
    return [line.strip() for line in data.splitlines() if line.strip()]


def _build_index(lexicon: list[str]) -> dict[tuple[str, ...], str]:
    """Map each term's token tuple to the term, so lookup is exact and whitespace-agnostic.

    Keyed by tokens rather than surface form, so a name appearing twice in the lexicon
    collapses to one entry and cannot make the extractor emit the same drug twice.
    """
    index: dict[tuple[str, ...], str] = {}
    for term in lexicon:
        key = tuple(token for token, _, _ in _tokenize(term))
        if key:
            index.setdefault(key, term)
    return index


def _build_lengths_by_first_token(index: dict[tuple[str, ...], str]) -> dict[str, list[int]]:
    """Candidate term lengths per first token, longest first.

    Without this, every position would try every length up to the longest term in the
    lexicon (23 tokens for some RxNorm chemical names) instead of the few that can match.
    """
    lengths: dict[str, set[int]] = {}
    for key in index:
        lengths.setdefault(key[0], set()).add(len(key))
    return {token: sorted(sizes, reverse=True) for token, sizes in lengths.items()}


_LEXICON = _load_lexicon()
_INDEX = _build_index(_LEXICON)
_LENGTHS_BY_FIRST_TOKEN = _build_lengths_by_first_token(_INDEX)


def _find_drugs(line: str) -> list[tuple[str, int, int]]:
    """Find lexicon terms in a line as (drug, start, end), leftmost-longest, non-overlapping.

    Longest candidate first at each position, so "insulin glargine" wins over "insulin".
    The scan then advances past the whole match, so a multi-word term cannot be re-entered
    at its second token.
    """
    tokens = _tokenize(line)
    found: list[tuple[str, int, int]] = []
    i = 0
    while i < len(tokens):
        for length in _LENGTHS_BY_FIRST_TOKEN.get(tokens[i][0], ()):
            if i + length > len(tokens):
                continue
            key = tuple(token for token, _, _ in tokens[i : i + length])
            if key in _INDEX:
                found.append((" ".join(key), tokens[i][1], tokens[i + length - 1][2]))
                i += length
                break
        else:
            i += 1
    return found


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

        drug_matches = _find_drugs(line)
        for i, (drug, drug_start, _drug_end) in enumerate(drug_matches):
            segment_start = drug_start
            segment_end = drug_matches[i + 1][1] if i + 1 < len(drug_matches) else len(line)
            segment = line[segment_start:segment_end]

            logger.debug("drug match: %r at [%d, %d)", drug, segment_start, segment_end)
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
                    drug=drug,
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
