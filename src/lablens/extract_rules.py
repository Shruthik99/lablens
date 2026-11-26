"""Rule-based medication extractor: finds known drug names and line-local attributes."""

import csv
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
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

# Dose units for the ambiguity gate only, kept separate from _STRENGTH_RE so widening it
# cannot change extracted field values. The ambiguous names are mostly electrolytes, which
# are dosed in mEq and mmol; without these, "Potassium 20 mEq PO daily" would be discarded.
# Deliberately no "%": "oxygen saturation 97%" would otherwise read as a dose.
_GATE_DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|meq|mmol|iu|puffs?|tablets?|capsules?|ml|drops?)\b",
    re.IGNORECASE,
)

# The seven medication verbs, with the inflections that actually appear in discharge text:
# "Start lisinopril" is rarer than "Started on lisinopril".
# Note "increase"/"decrease" are deliberately absent. They are dose-change verbs that
# always carry a dose anyway, and including them promotes diet advice such as
# "increase water and iron rich foods" into medications.
_MED_VERB_RE = re.compile(
    r"\b(?:start(?:s|ed|ing)?|stop(?:s|ped|ping)?|continu(?:e|es|ed|ing)|"
    r"tak(?:e|es|en|ing)|prescrib(?:e|es|ed|ing)|discontinu(?:e|es|ed|ing)|"
    r"hold(?:s|ing)?|held)\b",
    re.IGNORECASE,
)

# How far the gate looks for supporting context, in tokens.
_GATE_TOKENS_AFTER = 4
_GATE_TOKENS_BEFORE = 3


# Hyphens and apostrophes are separators, not part of tokens. That keeps "metformin"
# findable inside "metformin-containing" (which the old \b alternation also matched) and
# makes hyphenated lexicon names like "alpha-tocopherol" match however the text spells them.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


HANDWRITTEN = "handwritten"
INGREDIENTS = "ingredients"
INGREDIENTS_CONTEXT = "ingredients_context"
WITH_BRANDS = "with_brands"
WITH_BRANDS_CONTEXT = "with_brands_context"

LEXICON_MODES = (
    HANDWRITTEN,
    INGREDIENTS,
    INGREDIENTS_CONTEXT,
    WITH_BRANDS,
    WITH_BRANDS_CONTEXT,
)

DEFAULT_LEXICON_MODE = HANDWRITTEN

# Which RxNorm term types each mode admits, and whether ambiguous names need context.
_MODE_TERM_TYPES = {
    INGREDIENTS: ("IN", "PIN", ""),
    INGREDIENTS_CONTEXT: ("IN", "PIN", ""),
    WITH_BRANDS: ("IN", "PIN", "BN", ""),
    WITH_BRANDS_CONTEXT: ("IN", "PIN", "BN", ""),
}
_CONTEXT_MODES = frozenset({INGREDIENTS_CONTEXT, WITH_BRANDS_CONTEXT})


@dataclass(frozen=True)
class LexiconEntry:
    """One lexicon term with the provenance needed to trace a prediction back to RxNorm."""

    name: str
    rxcui: str | None
    tty: str | None
    ambiguous: bool


def _read_resource(filename: str) -> str:
    return resources.files("lablens").joinpath(f"resources/{filename}").read_text(encoding="utf-8")


def _load_lexicon() -> list[str]:
    data = _read_resource("drug_lexicon.txt")
    return [line.strip() for line in data.splitlines() if line.strip()]


def _load_entries(mode: str) -> list[LexiconEntry]:
    """Load the lexicon entries a mode admits.

    "handwritten" reads the original committed list, so it stays byte-for-byte the
    behaviour the extractor had before RxNorm was introduced.
    """
    if mode == HANDWRITTEN:
        return [LexiconEntry(name, None, None, False) for name in _load_lexicon()]

    allowed = _MODE_TERM_TYPES[mode]
    entries = []
    rows = csv.DictReader(_read_resource("rxnorm_lexicon.tsv").splitlines(), delimiter="\t")
    for row in rows:
        if row["tty"] not in allowed:
            continue
        entries.append(
            LexiconEntry(
                name=row["name"],
                rxcui=row["rxcui"] or None,
                tty=row["tty"] or None,
                ambiguous=row["ambiguous"] == "1",
            )
        )
    return entries


@dataclass(frozen=True)
class LexiconIndex:
    """A mode's lexicon, prepared for matching."""

    entries: dict[tuple[str, ...], LexiconEntry]
    lengths_by_first_token: dict[str, list[int]]
    requires_context: bool

    def __len__(self) -> int:
        return len(self.entries)


def _build_index(entries: list[LexiconEntry], *, requires_context: bool) -> LexiconIndex:
    """Index entries by token tuple, with candidate lengths bucketed by first token.

    Keying by tokens rather than surface form means a name appearing twice collapses to
    one entry and cannot make the extractor emit the same drug twice. Bucketing the
    lengths keeps matching cheap: without it every position would try every length up to
    the longest term in the lexicon (23 tokens for some RxNorm chemical names).
    """
    by_key: dict[tuple[str, ...], LexiconEntry] = {}
    for entry in entries:
        key = tuple(token for token, _, _ in _tokenize(entry.name))
        if key:
            by_key.setdefault(key, entry)

    lengths: dict[str, set[int]] = {}
    for key in by_key:
        lengths.setdefault(key[0], set()).add(len(key))

    return LexiconIndex(
        entries=by_key,
        lengths_by_first_token={
            token: sorted(sizes, reverse=True) for token, sizes in lengths.items()
        },
        requires_context=requires_context,
    )


def _check_mode(mode: str) -> str:
    if mode not in LEXICON_MODES:
        raise ValueError(f"unknown lexicon mode {mode!r}; valid modes: {', '.join(LEXICON_MODES)}")
    return mode


# maxsize=2 rather than the default 128: each RxNorm index holds up to 23k terms, and the
# comparison runner walks every mode in turn.
@lru_cache(maxsize=2)
def _get_index(mode: str) -> LexiconIndex:
    _check_mode(mode)
    return _build_index(_load_entries(mode), requires_context=mode in _CONTEXT_MODES)


def clear_lexicon_cache() -> None:
    """Drop cached indexes. Tests that swap a lexicon resource must call this."""
    _get_index.cache_clear()


def warm_lexicon_cache(mode: str = DEFAULT_LEXICON_MODE) -> None:
    """Build a mode's index up front.

    Callers that time extraction should warm first, or the first call pays the whole
    index build and reports it as that case's extraction time.
    """
    _get_index(mode)


def lexicon_size(mode: str = DEFAULT_LEXICON_MODE) -> int:
    """Number of distinct terms a mode matches on."""
    return len(_get_index(mode))


_LEXICON = _load_lexicon()


def _find_drugs(line: str, index: LexiconIndex) -> list[tuple[LexiconEntry, str, int, int]]:
    """Find lexicon terms as (entry, drug, start, end), leftmost-longest, non-overlapping.

    Longest candidate first at each position, so "insulin glargine" wins over "insulin".
    The scan then advances past the whole match, so a multi-word term cannot be re-entered
    at its second token.

    `drug` is the matched tokens joined by single spaces, not the canonical RxNorm name:
    canonical names carry punctuation ("insulin, regular, human") that would not match
    gold labels written as plain text.
    """
    tokens = _tokenize(line)
    found: list[tuple[LexiconEntry, str, int, int]] = []
    i = 0
    while i < len(tokens):
        for length in index.lengths_by_first_token.get(tokens[i][0], ()):
            if i + length > len(tokens):
                continue
            key = tuple(token for token, _, _ in tokens[i : i + length])
            entry = index.entries.get(key)
            if entry is not None:
                found.append((entry, " ".join(key), tokens[i][1], tokens[i + length - 1][2]))
                i += length
                break
        else:
            i += 1
    return found


def _has_clinical_context(line: str, tokens: list[tuple[str, int, int]], i: int, length: int):
    """True if an ambiguous name at tokens[i:i+length] is used as a medication here.

    Looks at nearby tokens rather than the whole sentence. A sentence window is both too
    permissive and too fragile: in "Patient is iron deficient; started on ferrous sulfate
    325 mg PO daily" the dose belongs to a different drug, and splitting on "." cuts
    "amlodipine 0.5 mg" in half.

    A dose or route may follow the name ("Potassium 20 mEq PO daily"), or a medication
    verb may precede it ("You were prescribed iron"). No dose is required.
    """
    end = i + length

    after_tokens = tokens[end : end + _GATE_TOKENS_AFTER]
    if after_tokens:
        after = line[tokens[end - 1][2] : after_tokens[-1][2]]
        if _GATE_DOSE_RE.search(after) or _ROUTE_RE.search(after):
            return True

    before_tokens = tokens[max(0, i - _GATE_TOKENS_BEFORE) : i]
    if before_tokens:
        before = line[before_tokens[0][1] : tokens[i][1]]
        if _MED_VERB_RE.search(before):
            return True

    return False


def _apply_context_gate(
    line: str, matches: list[tuple[LexiconEntry, str, int, int]]
) -> list[tuple[LexiconEntry, str, int, int]]:
    """Drop ambiguous names that no nearby clinical context supports.

    Runs before attribute windows are computed: a window ends at the next match's start,
    so a name dropped afterwards would still have truncated the previous drug's window.
    """
    tokens = _tokenize(line)
    starts = {token[1]: index for index, token in enumerate(tokens)}

    kept = []
    for entry, drug, start, end in matches:
        if entry.ambiguous:
            i = starts[start]
            length = len(drug.split(" "))
            if not _has_clinical_context(line, tokens, i, length):
                logger.debug("ambiguous name %r dropped: no clinical context", drug)
                continue
        kept.append((entry, drug, start, end))
    return kept


def _first_match(pattern: re.Pattern[str], line: str) -> str | None:
    match = pattern.search(line)
    return match.group(0) if match else None


def extract(text: str, mode: str = DEFAULT_LEXICON_MODE) -> list[Medication]:
    index = _get_index(_check_mode(mode))
    medications: list[Medication] = []
    offset = 0

    for line_number, raw_line in enumerate(text.split("\n"), start=1):
        line_start = offset
        offset = line_start + len(raw_line) + 1
        line = raw_line.removesuffix("\r")
        line_end = line_start + len(line)

        logger.debug("line %d: %r", line_number, line)

        drug_matches = _find_drugs(line, index)
        if index.requires_context:
            drug_matches = _apply_context_gate(line, drug_matches)

        for i, (_entry, drug, drug_start, _drug_end) in enumerate(drug_matches):
            segment_start = drug_start
            segment_end = drug_matches[i + 1][2] if i + 1 < len(drug_matches) else len(line)
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
