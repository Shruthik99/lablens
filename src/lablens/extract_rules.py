"""Rule-based medication extractor: finds known drug names and line-local attributes."""

import csv
import hashlib
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from lablens.schema import Medication

logger = logging.getLogger(__name__)

# A plain number, or a comma-grouped one ("25,000", "1,000"); either may carry a decimal.
_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"

# A dose range ("40-60 mg"), or just a single number -- the whole range is the strength,
# not only its second half (what a bare _NUMBER would match, the hyphen breaking it).
_NUMBER_RANGE = rf"(?:{_NUMBER})(?:\s*-\s*(?:{_NUMBER}))?"

_STRENGTH_RE = re.compile(
    rf"\b(?:{_NUMBER_RANGE})\s*(?:mg|mcg|g|units?)"
    rf"(?:\s*/\s*(?:(?:{_NUMBER})\s*(?:ml|mg)|kg(?:\s*/\s*day)?|h(?:r)?))?\b",
    re.IGNORECASE,
)

_ROUTE_RE = re.compile(
    r"\b(?:by mouth|subcutaneous|subcut|subq|intramuscular|intra-articular|"
    r"inhalation|inhaler|inhaled|injection|topical|oral|po|iv|im|sc)\b",
    re.IGNORECASE,
)

# "daily"/"a day"/"per day" are interchangeable after a count ("once", "twice", "N
# times"); keeping them as one shared suffix covers all three without hand-enumerating
# each combination.
_DAILY_SUFFIX = r"(?:daily|a day|per day)"

_BASE_FREQUENCY = (
    rf"once {_DAILY_SUFFIX}|twice {_DAILY_SUFFIX}|thrice {_DAILY_SUFFIX}|"
    rf"(?:\d+|two|three|four|five|six) times {_DAILY_SUFFIX}|"
    r"weekly|nightly|at bedtime|qhs|every morning|"
    r"every evening|every \d+ h(?:ours?)?|q\d+h|bid|tid|qid|daily"
)

_FREQUENCY_RE = re.compile(
    rf"\b(?:(?:{_BASE_FREQUENCY})\s+(?:prn|as needed)|{_BASE_FREQUENCY}|as needed|prn|"
    # A bare per-day suffix on a strength ("75 mg/day", "80 mg/d") is a frequency, not
    # part of the dose; excluded when it follows "kg" so "5 mg/kg/day" stays one strength.
    rf"(?<!kg)/\s*d(?:ay)?)\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"\b(?:for \d+ days?|x\s?\d+\s?d(?:ays?)?)\b",
    re.IGNORECASE,
)

_DOSAGE_RE = re.compile(
    rf"\b(?:{_NUMBER})\s*(?:puffs?|tablets?|capsules?|ml|drops?)\b",
    re.IGNORECASE,
)

# A volume stated as a diluent ("25,000 units in 250 mL D5W") is not a dosage -- the
# drug's own dose is the strength, not how much fluid it was mixed into.
_DILUENT_VOLUME_RE = re.compile(r"\bin\s*\Z", re.IGNORECASE)

# Sentence boundary for the attribute window only: punctuation followed by whitespace.
# A decimal point ("0.5 mg") is never followed by whitespace, so it is never a false split.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+")

# A lab concentration unit, as opposed to a dose: "mg/dL", "mmol/L", "g/dL", "ng/mL",
# "mIU/mL". Distinct from _STRENGTH_RE's units, which are per-dose, not per-volume.
_LAB_UNIT_RE = r"(?:mg|mmol|g|ng|miu)\s*/\s*(?:dl|ml|l)"

# Drug mentions in a lab/test context, never a medication regardless of ambiguity flag:
# a result ("sodium level", "glucose value", "lactate of above 15 mmol/L"), a receptor
# name ("insulin receptor"), a resistance profile ("penicillin-resistant"), a number in a
# lab unit directly after the name ("glucose 340 mg/dL"), or a lab panel prefix ("serum
# digoxin", "levels of immunoglobulin G").
_LAB_SUFFIX_RE = re.compile(
    rf"\A[\s(-]*(?:level|receptors?|resistant|values?|"
    rf"of\s+(?:(?:above|below|around|approximately|over|under)\s+)?(?:{_NUMBER})|"
    rf"(?:{_NUMBER})\s*{_LAB_UNIT_RE})\b",
    re.IGNORECASE,
)
_LAB_PREFIX_RE = re.compile(r"\b(?:serum|levels?\s+of|values?\s+of)\s*\Z", re.IGNORECASE)
_LAB_CONTEXT_CHARS = 30

# "blood urea nitrogen" (BUN): a lab test name, never a medication, that happens to
# tokenize across two ambiguous lexicon entries ("urea", "nitrogen"). Checked as its own
# phrase (both sides must match) rather than folded into the suffix/prefix alternatives
# above, since neither "blood" nor "nitrogen" alone implies a lab context.
_BUN_PREFIX_RE = re.compile(r"\bblood\s*\Z", re.IGNORECASE)
_BUN_SUFFIX_RE = re.compile(r"\A\s*nitrogen\b", re.IGNORECASE)

# Dose units for the ambiguity gate only, kept separate from _STRENGTH_RE so widening it
# cannot change extracted field values. The ambiguous names are mostly electrolytes, which
# are dosed in mEq and mmol; without these, "Potassium 20 mEq PO daily" would be discarded.
# Deliberately no "%": "oxygen saturation 97%" would otherwise read as a dose.
_GATE_DOSE_RE = re.compile(
    rf"\b(?:{_NUMBER})\s*(?:mg|mcg|g|units?|meq|mmol|iu|puffs?|tablets?|capsules?|ml|drops?)\b",
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

# Phrases that mean a different (or split-across-tokens) drug name than their own tokens
# would suggest: a name split across two tokens that exists in the lexicon only as one
# fused token ("methyl prednisolone", "methyl-prednisolone" -- hyphens are already
# separators, so both spellings tokenize identically), and aliases for a specific drug
# that are themselves never lexicon entries ("tissue plasminogen activator", "tPA" both
# mean alteplase; see also "plasminogen" in ambiguity_list.txt). Merging/renaming in the
# token stream (rather than rewriting the line) lets them match without shifting the
# character offsets evidence/segments rely on.
_COMPOUND_TOKEN_MERGES: dict[tuple[str, ...], str] = {
    ("methyl", "prednisolone"): "methylprednisolone",
    ("tissue", "plasminogen", "activator"): "alteplase",
    ("tpa",): "alteplase",
}
_MAX_MERGE_LENGTH = max(len(key) for key in _COMPOUND_TOKEN_MERGES)


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    tokens = [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    merged = []
    i = 0
    while i < len(tokens):
        for length in range(min(_MAX_MERGE_LENGTH, len(tokens) - i), 0, -1):
            window = tuple(token for token, _, _ in tokens[i : i + length])
            fused_name = _COMPOUND_TOKEN_MERGES.get(window)
            if fused_name is not None:
                merged.append((fused_name, tokens[i][1], tokens[i + length - 1][2]))
                i += length
                break
        else:
            merged.append(tokens[i])
            i += 1
    return merged


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

DEFAULT_LEXICON_MODE = WITH_BRANDS_CONTEXT

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


def _load_name_list_resource(filename: str) -> set[str]:
    data = _read_resource(filename)
    return {
        line.strip().lower()
        for line in data.splitlines()
        if line.strip() and not line.startswith("#")
    }


def _load_organism_exclusions() -> set[str]:
    """Committed list of organism species names (culture/sensitivity results, allergens)
    that happen to be RxNorm IN/PIN/BN terms but are never themselves a medication. Unlike
    the ambiguity list, no clinical context can rescue these -- they're dropped outright."""
    return _load_name_list_resource("organism_exclusions.txt")


def _load_diagnostic_agent_exclusions() -> set[str]:
    """Committed list of imaging/diagnostic agents (contrast media, dyes) that are RxNorm
    terms but are administered to image or test, never to treat. Like organism names, no
    clinical context can rescue these -- they're dropped outright."""
    return _load_name_list_resource("diagnostic_agent_exclusions.txt")


def _load_entries(mode: str) -> list[LexiconEntry]:
    """Load the lexicon entries a mode admits.

    "handwritten" reads the original committed list, so it stays byte-for-byte the
    behaviour the extractor had before RxNorm was introduced.
    """
    if mode == HANDWRITTEN:
        return [LexiconEntry(name, None, None, False) for name in _load_lexicon()]

    allowed = _MODE_TERM_TYPES[mode]
    excluded = _load_organism_exclusions() | _load_diagnostic_agent_exclusions()
    entries = []
    rows = csv.DictReader(_read_resource("rxnorm_lexicon.tsv").splitlines(), delimiter="\t")
    for row in rows:
        if row["tty"] not in allowed:
            continue
        if row["name"] in excluded:
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


def lookup(name: str, mode: str = DEFAULT_LEXICON_MODE) -> LexiconEntry | None:
    """Look up a single name in a mode's lexicon index by its tokenized form.

    For callers that have a drug name from elsewhere (e.g. a weak-label source) and need
    its RxNorm provenance/ambiguity flag, without running full extraction over a note.
    """
    index = _get_index(_check_mode(mode))
    key = tuple(token for token, _, _ in _tokenize(name))
    return index.entries.get(key)


def _lexicon_filename(mode: str) -> str:
    return "drug_lexicon.txt" if _check_mode(mode) == HANDWRITTEN else "rxnorm_lexicon.tsv"


def lexicon_sha256(mode: str = DEFAULT_LEXICON_MODE) -> str:
    """Hash of the file a mode reads, so a run can be tied to an exact lexicon."""
    data = _read_resource(_lexicon_filename(mode)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def rxnorm_version(mode: str = DEFAULT_LEXICON_MODE) -> str | None:
    """RxNorm release a mode's lexicon was built from, or None for the hand-written list."""
    if _check_mode(mode) == HANDWRITTEN:
        return None
    rows = csv.DictReader(_read_resource("rxnorm_lexicon.tsv").splitlines(), delimiter="\t")
    first = next(rows, None)
    return first["rxnorm_version"] if first else None


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
    line: str,
    tokens: list[tuple[str, int, int]],
    starts: dict[int, int],
    matches: list[tuple[LexiconEntry, str, int, int]],
) -> list[tuple[LexiconEntry, str, int, int]]:
    """Drop ambiguous names that no nearby clinical context supports.

    Runs before attribute windows are computed: a window ends at the next match's start,
    so a name dropped afterwards would still have truncated the previous drug's window.
    """
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


def _apply_lab_context_filter(
    line: str, matches: list[tuple[LexiconEntry, str, int, int]]
) -> list[tuple[LexiconEntry, str, int, int]]:
    """Drop matches used as a lab result, receptor name, or resistance profile rather than
    a medication, regardless of lexicon mode or the ambiguity flag: "insulin receptor" and
    "penicillin-resistant" are unambiguous RxNorm ingredients that the ambiguity gate never
    looks at, yet are never themselves a prescribed medication here."""
    kept = []
    for entry, drug, start, end in matches:
        after = line[end : end + _LAB_CONTEXT_CHARS]
        before = line[max(0, start - _LAB_CONTEXT_CHARS) : start]
        if (
            _LAB_SUFFIX_RE.search(after)
            or _LAB_PREFIX_RE.search(before)
            or (_BUN_PREFIX_RE.search(before) and _BUN_SUFFIX_RE.search(after))
        ):
            logger.debug("name %r dropped: lab/test context", drug)
            continue
        kept.append((entry, drug, start, end))
    return kept


def _sentence_end(line: str, pos: int) -> int:
    """End offset of the sentence containing pos: the next "[.!?]" + whitespace break at
    or after pos, or the end of the line. A decimal point is never followed by whitespace,
    so this never splits a dose like "0.5 mg" in half."""
    match = _SENTENCE_BREAK_RE.search(line, pos)
    return match.start() if match else len(line)


def _sentence_start(line: str, pos: int) -> int:
    """Start offset of the sentence containing pos: the end of the last "[.!?]" +
    whitespace break before pos, or the start of the line."""
    start = 0
    for match in _SENTENCE_BREAK_RE.finditer(line, 0, pos):
        start = match.end()
    return start


# A strength/dosage/route match that reaches the very end of the text it's searched
# against -- used to find an attribute that directly precedes a drug name, e.g. "100 mg
# propofol", "topical atropine", or "4 g of acetaminophen": that attribute belongs to the
# following drug, never to one mentioned earlier in the same sentence. The "value" group
# isolates the attribute itself from the optional trailing "of" ("4 g" rather than "4 g
# of"); strength/dosage allow it ("NUMBER unit of DRUG" is a common dosing phrasing),
# route does not ("topical of atropine" is not idiomatic).
_PRECEDING_STRENGTH_RE = re.compile(
    rf"(?P<value>{_STRENGTH_RE.pattern})\s*(?:of\s*)?\Z", re.IGNORECASE
)
_PRECEDING_DOSAGE_RE = re.compile(rf"(?P<value>{_DOSAGE_RE.pattern})\s*(?:of\s*)?\Z", re.IGNORECASE)
_PRECEDING_ROUTE_RE = re.compile(rf"(?P<value>{_ROUTE_RE.pattern})\s*\Z", re.IGNORECASE)

# A route immediately after a drug's own name (optionally after its own strength/dosage,
# e.g. "lisinopril 10 mg PO") belongs to that drug. Anchored at the very start of the text
# it's searched against (the text right after the drug match) so a route describing some
# later, unrelated word ("albuterol and occasional inhaled corticosteroids") is not picked
# up just because it's somewhere in the forward window.
_FORWARD_ROUTE_RE = re.compile(
    rf"\A\s*(?:{_STRENGTH_RE.pattern}|{_DOSAGE_RE.pattern})?\s*(?P<route>{_ROUTE_RE.pattern})",
    re.IGNORECASE,
)


def _first_match(pattern: re.Pattern[str], line: str) -> str | None:
    match = pattern.search(line)
    return match.group(0) if match else None


def _dosage_match(segment: str) -> str | None:
    """Like _first_match(_DOSAGE_RE, segment), but a volume given as a diluent ("...units
    in 250 mL D5W") is not a dosage -- the drug's own dose is the strength, not how much
    fluid it was mixed into. Scoped to "ml": puffs/tablets/capsules/drops don't describe
    a diluent volume, so "given in 2 tablets" (if it ever occurs) is unaffected."""
    match = _DOSAGE_RE.search(segment)
    if match is None:
        return None
    if match.group(0).lower().endswith("ml") and _DILUENT_VOLUME_RE.search(
        segment[: match.start()]
    ):
        return None
    return match.group(0)


def _forward_route_match(text_after_drug: str) -> str | None:
    """A route immediately after the drug (optionally after its own strength/dosage),
    anchored so a route describing a later, unrelated word is never picked up."""
    match = _FORWARD_ROUTE_RE.search(text_after_drug)
    return match.group("route") if match else None


def _preceding_attributes(
    line: str, drug_matches: list[tuple[LexiconEntry, str, int, int]]
) -> tuple[dict[int, str], dict[int, str], dict[int, str], dict[int, int]]:
    """Find, for each drug match, a strength/dosage/route that directly precedes it in
    the same sentence (e.g. "100 mg propofol, and 50 mg rocuronium", "topical atropine,
    and oral prednisolone"): an attribute immediately followed by a drug name belongs to
    that drug, never to one mentioned earlier. Returns (strengths, dosages, routes,
    claim_starts) by match index; claim_starts lets the PRECEDING drug's own forward
    window stop before an attribute it doesn't own.
    """
    strengths: dict[int, str] = {}
    dosages: dict[int, str] = {}
    routes: dict[int, str] = {}
    claim_starts: dict[int, int] = {}

    for i, (_entry, _drug, drug_start, _drug_end) in enumerate(drug_matches):
        gap_start = max(
            drug_matches[i - 1][3] if i > 0 else 0,
            _sentence_start(line, drug_start),
        )
        gap_text = line[gap_start:drug_start]
        claim_start = None

        for pattern, field in (
            (_PRECEDING_STRENGTH_RE, strengths),
            (_PRECEDING_DOSAGE_RE, dosages),
            (_PRECEDING_ROUTE_RE, routes),
        ):
            match = pattern.search(gap_text)
            if match is None:
                continue
            field[i] = match.group("value")
            match_start = gap_start + match.start()
            claim_start = match_start if claim_start is None else min(claim_start, match_start)

        if claim_start is not None:
            claim_starts[i] = claim_start

    return strengths, dosages, routes, claim_starts


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
        drug_matches = _apply_lab_context_filter(line, drug_matches)

        tokens = _tokenize(line)
        starts = {token[1]: idx for idx, token in enumerate(tokens)}
        if index.requires_context:
            drug_matches = _apply_context_gate(line, tokens, starts, drug_matches)

        preceding_strengths, preceding_dosages, preceding_routes, claim_starts = (
            _preceding_attributes(line, drug_matches)
        )

        for i, (entry, drug, drug_start, drug_end) in enumerate(drug_matches):
            segment_start = drug_start
            segment_end = drug_matches[i + 1][2] if i + 1 < len(drug_matches) else len(line)
            segment_end = min(segment_end, _sentence_end(line, drug_start))
            if i + 1 in claim_starts:
                # The next drug's attribute precedes it in this gap; don't claim it here.
                segment_end = min(segment_end, claim_starts[i + 1])
            segment = line[segment_start:segment_end]

            has_context = _has_clinical_context(
                line, tokens, starts[drug_start], len(drug.split(" "))
            )

            logger.debug("drug match: %r at [%d, %d)", drug, segment_start, segment_end)
            logger.debug("attribute window: %r", segment)

            strength = preceding_strengths.get(i) or _first_match(_STRENGTH_RE, segment)
            logger.debug("strength: %s", strength or "not found")

            dosage = preceding_dosages.get(i) or _dosage_match(segment)
            logger.debug("dosage: %s", dosage or "not found")

            route = preceding_routes.get(i) or _forward_route_match(line[drug_end:segment_end])
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
                    rxcui=entry.rxcui,
                    tty=entry.tty,
                    has_context=has_context,
                    evidence=line,
                    start=line_start,
                    end=line_end,
                )
            )

    return medications
