"""Build the medication lexicon from the NLM RxNav API.

Runnable as: python -m lablens.rxnorm [--out PATH] [--timeout SECONDS] [--retries N]

Fetches RxNorm concepts for three term types -- IN (ingredient), PIN (precise ingredient)
and BN (brand name) -- and writes them to a TSV with provenance: the RxCUI, the term type,
the RxNorm version, and whether the name came from RxNorm or the hand-written lexicon.

Privacy: only the RxNorm vocabulary is fetched. No document text is ever sent anywhere.
The hand-written lexicon and the ambiguity list are committed inputs, never overwritten,
so re-running this script is idempotent and cannot destroy hand curation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from importlib import resources
from pathlib import Path

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
TERM_TYPES = ("IN", "PIN", "BN")

COLUMNS = ("name", "rxcui", "tty", "ambiguous", "source", "rxnorm_version")

SOURCE_RXNORM = "rxnorm"
SOURCE_HANDWRITTEN = "handwritten"

DEFAULT_OUT = Path(__file__).parent / "resources" / "rxnorm_lexicon.tsv"


class RxNavError(RuntimeError):
    """Raised when RxNav cannot be reached or returns something unusable."""


def _get_json(url: str, *, timeout: float, retries: int, backoff: float = 1.0) -> dict:
    """GET a JSON document, retrying with linear backoff on network errors."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise RxNavError(f"Could not fetch {url} after {retries + 1} attempts: {last_error}")


def fetch_version(*, timeout: float, retries: int) -> str:
    """Return the RxNorm release version, e.g. "02-Nov-2026"."""
    data = _get_json(f"{RXNAV_BASE}/version.json", timeout=timeout, retries=retries)
    version = data.get("version")
    if not version:
        raise RxNavError("RxNav did not report a version")
    return version


def fetch_concepts(tty: str, *, timeout: float, retries: int) -> list[dict[str, str]]:
    """Return every RxNorm concept of one term type as {rxcui, name, tty} dicts."""
    data = _get_json(f"{RXNAV_BASE}/allconcepts.json?tty={tty}", timeout=timeout, retries=retries)
    concepts = data.get("minConceptGroup", {}).get("minConcept")
    if not concepts:
        raise RxNavError(f"RxNav returned no concepts for tty={tty}")
    return concepts


def _read_lines(path_or_text: str) -> list[str]:
    """Parse a committed list file: one lowercase name per line, # comments ignored."""
    names = []
    for raw in path_or_text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.append(line.lower())
    return names


def load_handwritten() -> list[str]:
    data = (
        resources.files("lablens")
        .joinpath("resources/drug_lexicon.txt")
        .read_text(encoding="utf-8")
    )
    return _read_lines(data)


def load_ambiguity_list() -> list[str]:
    data = (
        resources.files("lablens")
        .joinpath("resources/ambiguity_list.txt")
        .read_text(encoding="utf-8")
    )
    return _read_lines(data)


def build_rows(
    concepts: list[dict[str, str]],
    handwritten: list[str],
    ambiguous: list[str],
    version: str,
) -> list[dict[str, str]]:
    """Join RxNorm concepts with the two committed lists into sorted lexicon rows.

    A name is kept once. Hand-written names win the source tie-break so the original
    lexicon stays traceable, and hand-written names RxNorm does not know (insulin, senna)
    are kept with an empty rxcui rather than silently dropped.
    """
    ambiguous_set = set(ambiguous)
    handwritten_set = set(handwritten)

    rows: dict[str, dict[str, str]] = {}
    for concept in concepts:
        name = concept["name"].strip().lower()
        if not name or name in rows:
            continue
        rows[name] = {
            "name": name,
            "rxcui": concept["rxcui"],
            "tty": concept["tty"],
            "ambiguous": "1" if name in ambiguous_set else "0",
            "source": SOURCE_HANDWRITTEN if name in handwritten_set else SOURCE_RXNORM,
            "rxnorm_version": version,
        }

    for name in handwritten:
        if name in rows:
            rows[name]["source"] = SOURCE_HANDWRITTEN
            continue
        rows[name] = {
            "name": name,
            "rxcui": "",
            "tty": "",
            "ambiguous": "1" if name in ambiguous_set else "0",
            "source": SOURCE_HANDWRITTEN,
            "rxnorm_version": version,
        }

    return [rows[name] for name in sorted(rows)]


def format_tsv(rows: list[dict[str, str]]) -> str:
    """Render rows as a TSV string with a header line."""
    lines = ["\t".join(COLUMNS)]
    for row in rows:
        values = [row[column] for column in COLUMNS]
        for value in values:
            if "\t" in value or "\n" in value:
                raise ValueError(f"value contains a tab or newline: {value!r}")
        lines.append("\t".join(values))
    return "\n".join(lines) + "\n"


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write LF-only, UTF-8 TSV so the same input hashes identically on every OS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(format_tsv(rows))


def _report(rows: list[dict[str, str]], handwritten: list[str], ambiguous: list[str]) -> None:
    by_tty: dict[str, int] = {}
    for row in rows:
        by_tty[row["tty"] or "(none)"] = by_tty.get(row["tty"] or "(none)", 0) + 1

    confirmed = [n for n in handwritten if any(r["name"] == n and r["rxcui"] for r in rows)]
    unconfirmed = [n for n in handwritten if n not in confirmed]
    flagged = sum(1 for row in rows if row["ambiguous"] == "1")
    unmatched = sorted(set(ambiguous) - {row["name"] for row in rows})

    print(f"rows: {len(rows)}")
    for tty in sorted(by_tty):
        print(f"  {tty}: {by_tty[tty]}")
    print(f"hand-written terms: {len(handwritten)}")
    print(f"  confirmed by RxNorm: {len(confirmed)}")
    print(f"  kept without an rxcui: {len(unconfirmed)} {unconfirmed}")
    print(f"ambiguous terms flagged: {flagged}")
    if unmatched:
        print(f"  ambiguity entries matching no lexicon term: {len(unmatched)} {unmatched}")


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Build the RxNorm medication lexicon.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Where to write the lexicon TSV (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retries per request after the first attempt (default: 3).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    version = fetch_version(timeout=args.timeout, retries=args.retries)
    print(f"RxNorm version: {version}")

    concepts: list[dict[str, str]] = []
    for tty in TERM_TYPES:
        fetched = fetch_concepts(tty, timeout=args.timeout, retries=args.retries)
        print(f"fetched {len(fetched)} {tty} concepts")
        concepts.extend(fetched)

    handwritten = load_handwritten()
    ambiguous = load_ambiguity_list()
    rows = build_rows(concepts, handwritten, ambiguous, version)

    write_tsv(args.out, rows)
    _report(rows, handwritten, ambiguous)
    print(f"wrote {args.out}")
    return args.out


if __name__ == "__main__":
    main()
