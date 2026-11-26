import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from lablens import rxnorm


def _mock_response(payload):
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    return response


def _concept(rxcui, name, tty):
    return {"rxcui": rxcui, "name": name, "tty": tty}


CONCEPTS = [
    _concept("29046", "lisinopril", "IN"),
    _concept("90176", "iron", "IN"),
    _concept("24947", "ferrous sulfate", "IN"),
    _concept("896747", "Today", "BN"),
]


@patch("lablens.rxnorm.urllib.request.urlopen")
def test_fetch_version_returns_the_rxnorm_release(mock_urlopen):
    mock_urlopen.return_value = _mock_response({"version": "02-Nov-2026"})

    assert rxnorm.fetch_version(timeout=5, retries=0) == "02-Nov-2026"


@patch("lablens.rxnorm.urllib.request.urlopen")
def test_fetch_concepts_returns_the_concept_list(mock_urlopen):
    mock_urlopen.return_value = _mock_response({"minConceptGroup": {"minConcept": CONCEPTS[:1]}})

    assert rxnorm.fetch_concepts("IN", timeout=5, retries=0) == CONCEPTS[:1]


@patch("lablens.rxnorm.urllib.request.urlopen")
def test_request_passes_the_timeout_through(mock_urlopen):
    mock_urlopen.return_value = _mock_response({"version": "02-Nov-2026"})

    rxnorm.fetch_version(timeout=7.5, retries=0)

    assert mock_urlopen.call_args.kwargs["timeout"] == 7.5


@patch("lablens.rxnorm.time.sleep")
@patch("lablens.rxnorm.urllib.request.urlopen")
def test_request_retries_then_succeeds(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = [
        urllib.error.URLError("connection refused"),
        _mock_response({"version": "02-Nov-2026"}),
    ]

    assert rxnorm.fetch_version(timeout=5, retries=1) == "02-Nov-2026"
    assert mock_urlopen.call_count == 2


@patch("lablens.rxnorm.time.sleep")
@patch("lablens.rxnorm.urllib.request.urlopen")
def test_request_raises_a_clear_error_when_every_attempt_fails(mock_urlopen, mock_sleep):
    mock_urlopen.side_effect = urllib.error.URLError("connection refused")

    with pytest.raises(rxnorm.RxNavError, match="after 3 attempts"):
        rxnorm.fetch_version(timeout=5, retries=2)


@patch("lablens.rxnorm.urllib.request.urlopen")
def test_empty_concept_list_is_an_error_rather_than_a_silently_empty_lexicon(mock_urlopen):
    mock_urlopen.return_value = _mock_response({"minConceptGroup": {}})

    with pytest.raises(rxnorm.RxNavError, match="no concepts"):
        rxnorm.fetch_concepts("IN", timeout=5, retries=0)


def test_rows_carry_rxcui_term_type_and_version_not_just_the_name():
    rows = rxnorm.build_rows(CONCEPTS, [], [], "02-Nov-2026")

    by_name = {row["name"]: row for row in rows}
    assert by_name["lisinopril"] == {
        "name": "lisinopril",
        "rxcui": "29046",
        "tty": "IN",
        "ambiguous": "0",
        "source": "rxnorm",
        "rxnorm_version": "02-Nov-2026",
    }


def test_concept_names_are_lowercased_so_matching_is_case_insensitive():
    rows = rxnorm.build_rows(CONCEPTS, [], [], "02-Nov-2026")

    assert "today" in {row["name"] for row in rows}


def test_handwritten_terms_rxnorm_does_not_know_are_kept_without_an_rxcui():
    rows = rxnorm.build_rows(CONCEPTS, ["insulin", "senna"], [], "02-Nov-2026")

    by_name = {row["name"]: row for row in rows}
    assert by_name["insulin"]["rxcui"] == ""
    assert by_name["insulin"]["tty"] == ""
    assert by_name["insulin"]["source"] == "handwritten"
    assert by_name["senna"]["source"] == "handwritten"


def test_handwritten_terms_rxnorm_knows_keep_their_rxcui_and_are_marked_handwritten():
    rows = rxnorm.build_rows(CONCEPTS, ["lisinopril"], [], "02-Nov-2026")

    by_name = {row["name"]: row for row in rows}
    assert by_name["lisinopril"]["rxcui"] == "29046"
    assert by_name["lisinopril"]["source"] == "handwritten"


def test_each_name_appears_exactly_once_so_the_extractor_cannot_emit_duplicates():
    duplicated = CONCEPTS + [_concept("999", "lisinopril", "PIN")]

    rows = rxnorm.build_rows(duplicated, ["lisinopril"], [], "02-Nov-2026")

    names = [row["name"] for row in rows]
    assert len(names) == len(set(names))


def test_ambiguous_terms_are_flagged_and_not_removed():
    rows = rxnorm.build_rows(CONCEPTS, [], ["iron"], "02-Nov-2026")

    by_name = {row["name"]: row for row in rows}
    assert by_name["iron"]["ambiguous"] == "1"
    assert by_name["lisinopril"]["ambiguous"] == "0"


def test_rows_are_sorted_so_the_file_is_stable_across_rebuilds():
    rows = rxnorm.build_rows(CONCEPTS, [], [], "02-Nov-2026")

    assert [row["name"] for row in rows] == sorted(row["name"] for row in rows)


def test_tsv_has_the_documented_header_and_one_line_per_row():
    rows = rxnorm.build_rows(CONCEPTS, [], [], "02-Nov-2026")

    text = rxnorm.format_tsv(rows)

    lines = text.splitlines()
    assert lines[0] == "name\trxcui\ttty\tambiguous\tsource\trxnorm_version"
    assert len(lines) == len(rows) + 1


def test_tsv_is_written_lf_only_so_it_hashes_identically_on_every_os(tmp_path):
    rows = rxnorm.build_rows(CONCEPTS, [], [], "02-Nov-2026")
    out = tmp_path / "lexicon.tsv"

    rxnorm.write_tsv(out, rows)

    assert b"\r\n" not in out.read_bytes()


def test_a_name_containing_a_tab_is_rejected_rather_than_corrupting_the_tsv():
    rows = [
        {
            "name": "bad\tname",
            "rxcui": "1",
            "tty": "IN",
            "ambiguous": "0",
            "source": "rxnorm",
            "rxnorm_version": "02-Nov-2026",
        }
    ]

    with pytest.raises(ValueError, match="tab or newline"):
        rxnorm.format_tsv(rows)


def test_committed_lists_are_readable_and_lowercase():
    handwritten = rxnorm.load_handwritten()
    ambiguous = rxnorm.load_ambiguity_list()

    assert "lisinopril" in handwritten
    assert "iron" in ambiguous
    assert all(name == name.lower() for name in handwritten + ambiguous)
    assert not any(name.startswith("#") for name in ambiguous)


def test_every_ambiguity_entry_matches_a_real_lexicon_term():
    """A typo in the curated list would silently gate nothing. Catch it here."""
    import csv
    from importlib import resources

    data = (
        resources.files("lablens")
        .joinpath("resources/rxnorm_lexicon.tsv")
        .read_text(encoding="utf-8")
    )
    names = {row["name"] for row in csv.DictReader(data.splitlines(), delimiter="\t")}

    assert set(rxnorm.load_ambiguity_list()) <= names


def test_committed_lexicon_keeps_every_handwritten_term():
    """Switching to RxNorm must not silently drop a drug the extractor used to find."""
    import csv
    from importlib import resources

    data = (
        resources.files("lablens")
        .joinpath("resources/rxnorm_lexicon.tsv")
        .read_text(encoding="utf-8")
    )
    names = {row["name"] for row in csv.DictReader(data.splitlines(), delimiter="\t")}

    assert set(rxnorm.load_handwritten()) <= names
