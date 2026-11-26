import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from lablens.extract_llm import GroundingStats, OllamaNotRunningError, extract


def _mock_response(parsed_content: dict) -> MagicMock:
    response = MagicMock()
    response.read.return_value = json.dumps(
        {"message": {"content": json.dumps(parsed_content)}}
    ).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


@patch("lablens.extract_llm.urllib.request.urlopen")
def test_parses_medications_from_ollama_response(mock_urlopen):
    mock_urlopen.return_value = _mock_response(
        {
            "medications": [
                {
                    "drug": "aspirin",
                    "strength": "81 mg",
                    "dosage": None,
                    "route": "PO",
                    "frequency": "daily",
                    "duration": None,
                }
            ]
        }
    )

    medications = extract("Aspirin 81 mg PO daily")

    assert len(medications) == 1
    med = medications[0]
    assert med.drug == "aspirin"
    assert med.strength == "81 mg"
    assert med.route == "PO"
    assert med.frequency == "daily"
    assert med.duration is None
    assert med.evidence == "Aspirin 81 mg PO daily"
    assert med.start == 0
    assert med.end == len("Aspirin 81 mg PO daily")


@patch("lablens.extract_llm.urllib.request.urlopen")
def test_grounding_check_drops_drug_not_in_text(mock_urlopen):
    mock_urlopen.return_value = _mock_response(
        {
            "medications": [
                {
                    "drug": "aspirin",
                    "strength": None,
                    "dosage": None,
                    "route": None,
                    "frequency": None,
                    "duration": None,
                },
                {
                    "drug": "metformin",
                    "strength": None,
                    "dosage": None,
                    "route": None,
                    "frequency": None,
                    "duration": None,
                },
            ]
        }
    )

    medications = extract("Patient takes Aspirin daily")

    drugs = [med.drug for med in medications]
    assert drugs == ["aspirin"]


@patch("lablens.extract_llm.urllib.request.urlopen")
def test_value_grounding_drops_value_not_in_evidence_line(mock_urlopen):
    mock_urlopen.return_value = _mock_response(
        {
            "medications": [
                {
                    "drug": "ibuprofen",
                    "strength": "801 mg",
                    "dosage": None,
                    "route": "PO",
                    "frequency": "TID PRN",
                    "duration": None,
                }
            ]
        }
    )

    stats = GroundingStats()
    medications = extract("Ibuprofen 800 mg PO TID PRN", stats=stats)

    assert len(medications) == 1
    med = medications[0]
    assert med.strength is None
    assert med.route == "PO"
    assert med.frequency == "TID PRN"
    assert stats.ungrounded_values == 1
    assert stats.ungrounded_drugs == 0


@patch("lablens.extract_llm.urllib.request.urlopen")
def test_value_grounding_rejects_substring_of_a_larger_number(mock_urlopen):
    mock_urlopen.return_value = _mock_response(
        {
            "medications": [
                {
                    "drug": "warfarin",
                    "strength": "5 mg",
                    "dosage": None,
                    "route": "PO",
                    "frequency": "daily",
                    "duration": None,
                }
            ]
        }
    )

    stats = GroundingStats()
    medications = extract("Warfarin 0.5 mg PO daily", stats=stats)

    assert medications[0].strength is None
    assert stats.ungrounded_values == 1


@patch("lablens.extract_llm.urllib.request.urlopen")
def test_value_grounding_rejects_substring_of_a_larger_number_no_decimal(mock_urlopen):
    mock_urlopen.return_value = _mock_response(
        {
            "medications": [
                {
                    "drug": "metoprolol",
                    "strength": "5 mg",
                    "dosage": None,
                    "route": None,
                    "frequency": "BID",
                    "duration": None,
                }
            ]
        }
    )

    stats = GroundingStats()
    medications = extract("Metoprolol 25 mg BID", stats=stats)

    assert medications[0].strength is None
    assert stats.ungrounded_values == 1


@patch("lablens.extract_llm.urllib.request.urlopen")
def test_value_grounding_accepts_exact_decimal_strength(mock_urlopen):
    mock_urlopen.return_value = _mock_response(
        {
            "medications": [
                {
                    "drug": "warfarin",
                    "strength": "0.5 mg",
                    "dosage": None,
                    "route": "PO",
                    "frequency": "daily",
                    "duration": None,
                }
            ]
        }
    )

    stats = GroundingStats()
    medications = extract("Warfarin 0.5 mg PO daily", stats=stats)

    assert medications[0].strength == "0.5 mg"
    assert stats.ungrounded_values == 0


@patch("lablens.extract_llm.urllib.request.urlopen")
def test_raises_clear_error_when_ollama_not_running(mock_urlopen):
    mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

    with pytest.raises(OllamaNotRunningError):
        extract("Aspirin 81 mg PO daily")
