from unittest.mock import MagicMock, patch

from lablens import extract_t5
from lablens.extract_t5 import extract
from lablens.grounding import GroundingStats


def _fake_model(generated_text: str) -> tuple[MagicMock, MagicMock]:
    tokenizer = MagicMock()
    tokenizer.return_value = {"input_ids": [[0]], "attention_mask": [[1]]}
    tokenizer.decode.return_value = generated_text
    model = MagicMock()
    model.generate.return_value = [[0]]
    return tokenizer, model


def test_load_model_is_cached():
    assert hasattr(extract_t5._load_model, "cache_info")


@patch("lablens.extract_t5._load_model")
def test_parses_valid_target_and_grounds(mock_load):
    mock_load.return_value = _fake_model("drug: aspirin | strength: 81 mg | frequency: daily")

    medications = extract("Aspirin 81 mg PO daily")

    assert len(medications) == 1
    med = medications[0]
    assert med.drug == "aspirin"
    assert med.strength == "81 mg"
    assert med.frequency == "daily"
    assert med.duration is None
    assert med.evidence == "Aspirin 81 mg PO daily"
    assert med.start == 0
    assert med.end == len("Aspirin 81 mg PO daily")


@patch("lablens.extract_t5._load_model")
def test_malformed_text_counts_parse_failure_and_returns_empty(mock_load):
    mock_load.return_value = _fake_model("this is not the expected format")
    stats = GroundingStats()

    medications = extract("Aspirin 81 mg PO daily", stats=stats)

    assert medications == []
    assert stats.parse_failures == 1


@patch("lablens.extract_t5._load_model")
def test_none_literal_returns_empty_list_without_parse_failure(mock_load):
    mock_load.return_value = _fake_model("none")
    stats = GroundingStats()

    medications = extract("Follow up with your doctor in one week.", stats=stats)

    assert medications == []
    assert stats.parse_failures == 0


@patch("lablens.extract_t5._load_model")
def test_drops_drug_not_grounded_in_text(mock_load):
    mock_load.return_value = _fake_model("drug: metformin")
    stats = GroundingStats()

    medications = extract("Patient takes Aspirin daily", stats=stats)

    assert medications == []
    assert stats.ungrounded_drugs == 1


@patch("lablens.extract_t5._load_model")
def test_nulls_ungrounded_field_without_dropping_drug(mock_load):
    mock_load.return_value = _fake_model("drug: ibuprofen | strength: 801 mg | frequency: TID")
    stats = GroundingStats()

    medications = extract("Ibuprofen 800 mg PO TID", stats=stats)

    assert len(medications) == 1
    assert medications[0].strength is None
    assert medications[0].frequency == "TID"
    assert stats.ungrounded_values == 1
    assert stats.ungrounded_drugs == 0


@patch("lablens.extract_t5._load_model")
def test_uses_given_model_dir(mock_load):
    mock_load.return_value = _fake_model("none")

    extract("text", model_dir="models/custom")

    mock_load.assert_called_once_with("models/custom")
