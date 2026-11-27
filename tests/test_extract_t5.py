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
def test_parses_dosage_and_route_and_grounds_them(mock_load):
    mock_load.return_value = _fake_model(
        "drug: aspirin | strength: 81 mg | dosage: 1 tablet | route: oral | frequency: daily"
    )

    medications = extract("Aspirin 81 mg, 1 tablet, oral, daily")

    assert len(medications) == 1
    med = medications[0]
    assert med.dosage == "1 tablet"
    assert med.route == "oral"


@patch("lablens.extract_t5._load_model")
def test_nulls_ungrounded_dosage_and_route_without_dropping_drug(mock_load):
    mock_load.return_value = _fake_model(
        "drug: aspirin | strength: 81 mg | dosage: 2 tablets | route: topical"
    )
    stats = GroundingStats()

    medications = extract("Aspirin 81 mg, 1 tablet, oral, daily", stats=stats)

    assert len(medications) == 1
    assert medications[0].dosage is None
    assert medications[0].route is None
    assert stats.ungrounded_values == 2


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


@patch("lablens.extract_t5._load_model")
def test_dedupes_exact_repeats_from_a_generation_loop(mock_load):
    """Flan-T5 can repeat the same medication many times in a row; only the first
    occurrence should survive, and the rest are counted as duplicates_removed."""
    repeated = " ; ".join(["drug: aspirin | strength: 81 mg | frequency: daily"] * 11)
    mock_load.return_value = _fake_model(repeated)
    stats = GroundingStats()

    medications = extract("Aspirin 81 mg PO daily", stats=stats)

    assert len(medications) == 1
    assert medications[0].drug == "aspirin"
    assert stats.duplicates_removed == 10


@patch("lablens.extract_t5._load_model")
def test_dedupe_is_case_and_whitespace_insensitive_on_drug_name(mock_load):
    """The drug name is normalized for the dedupe key, so "Aspirin" and "aspirin" (or
    repeated whitespace) still collapse to one entry."""
    generated = (
        "drug: Aspirin | strength: 81 mg | frequency: daily ; "
        "drug: aspirin  | strength: 81 mg | frequency: daily"
    )
    mock_load.return_value = _fake_model(generated)
    stats = GroundingStats()

    medications = extract("Aspirin 81 mg PO daily", stats=stats)

    assert len(medications) == 1
    assert stats.duplicates_removed == 1


@patch("lablens.extract_t5._load_model")
def test_dedupe_keeps_entries_that_differ_in_any_field(mock_load):
    """Same drug name but a different field (here strength) is not a duplicate."""
    generated = (
        "drug: aspirin | strength: 81 mg | frequency: daily ; "
        "drug: aspirin | strength: 325 mg | frequency: daily"
    )
    mock_load.return_value = _fake_model(generated)
    stats = GroundingStats()

    medications = extract("Aspirin 81 mg or 325 mg PO daily", stats=stats)

    assert len(medications) == 2
    assert stats.duplicates_removed == 0


@patch("lablens.extract_t5._load_model")
def test_dedupe_drops_a_less_complete_subset_copy(mock_load):
    """A second entry whose only field (strength) agrees with the first, more complete
    entry is a redundant partial copy, not a distinct medication -- it's dropped and the
    more complete entry's fields (route, frequency) survive."""
    generated = (
        "drug: aspirin | strength: 81 mg | route: oral | frequency: daily ; "
        "drug: aspirin | strength: 81 mg"
    )
    mock_load.return_value = _fake_model(generated)
    stats = GroundingStats()

    medications = extract("Aspirin 81 mg oral daily", stats=stats)

    assert len(medications) == 1
    assert medications[0].strength == "81 mg"
    assert medications[0].route == "oral"
    assert medications[0].frequency == "daily"
    assert stats.duplicates_removed == 1


@patch("lablens.extract_t5._load_model")
def test_dedupe_never_merges_conflicting_strengths(mock_load):
    """Two entries for the same drug with different strengths conflict -- neither is a
    subset of the other, so both are kept rather than merged into one."""
    generated = (
        "drug: aspirin | strength: 81 mg | route: oral ; "
        "drug: aspirin | strength: 325 mg | route: oral"
    )
    mock_load.return_value = _fake_model(generated)
    stats = GroundingStats()

    medications = extract("Aspirin 81 mg or 325 mg oral", stats=stats)

    assert len(medications) == 2
    assert {m.strength for m in medications} == {"81 mg", "325 mg"}
    assert stats.duplicates_removed == 0


@patch("lablens.extract_t5._load_model")
def test_two_distinct_medications_each_keep_their_own_fields(mock_load):
    """Guards against reintroducing no_repeat_ngram_size: the linearized format repeats
    "| strength:", "| frequency:" etc. for every medication, so an n-gram repetition
    penalty would block a second medication's fields from being generated at all."""
    generated = (
        "drug: lisinopril | strength: 10 mg | route: oral | frequency: daily ; "
        "drug: metformin | strength: 500 mg | route: oral | frequency: BID"
    )
    mock_load.return_value = _fake_model(generated)

    medications = extract("Lisinopril 10 mg oral daily. Metformin 500 mg oral BID.")

    assert len(medications) == 2
    assert medications[0].drug == "lisinopril"
    assert medications[0].strength == "10 mg"
    assert medications[0].route == "oral"
    assert medications[0].frequency == "daily"
    assert medications[1].drug == "metformin"
    assert medications[1].strength == "500 mg"
    assert medications[1].route == "oral"
    assert medications[1].frequency == "BID"
