import json

from lablens.grounding import GroundingStats
from lablens.prepare_t5_data import build_target_medications, ground_target_medications


def test_builds_medications_from_dict_treatments():
    summary = {
        "treatments": [
            {"name": "Lisinopril", "dosage": "10 mg", "frequency": "daily", "duration": "30 days"}
        ]
    }

    medications = build_target_medications(summary)

    assert medications == [
        {"drug": "Lisinopril", "strength": "10 mg", "frequency": "daily", "duration": "30 days"}
    ]


def test_parses_string_encoded_summary():
    summary_raw = json.dumps(
        {"treatments": [{"name": "Metformin", "dosage": "500 mg", "frequency": None}]}
    )

    medications = build_target_medications(summary_raw)

    assert medications == [
        {"drug": "Metformin", "strength": "500 mg", "frequency": None, "duration": None}
    ]


def test_parses_string_encoded_treatments():
    summary = {"treatments": json.dumps([{"name": "Aspirin", "dosage": "81 mg"}])}

    medications = build_target_medications(summary)

    assert medications == [
        {"drug": "Aspirin", "strength": "81 mg", "frequency": None, "duration": None}
    ]


def test_treats_none_string_and_empty_values_as_null():
    summary = {
        "treatments": [
            {"name": "Warfarin", "dosage": "None", "frequency": "  ", "duration": "5 days"}
        ]
    }

    medications = build_target_medications(summary)

    assert medications == [
        {"drug": "Warfarin", "strength": None, "frequency": None, "duration": "5 days"}
    ]


def test_skips_entries_with_no_name():
    summary = {"treatments": [{"dosage": "10 mg"}, {"name": "", "dosage": "5 mg"}]}

    medications = build_target_medications(summary)

    assert medications == []


def test_skips_non_dict_treatment_entries():
    summary = {"treatments": ["rest and fluids", {"name": "Ibuprofen", "dosage": "200 mg"}]}

    medications = build_target_medications(summary)

    assert medications == [
        {"drug": "Ibuprofen", "strength": "200 mg", "frequency": None, "duration": None}
    ]


def test_missing_or_malformed_treatments_yields_empty_list():
    assert build_target_medications({}) == []
    assert build_target_medications({"treatments": "not json"}) == []
    assert build_target_medications({"treatments": None}) == []
    assert build_target_medications("not json either") == []


def test_ground_target_medications_keeps_grounded_drug_and_fields():
    medications = [
        {"drug": "Lisinopril", "strength": "10 mg", "frequency": "daily", "duration": None}
    ]

    grounded = ground_target_medications(medications, "Lisinopril 10 mg PO daily")

    assert grounded == [
        {"drug": "Lisinopril", "strength": "10 mg", "frequency": "daily", "duration": None}
    ]


def test_ground_target_medications_drops_medication_with_ungrounded_drug():
    stats = GroundingStats()
    medications = [{"drug": "Metformin", "strength": "500 mg", "frequency": None, "duration": None}]

    grounded = ground_target_medications(medications, "Patient takes Aspirin daily", stats)

    assert grounded == []
    assert stats.ungrounded_drugs == 1


def test_ground_target_medications_nulls_ungrounded_field_without_dropping_drug():
    stats = GroundingStats()
    medications = [
        {"drug": "Ibuprofen", "strength": "801 mg", "frequency": "TID", "duration": None}
    ]

    grounded = ground_target_medications(medications, "Ibuprofen 800 mg PO TID", stats)

    assert grounded == [
        {"drug": "Ibuprofen", "strength": None, "frequency": "TID", "duration": None}
    ]
    assert stats.ungrounded_values == 1
    assert stats.ungrounded_drugs == 0


def test_ground_target_medications_matches_drug_names_on_word_boundaries():
    """A short drug name must not match as a substring inside an unrelated word."""
    stats = GroundingStats()
    medications = [{"drug": "Iron", "strength": None, "frequency": None, "duration": None}]

    grounded = ground_target_medications(medications, "Patient lives in a safe environment", stats)

    assert grounded == []
    assert stats.ungrounded_drugs == 1


def test_ground_target_medications_rescues_drug_via_trailing_parenthetical():
    """A class/procedure wrapper around a real drug name should fall back to the inner name."""
    stats = GroundingStats()
    medications = [
        {
            "drug": "Corticosteroids (prednisone)",
            "strength": None,
            "frequency": None,
            "duration": None,
        }
    ]

    grounded = ground_target_medications(medications, "Patient started on prednisone 10 mg", stats)

    assert grounded == [
        {"drug": "prednisone", "strength": None, "frequency": None, "duration": None}
    ]
    assert stats.rescued_drugs == 1
    assert stats.ungrounded_drugs == 0


def test_ground_target_medications_drops_when_parenthetical_text_also_ungrounded():
    stats = GroundingStats()
    medications = [
        {
            "drug": "Corticosteroids (prednisone)",
            "strength": None,
            "frequency": None,
            "duration": None,
        }
    ]

    grounded = ground_target_medications(medications, "Patient started on a steroid taper", stats)

    assert grounded == []
    assert stats.ungrounded_drugs == 1
    assert stats.rescued_drugs == 0
