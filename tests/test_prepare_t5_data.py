import json

from lablens import prepare_t5_data
from lablens.extract_rules import LexiconEntry
from lablens.grounding import GroundingStats
from lablens.prepare_t5_data import (
    MergeStats,
    _project_for_target,
    _write_jsonl,
    build_target_medications,
    ground_target_medications,
    merge_medications,
)
from lablens.schema import Medication
from lablens.t5_format import linearize_medications


def _med(drug: str, **kwargs) -> Medication:
    """Build a minimal valid Medication for hand-made fixtures."""
    kwargs.setdefault("evidence", drug)
    start = kwargs.pop("start", 0)
    end = kwargs.pop("end", start + max(len(drug), 1))
    return Medication(drug=drug, start=start, end=end, **kwargs)


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


def test_write_jsonl_uses_lf_only_no_crlf(tmp_path):
    """An unspecified newline mode lets Windows translate "\\n" to "\\r\\n" on write,
    which would make data_sha256 differ between Windows and Linux for identical content."""
    path = tmp_path / "out.jsonl"
    examples = [{"input": "a", "target": "b"}, {"input": "c", "target": "d"}]

    _write_jsonl(path, examples)

    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")


def test_write_jsonl_is_byte_identical_across_runs(tmp_path):
    examples = [{"input": "a", "target": "b"}, {"input": "c", "target": "d"}]
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"

    _write_jsonl(path_a, examples)
    _write_jsonl(path_b, examples)

    assert path_a.read_bytes() == path_b.read_bytes()


def test_merge_matched_by_name_keeps_b_attributes_and_fills_missing_fields(monkeypatch):
    """A drug in both sources, same normalized name: B's attributes win; A only fills a
    field B left null, and never overwrites a field B already has."""
    monkeypatch.setattr(prepare_t5_data.extract_rules, "lookup", lambda name, mode=None: None)
    gpt_meds = [
        {"drug": "Lisinopril", "strength": "10 mg", "frequency": None, "duration": "30 days"}
    ]
    rule_meds = [_med("lisinopril", dosage="1 tablet", route="oral", frequency="daily")]
    stats = MergeStats()

    merged = merge_medications(gpt_meds, rule_meds, stats)

    assert merged == [
        {
            "drug": "lisinopril",
            "strength": "10 mg",  # B had none, filled from A
            "dosage": "1 tablet",
            "route": "oral",
            "frequency": "daily",  # B's value, not overwritten by A
            "duration": "30 days",  # B had none, filled from A
            "sources": ("gpt", "rules"),
        }
    ]
    assert stats.both == 1
    assert stats.gpt_only == 0
    assert stats.rules_only == 0


def test_merge_matches_by_rxcui_when_normalized_names_differ(monkeypatch):
    """A brand name from GPT and an ingredient name from the rules extractor merge as
    "both" when they share an RxCUI, even though their normalized names don't match."""
    monkeypatch.setattr(
        prepare_t5_data.extract_rules,
        "lookup",
        lambda name, mode=None: LexiconEntry("tylenol", "161", "BN", False),
    )
    gpt_meds = [{"drug": "Tylenol", "strength": None, "frequency": "q6h", "duration": None}]
    rule_meds = [_med("acetaminophen", rxcui="161", strength="500 mg")]
    stats = MergeStats()

    merged = merge_medications(gpt_meds, rule_meds, stats)

    assert merged == [
        {
            "drug": "acetaminophen",
            "strength": "500 mg",
            "dosage": None,
            "route": None,
            "frequency": "q6h",
            "duration": None,
            "sources": ("gpt", "rules"),
        }
    ]
    assert stats.both == 1


def test_merge_keeps_gpt_only_drug_when_in_lexicon_and_not_ambiguous(monkeypatch):
    monkeypatch.setattr(
        prepare_t5_data.extract_rules,
        "lookup",
        lambda name, mode=None: LexiconEntry("warfarin", "11289", "IN", False),
    )
    gpt_meds = [{"drug": "Warfarin", "strength": "5 mg", "frequency": None, "duration": None}]
    stats = MergeStats()

    merged = merge_medications(gpt_meds, [], stats)

    assert merged == [
        {
            "drug": "Warfarin",
            "strength": "5 mg",
            "dosage": None,
            "route": None,
            "frequency": None,
            "duration": None,
            "sources": ("gpt",),
        }
    ]
    assert stats.gpt_only == 1
    assert stats.rejected_not_in_rxnorm == 0


def test_merge_rejects_gpt_only_drug_not_in_rxnorm_lexicon(monkeypatch):
    monkeypatch.setattr(prepare_t5_data.extract_rules, "lookup", lambda name, mode=None: None)
    gpt_meds = [
        {"drug": "HerbalSupplementX", "strength": None, "frequency": None, "duration": None}
    ]
    stats = MergeStats()

    merged = merge_medications(gpt_meds, [], stats)

    assert merged == []
    assert stats.rejected_not_in_rxnorm == 1
    assert stats.rejected_examples == ["HerbalSupplementX"]


def test_merge_rejects_gpt_only_drug_that_is_ambiguous_only(monkeypatch):
    monkeypatch.setattr(
        prepare_t5_data.extract_rules,
        "lookup",
        lambda name, mode=None: LexiconEntry("iron", None, None, True),
    )
    gpt_meds = [{"drug": "Iron", "strength": None, "frequency": None, "duration": None}]
    stats = MergeStats()

    merged = merge_medications(gpt_meds, [], stats)

    assert merged == []
    assert stats.rejected_not_in_rxnorm == 1
    assert stats.rejected_examples == ["Iron"]


def test_merge_keeps_rules_only_drug_and_counts_its_route(monkeypatch):
    monkeypatch.setattr(prepare_t5_data.extract_rules, "lookup", lambda name, mode=None: None)
    rule_meds = [_med("amlodipine", strength="5 mg", route="oral", frequency="daily")]
    stats = MergeStats()

    merged = merge_medications([], rule_meds, stats)

    assert merged == [
        {
            "drug": "amlodipine",
            "strength": "5 mg",
            "dosage": None,
            "route": "oral",
            "frequency": "daily",
            "duration": None,
            "sources": ("rules",),
        }
    ]
    assert stats.rules_only == 1
    assert stats.with_route == 1


def test_merge_dedupes_repeated_rule_mentions_keeping_the_first(monkeypatch):
    """The rules extractor finds a per-line match, so a drug mentioned on two lines
    (e.g. history and discharge meds) yields two Medication rows for one real drug."""
    monkeypatch.setattr(prepare_t5_data.extract_rules, "lookup", lambda name, mode=None: None)
    rule_meds = [
        _med("metformin", strength="500 mg", start=0, end=9),
        _med("metformin", strength="1000 mg", route="oral", start=20, end=29),
    ]

    merged = merge_medications([], rule_meds)

    assert merged == [
        {
            "drug": "metformin",
            "strength": "500 mg",
            "dosage": None,
            "route": None,
            "frequency": None,
            "duration": None,
            "sources": ("rules",),
        }
    ]


def test_project_for_target_drops_sources_but_keeps_dosage_and_route():
    medications = [
        {
            "drug": "metformin",
            "strength": "500 mg",
            "dosage": "1 tablet",
            "route": "oral",
            "frequency": "daily",
            "duration": None,
            "sources": ("gpt", "rules"),
        }
    ]

    projected = _project_for_target(medications)

    assert projected == [
        {
            "drug": "metformin",
            "strength": "500 mg",
            "dosage": "1 tablet",
            "route": "oral",
            "frequency": "daily",
            "duration": None,
        }
    ]
    assert linearize_medications(projected) == (
        "drug: metformin | strength: 500 mg | dosage: 1 tablet | route: oral | frequency: daily"
    )
