import logging

import pytest

from lablens import extract_rules
from lablens.extract_rules import extract


def test_single_drug_all_fields():
    text = "Start lisinopril 10 mg PO daily for 30 days"
    meds = extract(text)

    assert len(meds) == 1
    med = meds[0]
    assert med.drug == "lisinopril"
    assert med.strength == "10 mg"
    assert med.route.lower() == "po"
    assert med.frequency.lower() == "daily"
    assert med.duration.lower() == "for 30 days"
    assert med.evidence == text
    assert text[med.start : med.end] == text


def test_two_drugs_on_separate_lines():
    text = "metformin 500 mg PO BID\nlisinopril 10 mg PO daily"
    meds = extract(text)

    assert len(meds) == 2
    assert meds[0].drug == "metformin"
    assert meds[0].strength == "500 mg"
    assert meds[1].drug == "lisinopril"
    assert meds[1].strength == "10 mg"
    for med in meds:
        assert text[med.start : med.end] == med.evidence


def test_decimal_strength_not_truncated():
    text = "amlodipine 0.5 mg PO daily"
    meds = extract(text)

    assert len(meds) == 1
    assert meds[0].strength == "0.5 mg"


def test_abbreviations_recognized():
    text = "aspirin 81 mg PO BID\nibuprofen 400 mg PO PRN"
    meds = extract(text)

    assert len(meds) == 2
    assert meds[0].route.lower() == "po"
    assert meds[0].frequency.lower() == "bid"
    assert meds[1].frequency.lower() == "prn"


def test_multi_word_drug():
    text = "ferrous sulfate 325 mg PO daily"
    meds = extract(text)

    assert len(meds) == 1
    assert meds[0].drug == "ferrous sulfate"


def test_line_with_no_drug_returns_nothing():
    text = "Patient tolerated the procedure well."
    assert extract(text) == []


def test_evidence_equals_line_slice():
    text = "warfarin 5 mg PO daily\nNo medication changes today."
    meds = extract(text)

    assert len(meds) == 1
    med = meds[0]
    assert med.evidence == text[med.start : med.end]


def test_two_drugs_on_same_line_get_own_attributes():
    text = "aspirin 81 mg daily and metoprolol 25 mg BID"
    meds = extract(text)

    assert len(meds) == 2
    assert meds[0].drug == "aspirin"
    assert meds[0].strength == "81 mg"
    assert meds[0].frequency.lower() == "daily"
    assert meds[1].drug == "metoprolol"
    assert meds[1].strength == "25 mg"
    assert meds[1].frequency.lower() == "bid"


def test_crlf_line_endings_keep_evidence_offsets():
    text = "warfarin 5 mg PO daily\r\nNo medication changes today."
    meds = extract(text)

    assert len(meds) == 1
    med = meds[0]
    assert med.evidence == "warfarin 5 mg PO daily"
    assert med.evidence == text[med.start : med.end]


def test_frequency_captures_base_plus_prn():
    text = "Acetaminophen 500mg PO Q6H PRN"
    meds = extract(text)

    assert len(meds) == 1
    med = meds[0]
    assert med.drug == "acetaminophen"
    assert med.strength == "500mg"
    assert med.route.lower() == "po"
    assert med.frequency.lower() == "q6h prn"


def test_inhaler_dosage_and_route():
    text = "Albuterol Sulfate Inhaler 2 puffs Q4H PRN"
    meds = extract(text)

    assert len(meds) == 1
    med = meds[0]
    assert med.drug == "albuterol sulfate"
    assert med.route.lower() == "inhaler"
    assert med.dosage.lower() == "2 puffs"
    assert med.frequency.lower() == "q4h prn"


def test_insulin_glargine_subq_nightly():
    text = "Insulin Glargine 10 units SubQ Nightly"
    meds = extract(text)

    assert len(meds) == 1
    med = meds[0]
    assert med.drug == "insulin glargine"
    assert med.strength == "10 units"
    assert med.route.lower() == "subq"
    assert med.frequency.lower() == "nightly"


def test_benzonatate_tid_prn():
    text = "Benzonatate 100mg PO TID PRN Cough"
    meds = extract(text)

    assert len(meds) == 1
    med = meds[0]
    assert med.drug == "benzonatate"
    assert med.strength == "100mg"
    assert med.route.lower() == "po"
    assert med.frequency.lower() == "tid prn"


def test_longest_match_wins_over_the_shorter_name_it_starts_with():
    """ "insulin glargine" must not be reported as a bare "insulin" plus a stray match."""
    meds = extract("Insulin glargine 10 units SubQ nightly")

    assert [med.drug for med in meds] == ["insulin glargine"]


def test_scan_resumes_after_a_multi_word_match_not_inside_it():
    """Re-entering at "sulfate" would emit a second, overlapping medication."""
    meds = extract("Albuterol sulfate inhaler 2 puffs Q4H PRN")

    assert [med.drug for med in meds] == ["albuterol sulfate"]


def test_drug_name_is_not_matched_inside_a_longer_word():
    assert extract("Discussed the patient's home environment and iron-rich foods") == []


def test_repeated_whitespace_inside_a_multi_word_name_still_matches():
    meds = extract("Ferrous  sulfate 325 mg PO daily")

    assert [med.drug for med in meds] == ["ferrous sulfate"]
    assert meds[0].strength == "325 mg"


def test_drug_name_is_found_when_hyphenated_onto_another_word():
    meds = extract("Patient is on a metformin-containing regimen")

    assert [med.drug for med in meds] == ["metformin"]


def test_evidence_offsets_survive_the_token_matcher():
    text = "Ferrous  sulfate 325 mg PO daily\nwarfarin 5 mg PO daily"
    meds = extract(text)

    assert len(meds) == 2
    for med in meds:
        assert text[med.start : med.end] == med.evidence


def test_unknown_lexicon_mode_is_rejected_with_the_valid_modes_listed():
    with pytest.raises(ValueError, match="unknown lexicon mode"):
        extract("lisinopril 10 mg", mode="ingredience")


def test_each_mode_is_larger_than_the_one_before_it():
    sizes = {mode: extract_rules.lexicon_size(mode) for mode in extract_rules.LEXICON_MODES}

    assert sizes["handwritten"] == 50
    assert sizes["ingredients"] > sizes["handwritten"]
    assert sizes["with_brands"] > sizes["ingredients"]
    assert sizes["ingredients_context"] == sizes["ingredients"]
    assert sizes["with_brands_context"] == sizes["with_brands"]


def test_rxnorm_modes_find_a_drug_the_handwritten_lexicon_never_had():
    text = "Apixaban 5 mg PO BID"

    assert extract(text, mode="handwritten") == []
    assert [med.drug for med in extract(text, mode="ingredients")] == ["apixaban"]


def test_brand_names_are_only_found_in_brand_modes():
    text = "Take Lipitor 20 mg PO nightly"

    assert extract(text, mode="ingredients") == []
    assert [med.drug for med in extract(text, mode="with_brands")] == ["lipitor"]


def test_handwritten_terms_rxnorm_lacks_still_match_in_rxnorm_modes():
    """insulin and senna are not RxNorm IN/PIN names; dropping them would lose recall."""
    assert [med.drug for med in extract("Insulin 10 units subq", mode="ingredients")] == ["insulin"]
    assert [med.drug for med in extract("Senna 2 tablets nightly", mode="ingredients")] == ["senna"]


def test_ambiguous_name_with_no_clinical_context_is_not_a_medication():
    text = "Labs: sodium 138, potassium 4.1, calcium 9.2, glucose 110."

    assert extract(text, mode="ingredients_context") == []
    assert len(extract(text, mode="ingredients")) == 4


def test_ambiguous_name_in_diet_advice_is_not_a_medication():
    text = "Diet: increase water and iron rich foods."

    assert extract(text, mode="ingredients_context") == []


def test_ambiguous_name_in_a_vital_sign_is_not_a_medication():
    text = "Oxygen saturation was 97 percent on room air throughout the stay."

    assert extract(text, mode="ingredients_context") == []


def test_ambiguous_name_is_kept_when_a_dose_follows_it():
    meds = extract("Potassium 20 mEq PO daily.", mode="ingredients_context")

    assert [med.drug for med in meds] == ["potassium"]


def test_ambiguous_name_is_kept_on_a_verb_alone_without_any_dose():
    """Requiring a dose would lose real orders; the verb is enough."""
    meds = extract("You were prescribed iron for anemia.", mode="ingredients_context")

    assert [med.drug for med in meds] == ["iron"]
    assert meds[0].strength is None


def test_inflected_medication_verb_is_recognized():
    """ "Started on" is far more common in discharge text than a bare "start"."""
    meds = extract("Patient was started on oxygen at home.", mode="ingredients_context")

    assert [med.drug for med in meds] == ["oxygen"]


def test_a_neighbouring_drugs_dose_does_not_promote_an_ambiguous_word():
    """The dose belongs to ferrous sulfate, so it must not validate "iron"."""
    text = "Patient is iron deficient; started on ferrous sulfate 325 mg PO daily"

    meds = extract(text, mode="ingredients_context")

    assert [med.drug for med in meds] == ["ferrous sulfate"]


def test_unambiguous_drug_names_are_never_gated():
    meds = extract("Apixaban was discussed.", mode="ingredients_context")

    assert [med.drug for med in meds] == ["apixaban"]


def test_a_gated_out_name_does_not_steal_the_previous_drugs_attributes():
    """The attribute window ends at the next match, so gating must happen first."""
    text = "Insulin glargine 10 units subq nightly; recheck glucose later"

    meds = extract(text, mode="ingredients_context")

    assert [med.drug for med in meds] == ["insulin glargine"]
    assert meds[0].strength == "10 units"
    assert meds[0].frequency.lower() == "nightly"


def test_rxnorm_modes_carry_the_rxcui_and_term_type_not_just_the_name():
    meds = extract("Lisinopril 10 mg PO daily", mode="ingredients")

    assert meds[0].rxcui == "29046"
    assert meds[0].tty == "IN"


def test_handwritten_mode_leaves_provenance_empty_rather_than_inventing_it():
    meds = extract("Lisinopril 10 mg PO daily", mode="handwritten")

    assert meds[0].rxcui is None
    assert meds[0].tty is None


def test_debug_logging_captures_drug_match(caplog):
    text = "lisinopril 10 mg PO daily"
    with caplog.at_level(logging.DEBUG, logger="lablens.extract_rules"):
        extract(text)

    assert any("lisinopril" in record.getMessage() for record in caplog.records)
