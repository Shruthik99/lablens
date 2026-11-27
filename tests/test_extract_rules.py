import logging

import pytest

from lablens import extract_rules
from lablens.extract_rules import extract, lookup


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
    """ "iron" must not be found inside "environment"."""
    assert extract("Discussed the patient's home environment at length") == []


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


def test_lookup_returns_entry_for_a_known_ingredient():
    entry = lookup("lisinopril", mode="ingredients")

    assert entry is not None
    assert entry.rxcui == "29046"
    assert entry.tty == "IN"
    assert entry.ambiguous is False


def test_lookup_returns_none_for_an_unknown_name():
    assert lookup("not-a-real-drug-xyz") is None


def test_lookup_flags_an_ambiguous_entry():
    entry = lookup("iron", mode="ingredients")

    assert entry is not None
    assert entry.ambiguous is True


def test_attribute_window_does_not_cross_into_the_next_sentence():
    """A number in a later sentence on the same line must not be read as this drug's
    strength; only the next drug match or the end of this sentence bounds the window."""
    text = (
        "Started on amlodipine for blood pressure control. "
        "Pain was rated 5 mg on the visual scale today."
    )

    meds = extract(text)

    assert [med.drug for med in meds] == ["amlodipine"]
    assert meds[0].strength is None


def test_sentence_boundary_detection_does_not_split_a_decimal_dose():
    text = "amlodipine 0.5 mg PO daily for hypertension. Patient tolerated well."

    meds = extract(text)

    assert len(meds) == 1
    assert meds[0].strength == "0.5 mg"


def test_strength_keeps_a_per_weight_unit():
    meds = extract("vancomycin 15 mg/kg IV q12h")

    assert meds[0].strength == "15 mg/kg"


def test_strength_keeps_a_per_weight_per_day_unit():
    meds = extract("gentamicin 1 mg/kg/day IV daily")

    assert meds[0].strength == "1 mg/kg/day"


def test_strength_keeps_a_concentration_form_not_shortened():
    meds = extract("amoxicillin 125 mg/5 ml PO BID")

    assert meds[0].strength == "125 mg/5 ml"


def test_drug_name_followed_by_receptor_is_not_a_medication():
    text = "The biopsy showed reduced insulin receptor expression in the tissue sample."

    assert extract(text) == []


def test_drug_name_directly_marked_resistant_is_not_a_medication():
    text = "The infection was vancomycin-resistant and required a different antibiotic."

    assert extract(text) == []


def test_drug_name_preceded_by_serum_and_followed_by_level_is_not_a_medication():
    text = "Serum digoxin level was therapeutic, no dose change needed."

    assert extract(text) == []


def test_drug_name_preceded_by_serum_alone_is_not_a_medication():
    text = "Serum lithium was checked given mood symptoms."

    assert extract(text) == []


def test_lab_context_filter_does_not_drop_a_real_order():
    text = "insulin 10 units subcutaneous before meals"

    assert [med.drug for med in extract(text)] == ["insulin"]


def test_perform_without_context_is_not_a_medication():
    text = "Patient is expected to perform well after surgery."

    assert extract(text) == []


def test_thrive_without_context_is_not_a_medication():
    text = "The baby is able to thrive despite the diagnosis."

    assert extract(text) == []


def test_propylene_without_context_is_not_a_medication():
    text = "Sample tested positive for propylene in the industrial exposure panel."

    assert extract(text) == []


def test_calcitonin_requires_context():
    assert extract("Calcitonin level was checked.") == []

    meds = extract("Calcitonin 200 units nasal daily.")
    assert [med.drug for med in meds] == ["calcitonin"]


def test_progesterone_requires_context():
    assert extract("Progesterone level was drawn as part of fertility workup.") == []

    meds = extract("Progesterone 200 mg vaginal nightly was started.")
    assert [med.drug for med in meds] == ["progesterone"]


def test_human_chorionic_gonadotropin_requires_context():
    assert extract("hCG was negative.") == []

    meds = extract("Human chorionic gonadotropin 10000 units IM was given.")
    assert [med.drug for med in meds] == ["human chorionic gonadotropin"]


def test_organism_species_names_are_never_matched_as_drugs():
    text = "Blood cultures grew Escherichia coli and Enterococcus faecalis."

    assert extract(text) == []
    assert extract(text, mode="ingredients") == []


def test_colistin_still_matches_after_organism_exclusion():
    """colistin is a real antibiotic, historically named after Bacillus colistinus; it
    must not be caught by the organism exclusion list, which matches full species names."""
    meds = extract("colistin 2 million units IV q8h")

    assert [med.drug for med in meds] == ["colistin"]


def test_methyl_prednisolone_with_space_matches_as_one_drug():
    text = "methyl prednisolone 125 mg IV once daily"

    meds = extract(text)

    assert len(meds) == 1
    assert meds[0].drug == "methylprednisolone"
    assert meds[0].strength == "125 mg"
    assert meds[0].evidence == text
    assert text[meds[0].start : meds[0].end] == text


def test_methyl_hyphen_prednisolone_matches_as_one_drug():
    text = "methyl-prednisolone 40 mg IV daily"

    meds = extract(text)

    assert len(meds) == 1
    assert meds[0].drug == "methylprednisolone"
    assert meds[0].strength == "40 mg"


def test_has_context_true_when_dose_and_route_are_present():
    meds = extract("lisinopril 10 mg PO daily")

    assert meds[0].has_context is True


def test_has_context_false_for_a_bare_mention_but_still_extracted():
    """extract() must not lose recall for the live/API extractor: has_context is metadata
    for the weak-label merge, not a filter on extract() itself."""
    meds = extract("Patient has a history of lisinopril use.")

    assert len(meds) == 1
    assert meds[0].drug == "lisinopril"
    assert meds[0].has_context is False


def test_dose_before_drug_belongs_to_the_drug_it_precedes():
    """A dose directly followed by a drug name belongs to that drug, never to the
    previous one -- the common anesthesia-induction enumeration pattern."""
    meds = extract("100 mg propofol, and 50 mg rocuronium")

    assert [med.drug for med in meds] == ["propofol", "rocuronium"]
    assert meds[0].strength == "100 mg"
    assert meds[1].strength == "50 mg"


def test_dose_before_drug_does_not_steal_the_next_drugs_dose():
    meds = extract("200 mg propofol, 70 mg lidocaine")

    assert [med.drug for med in meds] == ["propofol", "lidocaine"]
    assert meds[0].strength == "200 mg"
    assert meds[1].strength == "70 mg"


def test_comma_grouped_strength_is_not_split():
    meds = extract("heparin drip (25,000 units in 250 mL D5W)")

    assert meds[0].strength == "25,000 units"


def test_comma_grouped_strength_with_mg_is_not_split():
    meds = extract("ferrous sulfate 1,000 mg PO daily")

    assert meds[0].strength == "1,000 mg"


def test_diluent_volume_after_in_is_not_a_dosage():
    meds = extract("heparin drip (25,000 units in 250 mL D5W)")

    assert meds[0].strength == "25,000 units"
    assert meds[0].dosage is None


def test_frequency_thrice_daily_never_becomes_daily():
    meds = extract("metronidazole (400 mg) thrice daily for 7 days")

    assert meds[0].frequency.lower() == "thrice daily"


def test_frequency_three_times_daily():
    meds = extract("amoxicillin 500 mg three times daily")

    assert meds[0].frequency.lower() == "three times daily"


def test_frequency_once_a_day():
    meds = extract("aspirin 100 mg once a day")

    assert meds[0].frequency.lower() == "once a day"


def test_frequency_n_times_per_day():
    """ "per day" and "a day" are interchangeable; both numeral and word counts work."""
    assert extract("tetrabenazine 12.5 mg two times per day")[0].frequency.lower() == (
        "two times per day"
    )
    assert extract("tetrabenazine 12.5 mg three times per day")[0].frequency.lower() == (
        "three times per day"
    )


def test_frequency_n_times_a_day():
    meds = extract("ibuprofen 400 mg 4 times a day")

    assert meds[0].frequency.lower() == "4 times a day"


def test_frequency_every_n_h():
    meds = extract("metoprolol 25 mg every 8 h")

    assert meds[0].frequency.lower() == "every 8 h"


def test_frequency_q_n_h_still_works():
    meds = extract("metoprolol 25 mg q6h")

    assert meds[0].frequency.lower() == "q6h"


def test_frequency_slash_day_suffix_on_a_strength():
    meds = extract("azathioprine 75 mg/day")

    assert meds[0].strength == "75 mg"
    assert meds[0].frequency == "/day"


def test_frequency_slash_d_suffix_on_a_strength():
    meds = extract("methylprednisolone 80 mg/d")

    assert meds[0].strength == "80 mg"
    assert meds[0].frequency == "/d"


def test_frequency_slash_day_does_not_fire_inside_a_per_weight_per_day_strength():
    """ "5 mg/kg/day" is one strength value; the trailing "/day" there is not a separate
    frequency match (unlike a bare "mg/day" with no "/kg")."""
    meds = extract("gentamicin 1 mg/kg/day IV daily")

    assert meds[0].strength == "1 mg/kg/day"
    assert meds[0].frequency.lower() == "daily"


def test_strength_rate_unit_per_hour():
    meds = extract("alteplase 0.5 mg/h infusion overnight")

    assert meds[0].strength == "0.5 mg/h"


def test_decimal_strength_and_per_weight_strength_still_work():
    """Regression guard for the two forms the previous audit fixed."""
    assert extract("amlodipine 0.5 mg PO daily")[0].strength == "0.5 mg"
    assert extract("vancomycin 5 mg/kg IV q12h")[0].strength == "5 mg/kg"


def test_tissue_plasminogen_activator_is_alteplase():
    meds = extract("He was subsequently started on tissue plasminogen activator therapy.")

    assert [med.drug for med in meds] == ["alteplase"]


def test_tpa_abbreviation_is_alteplase():
    meds = extract("He was started on tPA for the stroke.")

    assert [med.drug for med in meds] == ["alteplase"]


def test_bare_plasminogen_without_context_is_not_a_medication():
    text = "A diagnosis was made based on plasminogen activator inhibitor levels."

    assert extract(text) == []


def test_gadolinium_is_excluded_as_a_diagnostic_agent():
    assert extract("He received gadolinium injection for the MRI.") == []


def test_fluorescein_is_excluded_as_a_diagnostic_agent():
    assert extract("Fluorescein angiography revealed hypofluorescence.") == []


def test_iodinated_contrast_agents_are_excluded():
    assert extract("Iohexol contrast was administered for the CT scan.") == []
    assert extract("He received iodixanol contrast for the angiogram.") == []


def test_route_before_drug_belongs_to_the_drug_it_precedes():
    """A route word directly before a drug name belongs to that drug, never to the
    previous one -- the same rule as dose-before-drug, applied to routes."""
    meds = extract("topical atropine, and oral prednisolone")

    assert [med.drug for med in meds] == ["atropine", "prednisolone"]
    assert meds[0].route == "topical"
    assert meds[1].route == "oral"


def test_route_after_an_unrelated_later_word_is_not_claimed():
    """ "inhaled" describes "corticosteroids" (not itself a lexicon drug), not the earlier
    "albuterol" -- a forward route match must be adjacent to the drug (or its own
    strength/dosage), not just anywhere in the rest of the sentence."""
    meds = extract("albuterol and occasional inhaled corticosteroids")

    assert [med.drug for med in meds] == ["albuterol"]
    assert meds[0].route is None


def test_route_before_drug_with_no_other_attributes():
    meds = extract("intramuscular lorazepam 1 mg")

    assert meds[0].drug == "lorazepam"
    assert meds[0].route == "intramuscular"


def test_route_before_drug_with_strength_and_frequency_following():
    meds = extract("oral prednisone 25 mg daily")

    assert meds[0].drug == "prednisone"
    assert meds[0].route == "oral"
    assert meds[0].strength == "25 mg"
    assert meds[0].frequency.lower() == "daily"


def test_lab_value_phrase_is_not_a_medication():
    assert extract("an elevated glucose value around 340 mg/dL") == []


def test_lab_value_of_above_number_is_not_a_medication():
    assert extract("lactate of above 15 mmol/L") == []


def test_blood_urea_nitrogen_is_not_a_medication():
    """ "blood urea nitrogen" (BUN) tokenizes across two ambiguous lexicon entries
    ("urea", "nitrogen"); neither is a medication here."""
    assert extract("a blood urea nitrogen of 13 mg/dl") == []


def test_lab_level_of_drug_name_in_parens_is_not_a_medication():
    assert extract("serum levels of immunoglobulin G (4435 mg/dL)") == []


def test_bare_number_with_lab_unit_is_not_a_medication():
    assert extract("sodium 138 mmol/L was normal") == []
    assert extract("glucose 340 mg/dL on admission") == []


def test_dose_of_drug_belongs_to_the_drug_it_precedes():
    """ "NUMBER unit OF drugname" is the same dose-before-drug rule, with "of" as the
    connector: the dose still belongs to the drug it's immediately attached to, never to
    one named earlier."""
    meds = extract("4 g of acetaminophen and 50 mg of tramadol")

    assert [med.drug for med in meds] == ["acetaminophen", "tramadol"]
    assert meds[0].strength == "4 g"
    assert meds[1].strength == "50 mg"


def test_strength_range_is_not_split_at_the_hyphen():
    meds = extract("fluoxetine 40-60 mg/day")

    assert meds[0].strength == "40-60 mg"
    assert meds[0].frequency == "/day"


def test_indocyanine_green_is_excluded_as_a_diagnostic_agent():
    assert extract("Indocyanine green angiography showed good perfusion.") == []


def test_methylene_blue_is_excluded_as_a_diagnostic_agent():
    assert extract("Methylene blue was given for the perfusion study.") == []


def test_isosulfan_blue_is_excluded_as_a_diagnostic_agent():
    assert extract("The sentinel node was identified using isosulfan blue dye.") == []


def test_illicit_substances_require_context_like_any_ambiguous_name():
    assert extract("Urine toxicology was positive for cocaine use.") == []
    assert extract("He has a history of heroin use.") == []
    assert extract("Patient reported recent methamphetamine use.") == []

    meds = extract("He was given cocaine 2 mg topically for the procedure.")
    assert [med.drug for med in meds] == ["cocaine"]
