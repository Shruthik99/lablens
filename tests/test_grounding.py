from lablens.grounding import GroundingStats, ground_value, is_drug_grounded


def test_is_drug_grounded_true_when_present():
    assert is_drug_grounded("aspirin", "Patient takes Aspirin daily")


def test_is_drug_grounded_false_when_absent():
    assert not is_drug_grounded("metformin", "Patient takes Aspirin daily")


def test_is_drug_grounded_case_and_whitespace_insensitive():
    assert is_drug_grounded("Ibuprofen", "patient takes   IBUPROFEN   daily")


def test_is_drug_grounded_rejects_substring_inside_another_word():
    assert not is_drug_grounded("iron", "Patient lives in a safe environment")


def test_is_drug_grounded_accepts_whole_word_match_despite_similar_longer_word():
    assert is_drug_grounded("iron", "Patient started on iron supplementation")


def test_ground_value_keeps_value_present_in_evidence():
    assert ground_value("10 mg", "Lisinopril 10 mg PO daily") == "10 mg"


def test_ground_value_drops_value_absent_from_evidence():
    stats = GroundingStats()
    assert ground_value("20 mg", "Lisinopril 10 mg PO daily", stats) is None
    assert stats.ungrounded_values == 1


def test_ground_value_rejects_substring_of_a_larger_number():
    stats = GroundingStats()
    assert ground_value("5 mg", "Metoprolol 25 mg BID", stats) is None
    assert stats.ungrounded_values == 1


def test_ground_value_passes_through_none():
    assert ground_value(None, "anything") is None
