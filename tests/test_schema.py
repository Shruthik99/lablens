import pytest
from pydantic import ValidationError

from lablens.schema import Action, Medication, Setting


def test_valid_medication():
    med = Medication(
        drug="metformin",
        strength="500mg",
        dosage="1 tablet",
        route="oral",
        frequency="twice daily",
        duration="30 days",
        action=Action.START,
        setting=Setting.AT_DISCHARGE,
        evidence="Start metformin 500mg oral twice daily",
        start=0,
        end=38,
    )
    assert med.drug == "metformin"
    assert med.action == Action.START
    assert med.setting == Setting.AT_DISCHARGE


def test_missing_optional_fields_stay_none():
    med = Medication(
        drug="aspirin",
        evidence="aspirin",
        start=0,
        end=7,
    )
    assert med.strength is None
    assert med.dosage is None
    assert med.route is None
    assert med.frequency is None
    assert med.duration is None
    assert med.action == Action.UNKNOWN
    assert med.setting == Setting.UNKNOWN


def test_end_not_greater_than_start_rejected():
    with pytest.raises(ValidationError):
        Medication(drug="aspirin", evidence="aspirin", start=5, end=5)

    with pytest.raises(ValidationError):
        Medication(drug="aspirin", evidence="aspirin", start=5, end=3)


def test_empty_evidence_rejected():
    with pytest.raises(ValidationError):
        Medication(drug="aspirin", evidence="", start=0, end=7)
