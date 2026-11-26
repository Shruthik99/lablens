from lablens.evaluate import format_report, score_case, summarize
from lablens.schema import Medication


def make_med(drug: str, **overrides) -> Medication:
    defaults = {
        "strength": None,
        "dosage": None,
        "route": None,
        "frequency": None,
        "duration": None,
        "evidence": drug,
        "start": 0,
        "end": len(drug),
    }
    defaults.update(overrides)
    return Medication(drug=drug, **defaults)


def test_perfect_match():
    expected = [
        {
            "drug": "aspirin",
            "strength": "81 mg",
            "dosage": None,
            "route": "PO",
            "frequency": "daily",
            "duration": None,
        }
    ]
    predicted = [make_med("aspirin", strength="81 mg", route="PO", frequency="daily")]

    result = score_case("text", expected, predicted)
    assert result["true_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["false_positives"] == 0
    assert result["mismatches"] == []

    metrics = summarize([result])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert all(acc == 1.0 for acc in metrics["field_accuracy"].values())


def test_one_missed_drug():
    expected = [
        {
            "drug": "aspirin",
            "strength": "81 mg",
            "dosage": None,
            "route": None,
            "frequency": None,
            "duration": None,
        },
        {
            "drug": "metformin",
            "strength": "500 mg",
            "dosage": None,
            "route": None,
            "frequency": None,
            "duration": None,
        },
    ]
    predicted = [make_med("aspirin", strength="81 mg")]

    result = score_case("text", expected, predicted)
    assert result["true_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["false_positives"] == 0
    assert result["missed_drugs"] == [{"text": "text", "drug": "metformin"}]

    metrics = summarize([result])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5


def test_one_extra_drug():
    expected = [
        {
            "drug": "aspirin",
            "strength": "81 mg",
            "dosage": None,
            "route": None,
            "frequency": None,
            "duration": None,
        },
    ]
    predicted = [
        make_med("aspirin", strength="81 mg"),
        make_med("metformin", strength="500 mg"),
    ]

    result = score_case("text", expected, predicted)
    assert result["true_positives"] == 1
    assert result["false_negatives"] == 0
    assert result["false_positives"] == 1
    assert result["extra_drugs"] == [{"text": "text", "drug": "metformin"}]

    metrics = summarize([result])
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_one_wrong_field():
    expected = [
        {
            "drug": "aspirin",
            "strength": "81 mg",
            "dosage": None,
            "route": "PO",
            "frequency": "daily",
            "duration": None,
        }
    ]
    predicted = [make_med("aspirin", strength="325 mg", route="PO", frequency="daily")]

    result = score_case("text", expected, predicted)
    assert result["true_positives"] == 1
    assert len(result["mismatches"]) == 1
    mismatch = result["mismatches"][0]
    assert mismatch["field"] == "strength"
    assert mismatch["expected"] == "81 mg"
    assert mismatch["predicted"] == "325 mg"

    metrics = summarize([result])
    assert metrics["field_accuracy"]["strength"] == 0.0
    assert metrics["field_accuracy"]["route"] == 1.0
    assert metrics["field_accuracy"]["frequency"] == 1.0


def test_drug_name_matching_is_case_insensitive():
    expected = [
        {
            "drug": "Aspirin",
            "strength": None,
            "dosage": None,
            "route": None,
            "frequency": None,
            "duration": None,
        }
    ]
    predicted = [make_med("aspirin")]

    result = score_case("text", expected, predicted)
    assert result["true_positives"] == 1
    assert result["false_negatives"] == 0


def test_field_comparison_ignores_case_and_whitespace():
    expected = [
        {
            "drug": "aspirin",
            "strength": None,
            "dosage": None,
            "route": None,
            "frequency": "PO  Daily",
            "duration": None,
        }
    ]
    predicted = [make_med("aspirin", frequency="po daily")]

    result = score_case("text", expected, predicted)
    assert result["mismatches"] == []


def test_none_equals_none_for_missing_fields():
    expected = [
        {
            "drug": "aspirin",
            "strength": "81 mg",
            "dosage": None,
            "route": None,
            "frequency": None,
            "duration": None,
        }
    ]
    predicted = [make_med("aspirin", strength="81 mg")]

    result = score_case("text", expected, predicted)
    assert result["mismatches"] == []
    assert result["field_total"]["dosage"] == 1
    assert result["field_correct"]["dosage"] == 1


def test_format_report_smoke():
    metrics = summarize(
        [
            score_case(
                "text",
                [
                    {
                        "drug": "aspirin",
                        "strength": "81 mg",
                        "dosage": None,
                        "route": None,
                        "frequency": None,
                        "duration": None,
                    }
                ],
                [make_med("aspirin", strength="325 mg")],
            )
        ]
    )
    report = format_report(metrics)
    assert "precision=" in report
    assert "strength" in report
