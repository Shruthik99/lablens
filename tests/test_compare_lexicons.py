import json

from lablens.compare_lexicons import brand_only_drugs, format_table, load_cases, main, run_mode


def _metrics(**overrides):
    base = {
        "lexicon_size": 50,
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "field_accuracy": {
            "strength": 1.0,
            "dosage": None,
            "route": 1.0,
            "frequency": None,
            "duration": None,
        },
        "found_drugs": set(),
        "extra_drugs": [],
    }
    base.update(overrides)
    return base


def test_load_cases_concatenates_every_fixture_so_modes_see_identical_data(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(json.dumps([{"text": "aspirin 81 mg", "expected": []}]), encoding="utf-8")
    second.write_text(json.dumps([{"text": "no drugs here", "expected": []}]), encoding="utf-8")

    cases = load_cases([first, second])

    assert [case["text"] for case in cases] == ["aspirin 81 mg", "no drugs here"]


def test_table_reports_precision_as_not_available_when_nothing_was_predicted():
    """A clean run over negatives predicts nothing; 0.000 would read as total failure."""
    results = {
        "ingredients_context": _metrics(
            true_positives=0, false_positives=0, precision=0.0, recall=0.0, f1=0.0
        )
    }

    table = format_table(results)

    assert "n/a" in table
    assert "0.000      0.000" not in table


def test_table_reports_a_real_precision_when_something_was_predicted():
    table = format_table({"ingredients": _metrics(true_positives=3, false_positives=1)})

    assert "1.000" in table


def test_brand_coverage_counts_only_drugs_the_ingredient_modes_missed():
    results = {
        "ingredients": _metrics(found_drugs={"lisinopril"}),
        "with_brands": _metrics(found_drugs={"lisinopril", "lipitor"}),
    }

    assert brand_only_drugs(results) == {"lipitor"}


def test_run_mode_scores_a_mode_and_records_its_lexicon_size():
    cases = [{"text": "Apixaban 5 mg PO BID", "expected": [{"drug": "apixaban"}]}]

    metrics = run_mode("ingredients", cases)

    assert metrics["true_positives"] == 1
    assert metrics["lexicon_size"] > 50
    assert metrics["found_drugs"] == {"apixaban"}


def test_handwritten_mode_misses_a_drug_the_rxnorm_modes_find():
    cases = [{"text": "Apixaban 5 mg PO BID", "expected": [{"drug": "apixaban"}]}]

    assert run_mode("handwritten", cases)["false_negatives"] == 1
    assert run_mode("ingredients", cases)["false_negatives"] == 0


def test_main_evaluates_every_mode_on_the_same_cases(tmp_path, capsys):
    fixture = tmp_path / "cases.json"
    fixture.write_text(
        json.dumps([{"text": "Lisinopril 10 mg PO daily", "expected": [{"drug": "lisinopril"}]}]),
        encoding="utf-8",
    )

    results = main([str(fixture)])

    assert set(results) == {
        "handwritten",
        "ingredients",
        "ingredients_context",
        "with_brands",
        "with_brands_context",
    }
    assert "Incremental brand coverage" in capsys.readouterr().out
