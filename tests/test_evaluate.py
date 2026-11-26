from lablens.evaluate import build_run_data, format_report, score_case, summarize
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


def test_field_accuracy_is_none_when_no_drugs_matched():
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
    result = score_case("text", expected, predicted=[])

    metrics = summarize([result])

    assert metrics["false_negatives"] == 1
    assert all(acc is None for acc in metrics["field_accuracy"].values())


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


def test_format_report_shows_na_for_field_accuracy_when_no_matches():
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
                [],
            )
        ]
    )

    report = format_report(metrics)

    assert "strength: n/a" in report


def test_build_run_data_omits_field_accuracy_when_no_matches():
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
                [],
            )
        ]
    )

    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        metrics,
        lexicon_size=1,
        fixture_sha256="deadbeef",
        git_commit=None,
    )

    assert not any(key.startswith("field_accuracy_") for key in run_data["metrics"])


def test_build_run_data_includes_params_and_metrics():
    metrics = summarize(
        [
            score_case(
                "text",
                [
                    {
                        "drug": "aspirin",
                        "strength": "81 mg",
                        "dosage": None,
                        "route": "PO",
                        "frequency": "daily",
                        "duration": None,
                    }
                ],
                [make_med("aspirin", strength="81 mg", route="PO", frequency="daily")],
            )
        ]
    )

    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        metrics,
        lexicon_size=42,
        fixture_sha256="deadbeef",
        git_commit="abc1234",
    )

    assert run_data["params"] == {
        "extractor": "rules",
        "lexicon_size": 42,
        "fixture_path": "tests/fixtures/mini_eval.json",
        "fixture_sha256": "deadbeef",
        "git_commit": "abc1234",
    }
    assert run_data["metrics"] == {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "field_accuracy_strength": metrics["field_accuracy"]["strength"],
        "field_accuracy_dosage": metrics["field_accuracy"]["dosage"],
        "field_accuracy_route": metrics["field_accuracy"]["route"],
        "field_accuracy_frequency": metrics["field_accuracy"]["frequency"],
        "field_accuracy_duration": metrics["field_accuracy"]["duration"],
    }


def test_build_run_data_includes_grounding_for_llm_extractor():
    metrics = summarize([])
    metrics["ungrounded_values"] = 3
    metrics["ungrounded_drugs"] = 1

    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        metrics,
        fixture_sha256="deadbeef",
        git_commit=None,
        extractor="llm",
        model="qwen3:4b",
        prompt_version="v2",
    )

    assert run_data["params"]["grounding"] == "on"
    assert run_data["metrics"]["ungrounded_values"] == 3
    assert run_data["metrics"]["ungrounded_drugs"] == 1


def test_build_run_data_records_the_lexicon_mode_and_its_provenance():
    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        summarize([]),
        lexicon_size=18369,
        lexicon_mode="ingredients_context",
        lexicon_sha256="cafe1234",
        rxnorm_version="02-Nov-2026",
        fixture_sha256="deadbeef",
        git_commit=None,
    )

    assert run_data["params"]["lexicon_mode"] == "ingredients_context"
    assert run_data["params"]["lexicon_size"] == 18369
    assert run_data["params"]["lexicon_sha256"] == "cafe1234"
    assert run_data["params"]["rxnorm_version"] == "02-Nov-2026"


def test_build_run_data_omits_lexicon_params_for_non_rules_extractors():
    """A lexicon mode on an LLM run would be a meaningless param."""
    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        summarize([]),
        fixture_sha256="deadbeef",
        git_commit=None,
        extractor="llm",
        model="qwen3:4b",
    )

    assert "lexicon_mode" not in run_data["params"]
    assert "lexicon_sha256" not in run_data["params"]
    assert "rxnorm_version" not in run_data["params"]


def test_build_run_data_omits_git_commit_when_unavailable():
    metrics = summarize([])

    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        metrics,
        lexicon_size=1,
        fixture_sha256="deadbeef",
        git_commit=None,
    )

    assert "git_commit" not in run_data["params"]


def test_build_run_data_includes_model_dir_and_run_info_for_t5_extractor():
    metrics = summarize([])
    metrics["ungrounded_drugs"] = 2
    metrics["parse_failures"] = 1

    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        metrics,
        fixture_sha256="deadbeef",
        git_commit=None,
        extractor="t5",
        model_dir="models/flan-t5",
        t5_run_info={
            "git_commit": "abc123",
            "data_sha256": {"train.jsonl": "trainhash", "val.jsonl": "valhash"},
        },
    )

    assert run_data["params"]["model_dir"] == "models/flan-t5"
    assert run_data["params"]["model_git_commit"] == "abc123"
    assert run_data["params"]["model_train_sha256"] == "trainhash"
    assert run_data["params"]["model_val_sha256"] == "valhash"
    assert run_data["metrics"]["parse_failures"] == 1
    assert run_data["metrics"]["ungrounded_drugs"] == 2


def test_build_run_data_omits_model_run_info_fields_when_unavailable():
    metrics = summarize([])

    run_data = build_run_data(
        "tests/fixtures/mini_eval.json",
        metrics,
        fixture_sha256="deadbeef",
        git_commit=None,
        extractor="t5",
        model_dir="models/flan-t5",
        t5_run_info=None,
    )

    assert run_data["params"]["model_dir"] == "models/flan-t5"
    assert "model_git_commit" not in run_data["params"]
    assert "model_train_sha256" not in run_data["params"]
    assert "model_val_sha256" not in run_data["params"]
