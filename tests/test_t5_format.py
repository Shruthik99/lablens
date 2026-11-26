import pytest

from lablens.t5_format import linearize_medications, parse_medications


def test_linearize_single_medication_all_fields():
    text = linearize_medications(
        [{"drug": "ibuprofen", "strength": "800 mg", "frequency": "TID", "duration": None}]
    )

    assert text == "drug: ibuprofen | strength: 800 mg | frequency: TID"


def test_linearize_omits_null_fields():
    text = linearize_medications(
        [{"drug": "aspirin", "strength": None, "frequency": None, "duration": None}]
    )

    assert text == "drug: aspirin"


def test_linearize_multiple_medications_joined_with_semicolon():
    text = linearize_medications(
        [
            {"drug": "ibuprofen", "strength": "800 mg", "frequency": "TID", "duration": None},
            {"drug": "aspirin", "strength": "81 mg", "frequency": None, "duration": None},
        ]
    )

    assert (
        text
        == "drug: ibuprofen | strength: 800 mg | frequency: TID ; drug: aspirin | strength: 81 mg"
    )


def test_linearize_empty_list_returns_none_literal():
    assert linearize_medications([]) == "none"


def test_parse_normal_case():
    medications = parse_medications(
        "drug: ibuprofen | strength: 800 mg | frequency: TID ; drug: aspirin | strength: 81 mg"
    )

    assert medications == [
        {"drug": "ibuprofen", "strength": "800 mg", "frequency": "TID", "duration": None},
        {"drug": "aspirin", "strength": "81 mg", "frequency": None, "duration": None},
    ]


def test_parse_omitted_fields_default_to_none():
    medications = parse_medications("drug: aspirin")

    assert medications == [
        {"drug": "aspirin", "strength": None, "frequency": None, "duration": None}
    ]


def test_parse_none_literal_returns_empty_list():
    assert parse_medications("none") == []


def test_parse_malformed_input_returns_none():
    assert parse_medications("this is not the expected format") is None
    assert parse_medications("drug: | strength: 800 mg") is None
    assert parse_medications("drug: aspirin | bogus_field: x") is None
    assert parse_medications("") is None


def test_round_trips_through_the_prepare_and_extract_functions():
    medications = [
        {"drug": "ibuprofen", "strength": "800 mg", "frequency": "TID", "duration": None},
        {"drug": "amoxicillin", "strength": "500 mg", "frequency": "TID", "duration": "x 7 days"},
    ]

    assert parse_medications(linearize_medications(medications)) == medications


def test_every_target_in_a_small_sample_round_trips_through_the_tokenizer_unchanged():
    """The flan-t5-base tokenizer maps "{" and "}" to the same <unk> token, which made the
    old JSON target format unreproducible. This format must survive encode -> decode exactly.

    Skipped when transformers isn't installed (it's an optional "train" extra, not part of
    the dev/api extras CI installs) rather than failing the whole suite.
    """
    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-base")

    samples = [
        linearize_medications(
            [{"drug": "ibuprofen", "strength": "800 mg", "frequency": "TID", "duration": None}]
        ),
        linearize_medications(
            [
                {"drug": "ibuprofen", "strength": "800 mg", "frequency": "TID", "duration": None},
                {"drug": "aspirin", "strength": "81 mg", "frequency": None, "duration": None},
            ]
        ),
        linearize_medications(
            [
                {
                    "drug": "insulin glargine",
                    "strength": None,
                    "frequency": "nightly",
                    "duration": None,
                }
            ]
        ),
        linearize_medications(
            [
                {
                    "drug": "amoxicillin",
                    "strength": "500 mg",
                    "frequency": "TID",
                    "duration": "x 7 days",
                }
            ]
        ),
        linearize_medications([]),
    ]

    for text in samples:
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"round-trip changed: {text!r} -> {decoded!r}"
