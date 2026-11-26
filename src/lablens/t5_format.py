"""Brace-free linearized target format for Flan-T5 medication extraction.

The flan-t5-base tokenizer maps both "{" and "}" to the same <unk> token (see
tests/test_t5_format.py), so a JSON target can never be learned or reproduced losslessly.
This format uses only characters the tokenizer round-trips exactly: letters, digits,
":", "|", ";", and single spaces (no newlines, which SentencePiece may normalize).

Grammar: "drug: <drug> | strength: <s> | frequency: <f> | duration: <d> ; drug: ..."
Null fields are omitted; an empty medication list is the literal string "none".
"""

from __future__ import annotations

NONE_TARGET = "none"
FIELD_SEP = " | "
MED_SEP = " ; "
_DRUG_PREFIX = "drug: "
_OPTIONAL_FIELDS = ("strength", "frequency", "duration")


def linearize_medications(medications: list[dict[str, str | None]]) -> str:
    """Serialize medications into the linearized target string."""
    if not medications:
        return NONE_TARGET
    return MED_SEP.join(_linearize_one(med) for med in medications)


def _linearize_one(med: dict[str, str | None]) -> str:
    parts = [_DRUG_PREFIX + med["drug"]]
    for field in _OPTIONAL_FIELDS:
        value = med.get(field)
        if value is not None:
            parts.append(f"{field}: {value}")
    return FIELD_SEP.join(parts)


def parse_medications(text: str) -> list[dict[str, str | None]] | None:
    """Parse the linearized format back into medication dicts, or None if malformed."""
    text = text.strip()
    if text == NONE_TARGET:
        return []
    if not text:
        return None

    medications = []
    for chunk in text.split(MED_SEP):
        fields = chunk.split(FIELD_SEP)
        if not fields[0].startswith(_DRUG_PREFIX):
            return None
        drug = fields[0][len(_DRUG_PREFIX) :]
        if not drug:
            return None

        med: dict[str, str | None] = {
            "drug": drug,
            "strength": None,
            "frequency": None,
            "duration": None,
        }
        for field_str in fields[1:]:
            if not _set_field(med, field_str):
                return None
        medications.append(med)
    return medications


def _set_field(med: dict[str, str | None], field_str: str) -> bool:
    for field in _OPTIONAL_FIELDS:
        prefix = f"{field}: "
        if field_str.startswith(prefix):
            med[field] = field_str[len(prefix) :]
            return True
    return False
