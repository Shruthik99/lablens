"""Audit tool: inspect rules-only medications from the weak-label merge, with evidence.

Re-runs the same data prep as lablens.prepare_t5_data on the same sample (--n/--seed),
keeping merge_medications' per-record "sources" tag, then prints a fixed-seed random
sample of medications whose source is rules-only — each with its merged fields and the
evidence line the rule extractor matched it on. Read-only: doesn't touch data/gold/ or
tests/fixtures/, and writes nothing.

Runnable as: python -m lablens.audit_weak_labels [--n 500] [--seed 42] [--sample 25]
"""

from __future__ import annotations

import argparse
import random
import sys

from lablens import extract_rules
from lablens.grounding import normalize
from lablens.prepare_t5_data import (
    _FLAN_T5_TOKENIZER,
    _MAX_INPUT_TOKENS,
    _MERGE_LEXICON_MODE,
    _fix_literal_newlines,
    _truncate_to_tokens,
    build_target_medications,
    ground_target_medications,
    merge_medications,
)
from lablens.splits import HELD_OUT_NOTE_INDEXES


def _collect_rules_only(n: int, seed: int) -> list[dict[str, object]]:
    """Mirror prepare_t5_data._iter_examples, but keep each rules-only merged medication
    together with the evidence line from the rule extractor that produced it."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_FLAN_T5_TOKENIZER)
    dataset = load_dataset("AGBonnet/augmented-clinical-notes", split="train")
    dataset = dataset.shuffle(seed=seed)

    rules_only: list[dict[str, object]] = []
    n_examples = 0

    for row in dataset:
        if row["idx"] in HELD_OUT_NOTE_INDEXES:
            continue

        gpt_meds = build_target_medications(row["summary"])
        note_text = _fix_literal_newlines(row["note"])
        input_text = _truncate_to_tokens(
            tokenizer, "extract medications: " + note_text, _MAX_INPUT_TOKENS
        )
        gpt_meds = ground_target_medications(gpt_meds, input_text)

        rules_text = _truncate_to_tokens(tokenizer, note_text, _MAX_INPUT_TOKENS)
        rule_meds = extract_rules.extract(rules_text, mode=_MERGE_LEXICON_MODE)

        # First-mention-per-drug evidence, matching merge_medications' own dedup rule.
        evidence_by_name: dict[str, str] = {}
        for med in rule_meds:
            evidence_by_name.setdefault(normalize(med.drug), med.evidence)

        medications = merge_medications(gpt_meds, rule_meds)
        if not medications:
            continue

        for med in medications:
            if med["sources"] != ("rules",):
                continue
            rules_only.append(
                {
                    "note_idx": row["idx"],
                    "drug": med["drug"],
                    "strength": med["strength"],
                    "dosage": med["dosage"],
                    "route": med["route"],
                    "frequency": med["frequency"],
                    "duration": med["duration"],
                    "evidence": evidence_by_name.get(normalize(med["drug"]), ""),
                }
            )

        n_examples += 1
        if n_examples >= n:
            break

    return rules_only


def main(argv: list[str] | None = None) -> None:
    # Clinical note text routinely carries characters (µg, °, ×) outside the codepage
    # Windows otherwise uses for stdout when it isn't a console (e.g. piped to a file);
    # without this, printing one such evidence line crashes the run partway through.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Audit rules-only weak labels with their evidence line."
    )
    parser.add_argument("--n", type=int, default=500, help="Number of examples (default: 500).")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed (default: 42).")
    parser.add_argument(
        "--sample",
        type=int,
        default=25,
        help="Number of rules-only medications to print (default: 25).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    rules_only = _collect_rules_only(args.n, args.seed)
    print(f"Rules-only medications found: {len(rules_only)}")

    sample_size = min(args.sample, len(rules_only))
    sample = random.Random(args.seed).sample(rules_only, sample_size)

    for i, med in enumerate(sample, start=1):
        print(f"\n[{i}] note #{med['note_idx']} — {med['drug']}")
        print(
            f"    strength={med['strength']!r} dosage={med['dosage']!r} "
            f"route={med['route']!r} frequency={med['frequency']!r} "
            f"duration={med['duration']!r}"
        )
        print(f"    evidence: {med['evidence']!r}")


if __name__ == "__main__":
    main()
