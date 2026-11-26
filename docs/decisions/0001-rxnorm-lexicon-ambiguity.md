# 1. RxNorm lexicon: flag ambiguous names, do not delete them

Date: 2026-11-26
Status: Accepted

## Context

The rules extractor recognised drugs from a 50-term hand-written list. That list was the
recall ceiling: any drug not on it was invisible. Phase 3 replaces it with RxNorm concepts
fetched from the NLM RxNav API — 14,718 IN, 3,688 PIN and 5,114 BN concepts
(RxNorm version 02-Nov-2026).

Clinical NLP research on medication dictionaries (MedEx, MedXN) reports that large drug
vocabularies trade recall for precision, because drug names collide with ordinary English.
Measured on ordinary discharge prose, RxNorm IN/PIN/BN fires on `air`, `calcium`, `glucose`,
`iron`, `magnesium`, `oxygen`, `potassium`, `sodium`, `water`, `vancomycin` and the brand
name `today`.

The obvious fix — delete the offending names — is wrong. `iron`, `oxygen`, `potassium` and
`magnesium` are *both* ordinary words and real medications. Deleting them trades a precision
problem for a silent recall problem, and silently losing a real discharge medication is the
worse failure for this project.

## Decision

Keep every name in the lexicon. Flag the ambiguous ones and resolve them with context.

1. **`resources/ambiguity_list.txt`** is a curated, committed, hand-reviewed list of names
   that commonly appear in a discharge document in a non-medication sense: lab panels, vital
   signs, diet advice, materials and devices. Names that are only ever drugs (`niacin`,
   `biotin`, `tylenol`) are deliberately absent — flagging them would cost recall for no
   precision gain. Every entry was verified to exist in RxNorm.

2. **The lexicon TSV is generated, the curation is not.** `python -m lablens.rxnorm` joins
   two committed inputs (`drug_lexicon.txt`, `ambiguity_list.txt`) with the RxNav fetch and
   writes `resources/rxnorm_lexicon.tsv`. Re-running is idempotent and cannot destroy hand
   curation. If the ambiguity flag lived only in the generated file, the next rebuild would
   wipe it.

3. **A flagged name is emitted only with clinical context nearby** — a dose, a route, or a
   medication verb. No dose is required: `You were prescribed iron for anemia` and
   `Continue oxygen by nasal cannula` are both kept on the verb alone.

4. **The hand-written lexicon is preserved, not replaced.** 48 of its 50 terms exist in
   RxNorm IN/PIN; `insulin` and `senna` do not (RxNorm carries `insulin human` /
   `insulin glargine` and `sennosides`). They are kept with `source=handwritten` and an
   empty `rxcui`, and a test asserts they survive.

## Consequences

Precision is recovered without deleting anything, and every decision is auditable: each row
carries its RxCUI, term type, source and RxNorm version, so a prediction can be traced back
to a vocabulary release.

The list is a judgement call and will be wrong at the edges. It is committed and reviewable
precisely so it can be argued with, and it grows as false positives are observed rather than
being guessed up front.

### Known limitation

The context check does not fix lab tests named after drugs:
`Vancomycin level was 15 before the fourth dose.` `vancomycin` is an unambiguous drug name,
so it is never gated, and a dose sits nearby. The annotation guide explicitly excludes these
(`docs/annotation-guide.md`, "Do not label ... lab tests named after drugs"). It is kept in
the negatives fixture as a known failing case rather than patched with an ad-hoc rule or
quietly dropped from the evaluation.
