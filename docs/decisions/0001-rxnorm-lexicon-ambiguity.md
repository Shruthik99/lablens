# 1. RxNorm lexicon: flag ambiguous names, don't delete them (2026-11-26, accepted)
**Context:** the 50-term hand-written lexicon capped recall. RxNorm IN/PIN/BN (23,520 concepts, v02-Nov-2026) adds coverage, but many names are also ordinary English: iron, oxygen, sodium, calcium, glucose, water, air, vancomycin.
**Decision:** keep every name, including the hand-written 50 (`insulin`/`senna` aren't RxNorm terms but stay). Flag ambiguous names in a curated `ambiguity_list.txt`; the `*_context` modes emit a flagged name only with a nearby dose, route, or medication verb — no dose required.
**Evidence** (38 cases: mini_eval + negatives + coverage):

| mode | terms | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|---|
| handwritten | 50 | 12 | 0 | 16 | 1.000 | 0.429 | 0.600 |
| ingredients | 18369 | 23 | 10 | 5 | 0.697 | 0.821 | 0.754 |
| ingredients_context | 18369 | 23 | 1 | 5 | 0.958 | 0.821 | 0.885 |
| with_brands | 23483 | 28 | 10 | 0 | 0.737 | 1.000 | 0.848 |
| with_brands_context | 23483 | 28 | 1 | 0 | 0.966 | 1.000 | 0.982 |

**Known limits:** the gate can't catch a lab test named after a drug (`Vancomycin level was 15 before the fourth dose` stays a false positive in every mode, kept in the negatives fixture on purpose); RxNorm is US-only, so non-US brand names and dosing conventions aren't covered.
