# 1. RxNorm lexicon: flag ambiguous names, don't delete them (2026-11-26, accepted)
**Context:** the 50-term hand-written lexicon capped recall. RxNorm IN/PIN/BN (v02-Nov-2026) adds coverage, but many names are also ordinary English or lab terms: iron, oxygen, sodium, calcium, glucose, water, air.
**Decision:** keep every name, including the hand-written 50. Flag ambiguous names in `ambiguity_list.txt`; the `*_context` modes keep a flagged name only with a nearby dose, route, or medication verb (no dose required). Lab contexts ("level", "receptor", "-resistant", "serum") and organism names are excluded. Default mode: `with_brands_context`.
**Evidence** (38 cases: mini_eval + negatives + coverage, after the weak-label audit fixes):

| mode | terms | tp | fp | fn | precision | recall | f1 |
|---|---|---|---|---|---|---|---|
| handwritten | 50 | 12 | 0 | 16 | 1.000 | 0.429 | 0.600 |
| ingredients | 18344 | 23 | 9 | 5 | 0.719 | 0.821 | 0.767 |
| ingredients_context | 18344 | 23 | 0 | 5 | 1.000 | 0.821 | 0.902 |
| with_brands | 23458 | 28 | 9 | 0 | 0.757 | 1.000 | 0.862 |
| with_brands_context | 23458 | 28 | 0 | 0 | 1.000 | 1.000 | 1.000 |

Measured on development fixtures that shaped the rules; not a measure of real-world accuracy. Real-text evaluation: ADE Corpus (Phase 3).

**Known limits:** RxNorm is US-only, so non-US brand names (e.g. Indian brands) aren't covered; context and lab-context rules are pattern-based and will miss phrasings they haven't seen.