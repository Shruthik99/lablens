# Annotation guide (v0.2)

## Purpose
Gold labels for medication extraction. Each medication mention gets attributes and context.
Based on n2c2 2018 (attributes) and n2c2 2022 CMED (medication change context).

## Unit of annotation
- One annotation per medication mention (a drug name in the text).
- Labels go on the drug name, not the trigger word ("Start [Plavix]" → label Plavix).
- The same drug mentioned twice = two mentions.

## What counts as a medication
Label:
- Specific drugs, generic or brand (metformin, Lipitor)
- Supplements given as treatment (ferrous sulfate, vitamin D)
- IV fluids and blood products if named (IV crystalloid, normal saline)

Do not label:
- Drug classes without a specific drug ("antibiotics", "chemotherapy")
- Procedures and therapies (physiotherapy, root canal, "conservative management")
- Allergies ("allergic to penicillin")
- Lab tests named after drugs ("vancomycin level")

## Attribute spans (link each to its drug)
| Field | Meaning | Example |
|---|---|---|
| strength | amount of drug per unit | 800 mg, 2.5 mg/3 ml |
| dosage | amount the patient takes | 2 tablets, 5 mL, 10 units |
| form | how it is made | tablet, capsule, cream, inhaler |
| route | how it is taken | PO, by mouth, IV, topical, inhalation |
| frequency | how often | twice daily, BID, q4h, PRN |
| duration | for how long | for 5 days, x7d |
| reason | why, only if the text says it | for pain |

Span rules: shortest span that holds the full value ("800 mg", not "800 mg tablets").
Leave a field empty if the text doesn't say it. Never infer.

## Context labels (per mention)
| Field | Values |
|---|---|
| action | start, stop, increase, decrease, other change, one-time dose, no change, unknown |
| temporality | past, present, future |
| certainty | certain, conditional, hypothetical |
| negated | yes, no |
| setting | before admission, in hospital, at discharge, unknown |

Examples:
- "Discharge meds: amoxicillin 500 mg PO BID x 5 days" → start, present, certain, no, at discharge
- "Continue lisinopril 10 mg daily" → no change, present, certain, no, at discharge
- "Metformin was stopped on admission" → stop, past, certain, no, in hospital
- "Given ondansetron 4 mg IV x1 in the ED" → one-time dose, past, certain, no, in hospital
- "If BP < 100, hold amlodipine" → stop, future, conditional, no, at discharge
- "Consider starting a statin at follow-up" → start, future, hypothetical, no, unknown
- "Do not restart warfarin" → start, future, certain, yes, at discharge
- "Increase furosemide to 40 mg daily" → increase, present, certain, no, at discharge

## Tricky cases
- Several events in one mention: "Keflex 500 mg qid x 7 days" = start now + stop in future → two events on one mention.
- Same drug in admission and discharge lists → separate mentions; label each by its own section.
- Strength vs dosage: "Ibuprofen 800 mg, take 1 tablet" → strength 800 mg, dosage 1 tablet.
- Insulin and other injectables: "10 units" is the dosage (amount given), not the strength. Strength is a concentration like 100 units/mL.
- Inhalers: "Inhaler" is the form; the route is inhalation.
- Abbreviations: label as written (PO, BID, PRN); don't expand them in the span.
- Conflicts between sections: label each mention as written; never resolve which one is right.
- Unclear status: action = unknown. Don't guess.

## Derived patient status (rules, not labels)
- new: start + at discharge
- changed: increase / decrease / other change + at discharge
- stopped: stop + at discharge (or stop with future temporality)
- continued: no change + at discharge
- not shown as a discharge medicine: one-time dose, or setting = in hospital
- unclear: unknown action or setting, or conflicting mentions

## Deliberately not labeled
- Actor (who made the change): low agreement in CMED, not needed by patients.
- Adverse drug events: out of scope.

## Process
- Tool: Label Studio.
- Pilot: 20–30 documents; update this guide whenever a new case comes up (note it in the changelog).
- Consistency check: re-label ~10% a week later without looking; report agreement.

## Changelog
- v0.1: first version.
- v0.2: insulin units = dosage; inhaler = form (found while testing the rules extractor).
