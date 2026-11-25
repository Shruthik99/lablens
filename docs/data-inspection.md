# Data inspection

Checked each dataset before building anything. Notebook: notebooks/01_inspect_notes.ipynb

## Augmented Clinical Notes (main dataset)
- 30,000 case-report notes. Sampled 40 (seed 42).
- `note` is cut at ~300 words; `full_note` is the complete text (315–1,117 words, median 459).
- Most notes are longer than Flan-T5's input limit, so they need chunking.
- Summary JSON needed `strict=False`; `treatments` was a plain string in 2/40.
- Treatments: 66 across 32 notes. Dose 42%, frequency 35%, no route or status field.
- Treatments mix real drugs, drug classes and procedures, and describe history, not discharge orders.
- Age and sex present in 40/40.
- Conclusion: the existing labels are weak hints. Gold labels have to come from my own annotation.
- Summaries were generated from the truncated `note`: 64/66 treatment names appear in the first ~300 words, only 1 after. Weak-label training uses `note` as input; my gold labels use `full_note`.

## MedQuAD
- 16,407 of 47,441 rows have answers. Mostly genetic and rare diseases.
- Good for diseases, missing procedures (craniotomy, intubation, catheter: 0).
- License: CC BY 4.0.
- Use: disease explanations.

## MeSH
- Free NLM API. Found 6/7 test terms, including the procedures MedQuAD missed.
- Prefix matching picked a wrong entry ("catheter" → Catheter Ablation).
- Phrases like "perilesional edema" are not entries.
- The API dropped connections a few times; worked on retry. Lookups need retries, caching and a fallback.
- Terms: credit NLM, state the MeSH version.
- Use: procedure and term explanations, with phrase splitting and a "not in our references" fallback.

## Noisy Medical Document Images (Kaggle)
- 500 discharge summary images, one JSON answer each (not the full page text).
- Only 5 diagnoses and 19 unique medication lines.
- Mild noise, digital renders, real hospital names.
- Use: OCR testing only. Not for training (a model would memorize it).

## ClinOCR-Bench
- 384 documents (328 in the test split), 16 templates, 6 subsets (normal, handwriting, poor, rotated, tables, mixed).
- Documents are fictional (LLM-written content), but the damage is physical: printed, crumpled, copied, rotated, then scanned or photographed.
- Full page text as ground truth, so page-level OCR error can be measured.
- License: MIT (checked on the dataset page).
- Use: main OCR test set. Test split only. Not proof of performance on real hospital paperwork.

## Medicine lookups
- RxNorm: found 5/5 test drugs.
- openFDA labels: returned wrong products (OTC for 800 mg ibuprofen, a homeopathic and then a multivitamin label for ferrous sulfate, nothing for belimumab).
- RxClass EPC class: 4/5 drugs, no combination-product noise. Ferrous sulfate has no EPC; its physiologic effect (PE) class works as a fallback ("Increased Erythroid Cell Production"). PE says what a drug does, not its type.
- ATC, VA and disease relations in RxClass were not usable (combination products, odd classes, contraindications mixed in).
- Decision: "what this medicine is" comes from the RxClass class in plain words, plus a MedlinePlus link. Label lookup is optional, later.

## Hindi
- IN22-Conv: 1,503 sentences, English + 22 Indic languages. Healthcare is 67 casual sentences, no medications.
- FLORES+: 1,012 Hindi sentences aligned with English. Terms: no re-hosting, no training on it.
- Both test general translation only. Medical meaning needs my own check set (drugs, doses, "do not take").