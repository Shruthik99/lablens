# Background

## The problem

Discharge documents are written for clinicians, not patients. A 2024 systematic review of older adults discharged from emergency departments found that patients understood their medication instructions less well than their self-care and follow-up instructions, and named medication instructions a priority for improvement (Haimovich et al., 2024).

I started this project after a family member's surgery, when the discharge papers used terms we did not understand.

## What research has found

- Zaretsky et al. (2024) used GPT-4 to rewrite 50 discharge summaries for patients. The reading level dropped from about 11th grade to 6th grade, but only 54 of 100 physician reviews gave the top accuracy rating, and 18 of 100 reviews noted safety concerns.
- Williams et al. (2025) found that LLM-written discharge narratives had similar overall quality to physicians' but more errors per summary (2.91 vs 1.82) and a higher, though still low, potential-harm score (0.84 vs 0.36).
- Small et al. (2025) found that LLM hospital-course summaries were more complete but contained more confabulations than clinicians' summaries.
- Goldsmith et al. (2026) compared 1,294 discharge summaries from 5 London hospitals (GPT-4o mini vs clinicians). Physicians approved the LLM summaries more often, but 6.0% of them contained hallucinations, 23.5% left out key information, and only 18.8% were perfect in every area. The authors conclude that clinician oversight and transparent links to source data are essential. These summaries were written for clinicians, not patients.
- Liu et al. (2026) found that small open LLMs rewriting clinical notes kept most facts correct but left details out, especially in long notes, and confused timing and measurements.

Taken together: LLMs can make clinical text easier to read, but usefulness and safety have to be measured separately.

## Existing tools

- Epic Emmie / Ask Emmie: an AI assistant in the MyChart patient portal that answers questions about the patient's own record.
- Oracle Health: a patient portal assistant that explains records in plain language.
- General assistants (ChatGPT Health, Copilot Health, Claude for Healthcare) accept uploaded files and records.

## How Lab Lens is different

Lab Lens is a research prototype, not a product. What it tries to combine:

- works on a document the patient has, from any hospital
- shows the exact source text for every patient-specific fact
- checks medication facts (dose, frequency, status) and says "your document doesn't say" instead of guessing
- local-first processing
- measures and reports its own errors (omissions, unsupported claims, wrong status)
- English and Hindi, with drug names and doses kept exactly as written

## How these findings shape the evaluation

| Finding | Lab Lens response |
|---|---|
| Medication instructions are understood less well than self-care and follow-up | Medication extraction is the core task |
| Readable summaries still had accuracy and safety problems | Readability and factual accuracy measured separately |
| LLM summaries omit information | Medication recall, source-fact coverage, warning-sign retention |
| LLMs confabulate | Unsupported-claim rate, source-span checks |
| LLMs confuse timing and recency | Status accuracy and event-order checks |
| Human review is still needed | Research-only scope, no clinical claims |

## References

1. Haimovich AD et al. Discharge instruction comprehension by older adults in the emergency department: a systematic review and meta-analysis. Acad Emerg Med. 2024;31(11):1165–1172.
2. Zaretsky J et al. Generative artificial intelligence to transform inpatient discharge summaries to patient-friendly language and format. JAMA Netw Open. 2024;7(3):e240357.
3. Williams CYK et al. Physician- and large language model–generated hospital discharge summaries. JAMA Intern Med. 2025;185(7):818–825.
4. Small WR et al. Hospital course summarization by an electronic health record–based large language model. JAMA Netw Open. 2025;8(8):e2526339.
5. Goldsmith K, Arribas M, Damirova F, et al. Clinician acceptance of large language model–generated discharge summaries. JAMA Netw Open. 2026;9(11):e2644496.
6. Liu J, Soni S, Nguyen A. Systematic evaluation of the quality of synthetic clinical notes rephrased by LLMs at million-note scale. Proceedings of BioNLP 2026: 353–371.