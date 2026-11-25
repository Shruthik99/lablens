# Lab Lens

Research prototype that helps patients and families understand hospital discharge documents.
It extracts each medication (drug, dose, route, frequency, status) with the exact source text,
explains medical terms, and answers questions only from the document and trusted sources.

## Status
Early development. Done: project setup, data inspection. Next: annotation guide and baseline extractor.

## Scope
- Research prototype, not medical advice, not for clinical use. No regulatory claims are made.
- Explains what a document says; does not judge severity or give medical advice.
- Evaluated and demoed only on public, fictional or synthetic documents. Real patient documents are out of scope.
- Local-first: document processing and the language model run locally. NLM lookups (RxNorm, RxClass, MeSH) use the network but receive only drug names and terms, never document text.

## Data
See [DATA_LICENSES.md](DATA_LICENSES.md) for sources and licenses, and
[docs/data-inspection.md](docs/data-inspection.md) for what the data looks like.

## Setup
```
python -m venv .venv
.venv\Scripts\Activate.ps1
make setup
make lint
make test
```