# Lab Lens

Research prototype that helps patients and families understand hospital discharge documents.
It extracts each medication (drug, dose, route, frequency, status) with the exact source text,
explains medical terms, and answers questions only from the document and trusted sources.

## Status
Early development. Done: project setup, data inspection, annotation guide, rule-based /
local-LLM / fine-tuned-T5 extractors, evaluation with MLflow tracking, RxNorm lexicon.
Next: gold labeling on the held-out notes.

## Medication lexicon
The rules extractor matches against RxNorm concepts fetched from the NLM RxNav API
(`make build-lexicon`, committed as `src/lablens/resources/rxnorm_lexicon.tsv` with RxCUI,
term type and RxNorm version). Names that are also ordinary English — `iron`, `oxygen`,
`sodium` — are kept but flagged, and the `*_context` modes emit them only when a dose,
route or medication verb is nearby. See
[docs/decisions/0001-rxnorm-lexicon-ambiguity.md](docs/decisions/0001-rxnorm-lexicon-ambiguity.md)
and compare configurations with `make eval-modes`.

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