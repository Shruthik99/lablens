# Lab Lens
Research prototype: extracts and explains medications from discharge documents.

## Commands
make setup | make lint | make test

## Rules
- Never edit anything in data/gold/ or tests/fixtures/expected/
- Never change metric thresholds or delete tests to make CI pass
- Small changes only; explain what you changed
- No real patient data anywhere in this repo
- Never send document text to external APIs; use the local model

## Hook-enforced protections
`.claude/settings.json` runs `.claude/hooks/protect_files.py` on every Read/Edit/Write/MultiEdit:
- `data/` and any `.env` file: blocked from Read, Edit, and Write
- `tests/fixtures/`: blocked from Edit and Write (reading is still allowed)

`.claude/hooks/format_python.py` runs after Edit/Write/MultiEdit and auto-formats the
touched file with `ruff --fix` and `black` if it's a `.py` file.