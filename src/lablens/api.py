"""FastAPI service exposing the rule-based medication extractor.

Request text is never logged: it may contain protected health information.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from lablens.extract_rules import extract, warm_lexicon_cache
from lablens.schema import Medication

app = FastAPI(title="Lab Lens API")

# Build the lexicon index at import, so the first request does not pay for it.
warm_lexicon_cache()

STATIC_DIR = Path(__file__).parent / "static"


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class ExtractResponse(BaseModel):
    extractor: str
    medications: list[Medication]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
def extract_medications(request: ExtractRequest) -> ExtractResponse:
    return ExtractResponse(extractor="rules", medications=extract(request.text))
