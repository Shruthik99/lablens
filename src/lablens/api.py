"""FastAPI service exposing the rule-based medication extractor.

Request text is never logged: it may contain protected health information.
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from lablens.extract_rules import extract
from lablens.schema import Medication

app = FastAPI(title="Lab Lens API")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class ExtractResponse(BaseModel):
    extractor: str
    medications: list[Medication]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract")
def extract_medications(request: ExtractRequest) -> ExtractResponse:
    return ExtractResponse(extractor="rules", medications=extract(request.text))
