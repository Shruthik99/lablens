from fastapi.testclient import TestClient

from lablens.api import app

client = TestClient(app)


def test_index_page_served():
    response = client.get("/")

    assert response.status_code == 200
    assert "Research prototype. Not medical advice." in response.text


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_returns_medication():
    response = client.post("/extract", json={"text": "Ibuprofen 800 mg PO TID PRN"})

    assert response.status_code == 200
    body = response.json()
    assert body["extractor"] == "rules"
    assert len(body["medications"]) == 1
    med = body["medications"][0]
    assert med["drug"] == "ibuprofen"
    assert med["strength"] == "800 mg"


def test_extract_rejects_empty_text():
    response = client.post("/extract", json={"text": ""})

    assert response.status_code == 422


def test_extract_rejects_too_long_text():
    response = client.post("/extract", json={"text": "a" * 20_001})

    assert response.status_code == 422
