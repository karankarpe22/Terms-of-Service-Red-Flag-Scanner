import io
import pytest
from tests.test_extractor import create_pdf_with_text, create_image_only_pdf


def test_api_submit_text_success(client):
    """Verify POST /api/documents/text successfully extracts, cleans, and segments text."""
    payload = {
        "title": "API Test Terms",
        "text": (
            "1. Introduction\nWelcome to our service.\n\n"
            "2. Termination\nWe may terminate your account at any time."
        ),
    }
    response = client.post("/api/documents/text", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["document_id"].startswith("doc_")
    assert data["source_type"] == "text"
    assert data["filename"] == "API Test Terms"
    assert data["total_clauses"] == 2
    assert len(data["clauses"]) == 2
    assert "1. Introduction" in data["clauses"][0]["section_title"]
    assert "2. Termination" in data["clauses"][1]["section_title"]
    assert data["clauses"][1]["primary_category"] == "Account"
    assert data["clauses"][1]["attention_level"] in ("Medium Attention", "High Attention")
    assert 0.0 <= data["clauses"][1]["category_confidence"] <= 1.0
    assert len(data["clauses"][1]["attention_reasons"]) > 0


def test_api_submit_text_empty_fails(client):
    """Verify POST /api/documents/text returns 422 or 400 when text is too short or empty."""
    response = client.post("/api/documents/text", json={"text": ""})
    assert response.status_code in (400, 422)


def test_api_upload_pdf_success(client):
    """Verify POST /api/documents/upload accepts valid PDF and returns segmented clauses."""
    pdf_bytes = create_pdf_with_text([
        "1. Acceptance of Terms\nUsers agree to comply with all rules.\n\n"
        "2. Privacy Policy\nData is stored securely on cloud servers."
    ])

    files = {"file": ("test_tos.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    response = client.post("/api/documents/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["document_id"].startswith("doc_")
    assert data["source_type"] == "pdf"
    assert data["filename"] == "test_tos.pdf"
    assert data["total_clauses"] == 2
    assert len(data["clauses"]) == 2
    assert "Page 1" in data["clauses"][0]["source_location"]
    assert data["clauses"][1]["primary_category"] == "Privacy & Data"
    assert data["clauses"][1]["attention_level"] is not None


def test_api_upload_non_pdf_fails(client):
    """Verify POST /api/documents/upload rejects non-PDF file extensions."""
    files = {"file": ("malicious.exe", io.BytesIO(b"binary data"), "application/octet-stream")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "Only machine-readable PDF" in response.json()["detail"]


def test_api_upload_empty_pdf_fails(client):
    """Verify POST /api/documents/upload rejects 0-byte file."""
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400


def test_api_upload_image_only_pdf_fails(client):
    """Verify POST /api/documents/upload returns clear message for scanned/image-only PDFs."""
    image_pdf_bytes = create_image_only_pdf()
    files = {"file": ("scanned.pdf", io.BytesIO(image_pdf_bytes), "application/pdf")}
    response = client.post("/api/documents/upload", files=files)
    assert response.status_code == 400
    assert "Scanned or image-only PDFs are not supported" in response.json()["detail"]


def test_api_get_document_by_id(client):
    """Verify GET /api/documents/{document_id} retrieves a previously ingested document."""
    # First ingest via text
    post_res = client.post("/api/documents/text", json={
        "title": "Retrieval Test",
        "text": "1. Test Clause\nThis clause should be retrievable by ID.",
    })
    doc_id = post_res.json()["document_id"]

    # Now retrieve by ID
    get_res = client.get(f"/api/documents/{doc_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["document_id"] == doc_id
    assert data["total_clauses"] == 1
    assert data["clauses"][0]["text"] == "This clause should be retrievable by ID."


def test_api_get_nonexistent_document_returns_404(client):
    """Verify GET /api/documents/{document_id} returns 404 for unknown IDs."""
    response = client.get("/api/documents/doc_does_not_exist_999")
    assert response.status_code == 404


def test_api_search_document_success(client):
    """Verify POST /api/documents/{document_id}/search returns semantic matches from FAISS."""
    # Ingest document
    post_res = client.post("/api/documents/text", json={
        "title": "Search Test ToS",
        "text": (
            "1. Subscriptions and Payments\n"
            "Paid subscriptions renew automatically every month and are non-refundable.\n\n"
            "2. Governing Law\n"
            "These terms are governed by the laws of California."
        ),
    })
    doc_id = post_res.json()["document_id"]

    # Search for auto-renewal
    search_res = client.post(f"/api/documents/{doc_id}/search", json={
        "query": "Will my subscription renew automatically?",
        "top_k": 2,
    })
    assert search_res.status_code == 200
    data = search_res.json()

    assert data["document_id"] == doc_id
    assert data["sufficient_evidence"] is True
    assert len(data["results"]) >= 1
    assert "renew automatically" in data["results"][0]["text"]
    assert data["results"][0]["primary_category"] == "Payment"
    assert data["results"][0]["meets_threshold"] is True


def test_api_search_with_category_filtering(client):
    """Verify search endpoint respects category filtering."""
    post_res = client.post("/api/documents/text", json={
        "title": "Category Filter ToS",
        "text": (
            "1. Payment Terms\nAll subscription charges recur monthly.\n\n"
            "2. Dispute Forum\nArbitration is mandatory for all claims."
        ),
    })
    doc_id = post_res.json()["document_id"]

    # Search with Dispute category filter
    search_res = client.post(f"/api/documents/{doc_id}/search", json={
        "query": "conditions and monthly terms",
        "categories": ["Dispute"],
        "top_k": 2,
    })
    assert search_res.status_code == 200
    data = search_res.json()

    for item in data["results"]:
        assert item["primary_category"] == "Dispute"


def test_api_search_nonexistent_document_returns_404(client):
    """Verify search on unknown document ID returns 404."""
    response = client.post("/api/documents/doc_ghost_id/search", json={"query": "test query"})
    assert response.status_code == 404
