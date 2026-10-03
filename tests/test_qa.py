"""Phase 8 Automated Test Suite: Evidence-Grounded Document Q&A.

Covers all 21 minimum requirements specified in the project context pack:
1. Question validation
2. Empty question rejection
3. Maximum question length
4. top_k validation
5. threshold validation
6. Successful retrieval
7. Document isolation
8. Insufficient evidence abstention
9. Gemini prompt construction
10. Valid structured answer parsing
11. Unsupported clause ID rejection
12. Fabricated quote rejection
13. Source location validation
14. Gemini unavailable handling
15. Malformed Gemini output
16. Missing document
17. Unindexed document
18. API endpoint success
19. API endpoint insufficient evidence
20. API endpoint LLM failure
21. Phase 1-7 regression
"""
import json
from unittest.mock import patch, MagicMock
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import settings
from backend.document.models import Document, Clause, SourceLocation
from backend.document.store import save_document
from backend.retrieval.vector_store import get_vector_store, DocumentNotIndexedError
from backend.api.schemas import QuestionRequest, QuestionResponse
from backend.llm.models import (
    QuestionAnswerOutput,
    QuestionSource,
    QAValidationResult,
)
from backend.llm.prompts import (
    QA_SYSTEM_INSTRUCTION,
    build_qa_prompt,
)
from backend.llm.validator import EvidenceValidator
from backend.llm.client import (
    GeminiService,
    MockGeminiService,
    LLMUnavailableError,
    LLMServiceError,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_clauses_doc_a():
    return [
        Clause(
            clause_id="doc_a_c1",
            document_id="doc_a",
            section_title="2. User Accounts and Termination",
            clause_index=1,
            text="You may terminate your account at any time. However, we reserve the right to suspend or terminate your account immediately, without notice or liability, for any reason whatsoever.",
            primary_category="Account",
            attention_level="High Attention",
            source_location=SourceLocation(page_number=1, start_char=0, end_char=170),
        ),
        Clause(
            clause_id="doc_a_c2",
            document_id="doc_a",
            section_title="3. Payment, Billing, and Auto-Renewal",
            clause_index=2,
            text="All paid plans are billed in advance on a recurring monthly basis. Your subscription will automatically renew at the end of each billing cycle unless you cancel it at least thirty (30) days prior to the expiration date. All subscription fees paid are non-refundable.",
            primary_category="Payment",
            attention_level="High Attention",
            source_location=SourceLocation(page_number=1, start_char=175, end_char=445),
        ),
        Clause(
            clause_id="doc_a_c3",
            document_id="doc_a",
            section_title="8. Data Retention Following Account Closure",
            clause_index=3,
            text="Upon termination of your account, we may retain your personal data, usage logs, and content for a period of up to seven (7) years to comply with regulatory obligations and enforce our agreements.",
            primary_category="Privacy & Data",
            attention_level="Medium Attention",
            source_location=SourceLocation(page_number=2, start_char=450, end_char=650),
        ),
    ]


@pytest.fixture
def sample_clauses_doc_b():
    return [
        Clause(
            clause_id="doc_b_c1",
            document_id="doc_b",
            section_title="1. Privacy and Third-Party Sharing",
            clause_index=1,
            text="We share your personal information and browsing activity with advertising partners and data brokers for commercial tracking and marketing purposes.",
            primary_category="Privacy & Data",
            attention_level="High Attention",
            source_location=SourceLocation(page_number=1, start_char=0, end_char=148),
        ),
    ]


@pytest.fixture
def setup_test_documents(sample_clauses_doc_a, sample_clauses_doc_b):
    """Set up and index Document A and Document B in vector store."""
    doc_a = Document(
        document_id="doc_test_qa_a",
        filename="terms_a.txt",
        raw_text="Document A Content",
        cleaned_text="Document A Content",
        source_type="text",
        clauses=sample_clauses_doc_a,
    )
    for c in doc_a.clauses:
        c.document_id = doc_a.document_id

    doc_b = Document(
        document_id="doc_test_qa_b",
        filename="privacy_b.txt",
        raw_text="Document B Content",
        cleaned_text="Document B Content",
        source_type="text",
        clauses=sample_clauses_doc_b,
    )
    for c in doc_b.clauses:
        c.document_id = doc_b.document_id

    save_document(doc_a)
    save_document(doc_b)

    vector_store = get_vector_store()
    vector_store.index_document(doc_a.document_id, doc_a.clauses)
    vector_store.index_document(doc_b.document_id, doc_b.clauses)

    return doc_a, doc_b


# ---------------------------------------------------------------------------
# 1. Question Validation
# ---------------------------------------------------------------------------
def test_question_validation():
    req = QuestionRequest(question="  Can the company terminate my account?   ")
    assert req.question == "Can the company terminate my account?"
    assert req.top_k == 5
    assert req.threshold is None


# ---------------------------------------------------------------------------
# 2. Empty Question Rejection
# ---------------------------------------------------------------------------
def test_empty_question_rejection():
    with pytest.raises(ValidationError):
        QuestionRequest(question="")

    with pytest.raises(ValidationError):
        QuestionRequest(question="     ")


# ---------------------------------------------------------------------------
# 3. Maximum Question Length
# ---------------------------------------------------------------------------
def test_maximum_question_length():
    valid_q = "A" * 1000
    req = QuestionRequest(question=valid_q)
    assert len(req.question) == 1000

    too_long = "A" * 1001
    with pytest.raises(ValidationError):
        QuestionRequest(question=too_long)


# ---------------------------------------------------------------------------
# 4. top_k Validation
# ---------------------------------------------------------------------------
def test_top_k_validation():
    req1 = QuestionRequest(question="Valid question", top_k=1)
    assert req1.top_k == 1

    req10 = QuestionRequest(question="Valid question", top_k=10)
    assert req10.top_k == 10

    with pytest.raises(ValidationError):
        QuestionRequest(question="Valid question", top_k=0)

    with pytest.raises(ValidationError):
        QuestionRequest(question="Valid question", top_k=11)


# ---------------------------------------------------------------------------
# 5. Threshold Validation
# ---------------------------------------------------------------------------
def test_threshold_validation():
    req = QuestionRequest(question="Valid question", threshold=0.45)
    assert req.threshold == 0.45

    with pytest.raises(ValidationError):
        QuestionRequest(question="Valid question", threshold=-0.1)

    with pytest.raises(ValidationError):
        QuestionRequest(question="Valid question", threshold=1.5)


# ---------------------------------------------------------------------------
# 6. Successful Retrieval
# ---------------------------------------------------------------------------
def test_successful_retrieval(setup_test_documents):
    doc_a, _ = setup_test_documents
    vector_store = get_vector_store()
    res = vector_store.search(
        document_id=doc_a.document_id,
        query="Will my subscription renew automatically?",
        top_k=3,
        threshold=0.30,
    )
    assert res.sufficient_evidence is True
    assert len(res.results) > 0
    # Top match should be the auto-renewal clause
    top_result = res.results[0]
    assert "auto-renewal" in top_result.section_title.lower() or "billing" in top_result.section_title.lower()


# ---------------------------------------------------------------------------
# 7. Document Isolation (Mandatory)
# ---------------------------------------------------------------------------
def test_document_isolation(setup_test_documents):
    doc_a, doc_b = setup_test_documents
    vector_store = get_vector_store()

    # Query Document A about third-party sharing (which is only in Document B)
    res_a = vector_store.search(
        document_id=doc_a.document_id,
        query="Does the company share my data with advertising partners and data brokers?",
        top_k=5,
        threshold=0.30,
    )

    # Strictly verify NO clause from Document B appears in Document A's retrieval
    for r in res_a.results:
        assert r.document_id == doc_a.document_id
        assert not r.clause_id.startswith("doc_b")
        assert "advertising partners and data brokers" not in r.text


# ---------------------------------------------------------------------------
# 8. Insufficient Evidence Abstention
# ---------------------------------------------------------------------------
def test_insufficient_evidence_abstention(setup_test_documents):
    doc_a, _ = setup_test_documents
    vector_store = get_vector_store()

    # Query with strict threshold that cannot be met
    res = vector_store.search(
        document_id=doc_a.document_id,
        query="Does the company manufacture nuclear reactors or submarines?",
        top_k=3,
        threshold=0.85,
    )
    assert res.sufficient_evidence is False


# ---------------------------------------------------------------------------
# 9. Gemini Prompt Construction
# ---------------------------------------------------------------------------
def test_gemini_prompt_construction(sample_clauses_doc_a):
    prompt = build_qa_prompt(
        question="Can the company terminate my account?",
        clauses=sample_clauses_doc_a[:1],
        scores={"doc_a_c1": 0.75},
    )
    assert "Can the company terminate my account?" in prompt
    assert "SUPPLIED EVIDENCE:" in prompt
    assert "doc_a_c1" in prompt
    assert "User Accounts and Termination" in prompt
    assert "0.75" in prompt

    # Verify system instruction rules
    assert "You are answering a question about a Terms of Service document." in QA_SYSTEM_INSTRUCTION
    assert "Do not make legal conclusions." in QA_SYSTEM_INSTRUCTION
    assert "Do not use outside knowledge." in QA_SYSTEM_INSTRUCTION


# ---------------------------------------------------------------------------
# 10. Valid Structured Answer Parsing
# ---------------------------------------------------------------------------
def test_valid_structured_answer_parsing():
    json_data = {
        "answer": "Yes, your subscription will automatically renew at the end of each billing cycle.",
        "evidence_sufficient": True,
        "confidence": 0.88,
        "sources": [
            {
                "clause_id": "doc_a_c2",
                "section_title": "3. Payment, Billing, and Auto-Renewal",
                "source_location": "Page 1",
                "quoted_text": "Your subscription will automatically renew at the end of each billing cycle",
            }
        ],
        "uncertainty": "",
        "disclaimer": "Informational document analysis only; not legal advice.",
    }
    parsed = QuestionAnswerOutput(**json_data)
    assert parsed.evidence_sufficient is True
    assert parsed.confidence == 0.88
    assert len(parsed.sources) == 1
    assert parsed.sources[0].clause_id == "doc_a_c2"


# ---------------------------------------------------------------------------
# 11. Unsupported Clause ID Rejection
# ---------------------------------------------------------------------------
def test_unsupported_clause_id_rejection(sample_clauses_doc_a):
    output = QuestionAnswerOutput(
        answer="The company terminates accounts immediately.",
        evidence_sufficient=True,
        confidence=0.85,
        sources=[
            QuestionSource(
                clause_id="non_existent_clause_999",
                section_title="User Accounts",
                source_location="Page 1",
                quoted_text="suspend or terminate your account immediately",
            )
        ],
    )
    val = EvidenceValidator.validate_qa(output, sample_clauses_doc_a)
    assert val.is_valid is False
    assert any("Unsupported clause ID" in e for e in val.validation_errors)
    assert val.output.evidence_sufficient is False


# ---------------------------------------------------------------------------
# 12. Fabricated Quote Rejection
# ---------------------------------------------------------------------------
def test_fabricated_quote_rejection(sample_clauses_doc_a):
    output = QuestionAnswerOutput(
        answer="The company offers a full refund within 90 days.",
        evidence_sufficient=True,
        confidence=0.85,
        sources=[
            QuestionSource(
                clause_id="doc_a_c2",
                section_title="3. Payment, Billing, and Auto-Renewal",
                source_location="Page 1",
                quoted_text="The company guarantees a 100% full money back refund within 90 days of purchase.",
            )
        ],
    )
    val = EvidenceValidator.validate_qa(output, sample_clauses_doc_a)
    assert val.is_valid is False
    assert any("Fabricated or modified quotation" in e for e in val.validation_errors)
    assert val.output.evidence_sufficient is False


# ---------------------------------------------------------------------------
# 13. Source Location Validation
# ---------------------------------------------------------------------------
def test_source_location_validation(sample_clauses_doc_a):
    output = QuestionAnswerOutput(
        answer="The company may retain data.",
        evidence_sufficient=True,
        confidence=0.85,
        sources=[
            QuestionSource(
                clause_id="doc_a_c3",
                section_title="8. Data Retention Following Account Closure",
                source_location="Page 99",  # Expected Page 2
                quoted_text="we may retain your personal data, usage logs, and content for a period of up to seven (7) years",
            )
        ],
    )
    val = EvidenceValidator.validate_qa(output, sample_clauses_doc_a)
    assert val.is_valid is False
    assert any("Location mismatch" in e for e in val.validation_errors)


def test_gemini_unavailable_handling(sample_clauses_doc_a):
    with patch("backend.config.settings.GEMINI_MOCK_MODE", False):
        service = GeminiService(api_key="")
        with pytest.raises(LLMUnavailableError):
            service.answer_question("test question", sample_clauses_doc_a)



# ---------------------------------------------------------------------------
# 15. Malformed Gemini Output
# ---------------------------------------------------------------------------
def test_malformed_gemini_output(sample_clauses_doc_a):
    service = GeminiService(api_key="dummy-key")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "NOT_A_VALID_JSON_STRING"
    mock_client.models.generate_content.return_value = mock_response

    with patch.object(service, "_get_client", return_value=mock_client):
        with pytest.raises(LLMServiceError) as exc_info:
            service.answer_question("Can my account be terminated?", sample_clauses_doc_a)
        assert "Malformed JSON" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 16. Missing Document
# ---------------------------------------------------------------------------
def test_missing_document_endpoint():
    resp = client.post(
        "/api/documents/non_existent_doc_id/ask",
        json={"question": "Will my subscription renew automatically?"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 17. Unindexed Document
# ---------------------------------------------------------------------------
def test_unindexed_document_endpoint():
    doc_unindexed = Document(
        document_id="doc_unindexed_qa",
        filename="unindexed.txt",
        raw_text="Some text",
        cleaned_text="Some text",
        source_type="text",
        clauses=[],
    )
    save_document(doc_unindexed)

    resp = client.post(
        f"/api/documents/{doc_unindexed.document_id}/ask",
        json={"question": "Will my subscription renew automatically?"},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 18. API Endpoint Success (POST /api/documents/{id}/ask)
# ---------------------------------------------------------------------------
def test_api_endpoint_success(setup_test_documents):
    doc_a, _ = setup_test_documents
    with patch("backend.config.settings.GEMINI_MOCK_MODE", True):
        resp = client.post(
            f"/api/documents/{doc_a.document_id}/ask",
            json={
                "question": "Will my subscription renew automatically?",
                "top_k": 3,
                "threshold": 0.30,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["evidence_sufficient"] is True
        assert data["answer"] is not None
        assert "renew" in data["answer"].lower()
        assert len(data["sources"]) > 0
        assert data["disclaimer"] == "Informational document analysis only; not legal advice."


# ---------------------------------------------------------------------------
# 19. API Endpoint Insufficient Evidence
# ---------------------------------------------------------------------------
def test_api_endpoint_insufficient_evidence(setup_test_documents):
    doc_a, _ = setup_test_documents
    with patch("backend.config.settings.GEMINI_MOCK_MODE", True):
        resp = client.post(
            f"/api/documents/{doc_a.document_id}/ask",
            json={
                "question": "Does the company manufacture nuclear reactors or submarines?",
                "top_k": 3,
                "threshold": 0.85,  # High threshold that will not match
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "insufficient_evidence"
        assert data["evidence_sufficient"] is False
        assert data["answer"] is None
        assert len(data["sources"]) == 0
        assert "enough evidence" in data["message"].lower()


# ---------------------------------------------------------------------------
# 20. API Endpoint LLM Failure
# ---------------------------------------------------------------------------
def test_api_endpoint_llm_failure(setup_test_documents):
    doc_a, _ = setup_test_documents
    with patch("backend.api.routes.get_llm_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.answer_question.side_effect = LLMUnavailableError("Gemini API key is not configured.")
        mock_get_svc.return_value = mock_svc

        resp = client.post(
            f"/api/documents/{doc_a.document_id}/ask",
            json={
                "question": "Can the company terminate my account?",
                "top_k": 3,
                "threshold": 0.30,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "llm_unavailable"
        assert data["answer"] is None
        # Preserves retrieved source clauses for user transparency
        assert len(data["sources"]) > 0
        assert "unavailable" in data["message"].lower()


# ---------------------------------------------------------------------------
# 21. Phase 1-7 Regression Test
# ---------------------------------------------------------------------------
def test_phase1_to_7_regression(setup_test_documents):
    doc_a, _ = setup_test_documents

    # Phase 2 & 3: Segmented and Classified
    assert len(doc_a.clauses) == 3
    assert doc_a.clauses[0].primary_category == "Account"
    assert doc_a.clauses[1].primary_category == "Payment"

    # Phase 4: Vector search
    store = get_vector_store()
    res = store.search(document_id=doc_a.document_id, query="renewal cancellation", top_k=2)
    assert res.total_matches > 0


    # Phase 5: Cross-clause relationship detection
    from backend.analysis.relationships import detect_relationships
    rels, metadata = detect_relationships(doc_a.clauses, doc_a.document_id)
    assert isinstance(rels, list)
    assert metadata["candidate_pairs"] > 0

    # Phase 6: Explanations
    mock_llm = MockGeminiService()
    exp = mock_llm.explain_clause(doc_a.clauses[0])
    assert exp.evidence_sufficient is True

    # Phase 7: Health & Frontend
    health_res = client.get("/api/health")
    assert health_res.status_code == 200

    ui_res = client.get("/app/")
    assert ui_res.status_code == 200
    assert "ToS Red-Flag Scanner" in ui_res.text

