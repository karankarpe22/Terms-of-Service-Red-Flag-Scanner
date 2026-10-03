"""Phase 6 Automated Test Suite: Gemini Integration, Evidence Grounding, and Hallucination Protection.

Covers:
1. Configurable GEMINI_MODEL_NAME.
2. Mode control: GEMINI_MOCK_MODE=True returns MockGeminiService.
3. Mode control: GEMINI_MOCK_MODE=False with missing key raises LLMUnavailableError.
4. Mode control: GEMINI_MOCK_MODE=False with valid key initializes official genai.Client.
5. Prompts contain mandatory non-legal advice disclaimer instruction.
6. Prompts include strict verbatim quotation requirement.
7. Prompts include abstention instruction (evidence_sufficient=False).
8. Structured output schema matches ExplanationOutput model fields.
9. Validator accepts valid verbatim quotation matching source clause.
10. Validator accepts case/whitespace normalized verbatim quotation.
11. Validator rejects fabricated quotation not in source clause.
12. Validator rejects evidence reference with unsupported clause_id.
13. Validator rejects evidence reference with mismatched source_location.
14. Validator rejects AI explanation asserting forbidden legal conclusion.
15. Validator preserves verbatim quotation even if source contains words like 'illegal' or 'void'.
16. Validator sets evidence_sufficient=False and populates uncertainty upon failure.
17. FAISS abstention when query finds no clauses above threshold propagates sufficient_evidence=False.
18. POST /api/documents/{id}/explain-clause returns valid explanation structure with Mock service.
19. POST /api/documents/{id}/explain-relationship returns valid explanation structure with Mock service.
20. POST /api/documents/{id}/synthesize returns valid synthesized explanation with Mock service.
21. API endpoints return 404 for nonexistent document.
22. API endpoints return 404 for nonexistent clause or relationship.
23. API endpoint returns status='llm_unavailable' when LLM service is unavailable.
"""
from unittest.mock import patch, MagicMock
import pytest
from backend.config import settings, Settings
from backend.document.models import Clause, SourceLocation
from backend.analysis.models import ClauseRelationship, RelationshipEvidence
from backend.llm.models import ExplanationOutput, EvidenceReference, ValidationResult
from backend.llm.prompts import (
    SYSTEM_INSTRUCTION,
    build_clause_explanation_prompt,
    build_relationship_explanation_prompt,
    build_synthesis_prompt,
)
from backend.llm.validator import EvidenceValidator
from backend.llm.client import (
    GeminiService,
    MockGeminiService,
    get_llm_service,
    LLMUnavailableError,
    LLMServiceError,
)
from backend.retrieval.vector_store import get_vector_store
from backend.document.store import save_document
from backend.document.models import Document


@pytest.fixture
def sample_clause_1():
    return Clause(
        clause_id="clause_001",
        document_id="doc_test_001",
        section_title="1. Subscription and Billing",
        clause_index=1,
        text="Subscriptions automatically renew each month unless cancelled at least 48 hours prior to renewal. No refunds are granted.",
        primary_category="Payment",
        secondary_categories=["Changes"],
        category_confidence=0.92,
        attention_level="High Attention",
        attention_reasons=["Automatic renewal without prompt", "Explicit waiver of refund rights"],
        source_location=SourceLocation(page_number=1, start_char=0, end_char=124),
    )


@pytest.fixture
def sample_clause_2():
    return Clause(
        clause_id="clause_002",
        document_id="doc_test_001",
        section_title="2. Account Termination",
        clause_index=2,
        text="We reserve the right to terminate or suspend your account immediately without prior notice or liability.",
        primary_category="Account",
        secondary_categories=["Dispute"],
        category_confidence=0.88,
        attention_level="High Attention",
        attention_reasons=["Immediate termination without notice"],
        source_location=SourceLocation(page_number=1, start_char=125, end_char=231),
    )


@pytest.fixture
def sample_legal_terms_clause():
    return Clause(
        clause_id="clause_legal_001",
        document_id="doc_test_002",
        section_title="3. Severability and Illegal Acts",
        clause_index=3,
        text="Users shall not perform illegal or unlawful acts. If any provision is deemed void or unenforceable, remaining provisions survive.",
        primary_category="Dispute",
        secondary_categories=["Liability"],
        category_confidence=0.85,
        attention_level="Medium Attention",
        attention_reasons=["Severability and illegal acts clause"],
        source_location=SourceLocation(page_number=2, start_char=0, end_char=132),
    )


# 1. Configurable GEMINI_MODEL_NAME
def test_configurable_gemini_model_name():
    service_default = GeminiService(api_key="fake-key")
    assert service_default.model_name == settings.GEMINI_MODEL_NAME

    custom_service = GeminiService(api_key="fake-key", model_name="gemini-1.5-pro-preview")
    assert custom_service.model_name == "gemini-1.5-pro-preview"


# 2. Mode control: GEMINI_MOCK_MODE=True returns MockGeminiService
def test_mock_mode_returns_mock_service():
    with patch.object(settings, "GEMINI_MOCK_MODE", True):
        service = get_llm_service()
        assert isinstance(service, MockGeminiService)


# 3. Mode control: GEMINI_MOCK_MODE=False with missing key raises LLMUnavailableError
def test_missing_key_raises_unavailable():
    with patch.object(settings, "GEMINI_MOCK_MODE", False):
        with patch.object(settings, "GEMINI_API_KEY", ""):
            service = GeminiService(api_key="")
            with pytest.raises(LLMUnavailableError) as exc_info:
                service._get_client()
            assert "Gemini API key is not configured" in str(exc_info.value)


# 4. Mode control: GEMINI_MOCK_MODE=False with valid key initializes official genai.Client
def test_valid_key_initializes_gemini_client():
    with patch("google.genai.Client") as mock_client_cls:
        service = GeminiService(api_key="valid-test-key")
        client = service._get_client()
        mock_client_cls.assert_called_once_with(api_key="valid-test-key")


# 5. Prompts contain mandatory non-legal advice disclaimer instruction
def test_prompts_include_non_legal_advice_disclaimer(sample_clause_1, sample_clause_2):
    assert "not a lawyer" in SYSTEM_INSTRUCTION.lower()
    assert "do not provide legal advice" in SYSTEM_INSTRUCTION.lower()
    assert "not legal advice" in ExplanationOutput.model_fields["disclaimer"].default.lower()

    prompt_clause = build_clause_explanation_prompt(sample_clause_1)
    assert "supplied evidence" in prompt_clause.lower()

    rel = ClauseRelationship(
        relationship_id="rel_001",
        document_id="doc_test_001",
        source_clause_id=sample_clause_1.clause_id,
        target_clause_id=sample_clause_2.clause_id,
        relationship_type="POTENTIAL_TENSION",
        confidence=0.85,
        rationale="Account may be terminated while billing renews.",
        evidence=RelationshipEvidence(
            source_text=sample_clause_1.text,
            target_text=sample_clause_2.text,
            trigger="billing renewal vs immediate termination",
        ),
        source_location=sample_clause_1.source_location.to_display_string(),
        target_location=sample_clause_2.source_location.to_display_string(),
    )
    prompt_rel = build_relationship_explanation_prompt(rel, sample_clause_1, sample_clause_2)
    assert "evidence" in prompt_rel.lower()

    prompt_syn = build_synthesis_prompt([sample_clause_1, sample_clause_2], [rel])
    assert "evidence" in prompt_syn.lower()


# 6. Prompts include strict verbatim quotation requirement
def test_prompts_include_strict_verbatim_quote_instruction(sample_clause_1):
    assert "exact verbatim quotes" in SYSTEM_INSTRUCTION.lower()
    prompt = build_clause_explanation_prompt(sample_clause_1)
    assert "verbatim" in prompt.lower()


# 7. Prompts include abstention instruction (evidence_sufficient=False)
def test_prompts_include_abstention_instruction(sample_clause_1):
    assert "evidence_sufficient" in SYSTEM_INSTRUCTION
    assert "uncertainty" in SYSTEM_INSTRUCTION


# 8. Structured output schema matches ExplanationOutput model fields
def test_structured_output_schema():
    schema = ExplanationOutput.model_json_schema()
    assert "properties" in schema
    props = schema["properties"]
    expected_fields = [
        "summary",
        "attention_explanation",
        "relationship_explanation",
        "key_points",
        "evidence_references",
        "evidence_sufficient",
        "uncertainty",
        "disclaimer",
    ]
    for field in expected_fields:
        assert field in props, f"Missing field '{field}' in schema"


# 9. Validator accepts valid verbatim quotation matching source clause
def test_validator_accepts_valid_verbatim_quote(sample_clause_1):
    output = ExplanationOutput(
        summary="Clause describes subscription terms and renewal policies.",
        attention_explanation="Requires attention because of auto-renewal.",
        relationship_explanation="",
        key_points=["Auto-renewal applies monthly.", "Refunds are waived."],
        evidence_references=[
            EvidenceReference(
                clause_id=sample_clause_1.clause_id,
                source_location=sample_clause_1.source_location.to_display_string(),
                quoted_text="Subscriptions automatically renew each month unless cancelled",
            )
        ],
        evidence_sufficient=True,
        uncertainty="",
    )
    res = EvidenceValidator.validate(output, [sample_clause_1])
    assert res.is_valid is True
    assert len(res.validation_errors) == 0


# 10. Validator accepts case/whitespace normalized verbatim quotation
def test_validator_accepts_case_and_whitespace_normalized_quote(sample_clause_1):
    output = ExplanationOutput(
        summary="Clause describes subscription terms.",
        attention_explanation="Auto-renewal policy.",
        key_points=[],
        evidence_references=[
            EvidenceReference(
                clause_id=sample_clause_1.clause_id,
                source_location=sample_clause_1.source_location.to_display_string(),
                # Differing whitespace and mixed case
                quoted_text="subscriptions  automatically   renew  each month",
            )
        ],
        evidence_sufficient=True,
    )
    res = EvidenceValidator.validate(output, [sample_clause_1])
    assert res.is_valid is True
    assert len(res.validation_errors) == 0


# 11. Validator rejects fabricated quotation not in source clause
def test_validator_rejects_fabricated_quote(sample_clause_1):
    output = ExplanationOutput(
        summary="Clause claims users may cancel anytime with a full refund.",
        attention_explanation="",
        key_points=[],
        evidence_references=[
            EvidenceReference(
                clause_id=sample_clause_1.clause_id,
                source_location=sample_clause_1.source_location.to_display_string(),
                quoted_text="Users can request a full refund within 30 days of purchase.",
            )
        ],
        evidence_sufficient=True,
    )
    res = EvidenceValidator.validate(output, [sample_clause_1])
    assert res.is_valid is False
    assert any("Fabricated or modified quotation" in err for err in res.validation_errors)
    assert res.output.evidence_sufficient is False


# 12. Validator rejects evidence reference with unsupported clause_id
def test_validator_rejects_unsupported_clause_id(sample_clause_1):
    output = ExplanationOutput(
        summary="Summary citing nonexistent clause.",
        evidence_references=[
            EvidenceReference(
                clause_id="clause_999",
                source_location="Page 1, Characters 0-50",
                quoted_text="Some text",
            )
        ],
        evidence_sufficient=True,
    )
    res = EvidenceValidator.validate(output, [sample_clause_1])
    assert res.is_valid is False
    assert any("Unsupported clause ID" in err for err in res.validation_errors)


# 13. Validator rejects evidence reference with mismatched source_location
def test_validator_rejects_location_mismatch(sample_clause_1):
    output = ExplanationOutput(
        summary="Summary citing incorrect location.",
        evidence_references=[
            EvidenceReference(
                clause_id=sample_clause_1.clause_id,
                source_location="Page 99, Characters 500-600",
                quoted_text="Subscriptions automatically renew each month",
            )
        ],
        evidence_sufficient=True,
    )
    res = EvidenceValidator.validate(output, [sample_clause_1])
    assert res.is_valid is False
    assert any("Location mismatch" in err for err in res.validation_errors)


# 14. Validator rejects AI explanation asserting forbidden legal conclusion
def test_validator_rejects_forbidden_legal_conclusion_in_explanation(sample_clause_1):
    output = ExplanationOutput(
        summary="This provision is illegal under European consumer law.",
        attention_explanation="It violates GDPR and is legally unenforceable.",
        key_points=["The company is breaking the law."],
        evidence_references=[
            EvidenceReference(
                clause_id=sample_clause_1.clause_id,
                source_location=sample_clause_1.source_location.to_display_string(),
                quoted_text="Subscriptions automatically renew each month",
            )
        ],
        evidence_sufficient=True,
    )
    res = EvidenceValidator.validate(output, [sample_clause_1])
    assert res.is_valid is False
    assert any("Forbidden legal conclusion detected" in err for err in res.validation_errors)


# 15. Validator preserves verbatim quotation even if source contains words like 'illegal' or 'void'
def test_validator_preserves_legal_words_in_verbatim_quotes_and_sources(sample_legal_terms_clause):
    """Crucial requirement: Verbatim quotes containing 'illegal' or 'void' from original contract

    must NOT trigger the forbidden legal language guardrail. Only AI-generated commentary is checked.
    """
    output = ExplanationOutput(
        summary="The clause defines contractual guidelines regarding prohibited user conduct and standard severability.",
        attention_explanation="Prohibits non-compliant activity and establishes that remaining terms survive invalidation.",
        relationship_explanation="",
        key_points=["Severability applies to invalid provisions.", "Prohibits unlawful activity."],
        evidence_references=[
            EvidenceReference(
                clause_id=sample_legal_terms_clause.clause_id,
                source_location=sample_legal_terms_clause.source_location.to_display_string(),
                # Note: This verbatim quote includes 'illegal or unlawful' and 'void or unenforceable'
                quoted_text="Users shall not perform illegal or unlawful acts. If any provision is deemed void or unenforceable",
            )
        ],
        evidence_sufficient=True,
        uncertainty="",
    )
    res = EvidenceValidator.validate(output, [sample_legal_terms_clause])
    assert res.is_valid is True
    assert len(res.validation_errors) == 0


# 16. Validator sets evidence_sufficient=False and populates uncertainty upon failure
def test_validator_failure_sets_insufficient_evidence_and_uncertainty(sample_clause_1):
    output = ExplanationOutput(
        summary="Valid summary.",
        evidence_references=[
            EvidenceReference(
                clause_id="clause_invalid",
                source_location="Page 1",
                quoted_text="fake quote",
            )
        ],
        evidence_sufficient=True,
    )
    res = EvidenceValidator.validate(output, [sample_clause_1])
    assert res.is_valid is False
    assert res.output.evidence_sufficient is False
    assert "Evidence validation failed" in res.output.uncertainty


# 17. FAISS abstention when query finds no clauses above threshold propagates sufficient_evidence=False
def test_retrieval_abstention_on_insufficient_evidence(sample_clause_1):
    store = get_vector_store()
    store.index_document(
        document_id="doc_test_retrieval_abstention",
        clauses=[sample_clause_1],
    )
    res = store.search(
        document_id="doc_test_retrieval_abstention",
        query="completely unrelated query regarding nuclear physics",
        threshold=0.99,
    )
    assert res.sufficient_evidence is False
    assert all(not r.meets_threshold for r in res.results)


# 18. POST /api/documents/{id}/explain-clause returns valid explanation structure with Mock service
def test_api_explain_clause_success(client):
    with patch.object(settings, "GEMINI_MOCK_MODE", True):
        payload = {
            "title": "Clause Explanation Test",
            "text": (
                "1. Payment and Renewal\nSubscriptions renew automatically every 30 days.\n\n"
                "2. Dispute Resolution\nAll disputes must be resolved via individual binding arbitration."
            ),
        }
        upload_res = client.post("/api/documents/text", json=payload)
        assert upload_res.status_code == 200
        doc_data = upload_res.json()
        doc_id = doc_data["document_id"]
        clause_id = doc_data["clauses"][0]["clause_id"]

        res = client.post(
            f"/api/documents/{doc_id}/explain-clause",
            json={"clause_id": clause_id},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["document_id"] == doc_id
        assert data["operation"] == "explain-clause"
        assert data["explanation"] is not None
        assert len(data["explanation"]["summary"]) > 0
        assert len(data["explanation"]["evidence_references"]) >= 1
        assert data["explanation"]["evidence_references"][0]["clause_id"] == clause_id
        assert "Informational document analysis only" in data["disclaimer"]


# 19. POST /api/documents/{id}/explain-relationship returns valid explanation structure with Mock service
def test_api_explain_relationship_success(client):
    with patch.object(settings, "GEMINI_MOCK_MODE", True):
        payload = {
            "title": "Relationship Explanation Test",
            "text": (
                "1. Payment and Renewal\nYou may cancel at any time, but subscriptions renew automatically each month unless notice is given 30 days prior to Section 2.\n\n"
                "2. Account Termination\nWe may terminate your account immediately without prior notice as stated in Section 1."
            ),
        }
        upload_res = client.post("/api/documents/text", json=payload)
        assert upload_res.status_code == 200
        doc_data = upload_res.json()
        doc_id = doc_data["document_id"]

        # Detect relationships first to get a valid relationship_id
        rel_res = client.post(f"/api/documents/{doc_id}/relationships")
        assert rel_res.status_code == 200
        rel_data = rel_res.json()
        assert rel_data["relationship_count"] >= 1
        rel_id = rel_data["relationships"][0]["relationship_id"]

        # Explain the relationship
        explain_res = client.post(
            f"/api/documents/{doc_id}/explain-relationship",
            json={"relationship_id": rel_id},
        )
        assert explain_res.status_code == 200
        data = explain_res.json()
        assert data["status"] == "success"
        assert data["document_id"] == doc_id
        assert data["operation"] == "explain-relationship"
        assert data["explanation"] is not None
        assert len(data["explanation"]["relationship_explanation"]) > 0
        assert len(data["explanation"]["evidence_references"]) == 2


# 20. POST /api/documents/{id}/synthesize returns valid synthesized explanation with Mock service
def test_api_synthesize_success(client):
    with patch.object(settings, "GEMINI_MOCK_MODE", True):
        payload = {
            "title": "Synthesis Test",
            "text": (
                "1. Subscriptions\nAll fees are non-refundable.\n\n"
                "2. Content Rights\nYou grant us an irrevocable worldwide license to use your content."
            ),
        }
        upload_res = client.post("/api/documents/text", json=payload)
        assert upload_res.status_code == 200
        doc_id = upload_res.json()["document_id"]

        syn_res = client.post(
            f"/api/documents/{doc_id}/synthesize",
            json={"user_concerns": ["Payment", "User Content"]},
        )
        assert syn_res.status_code == 200
        data = syn_res.json()
        assert data["status"] == "success"
        assert data["document_id"] == doc_id
        assert data["operation"] == "synthesize"
        assert data["explanation"] is not None
        assert len(data["explanation"]["summary"]) > 0


# 21. API endpoints return 404 for nonexistent document
def test_api_explain_endpoints_404_document(client):
    res_clause = client.post("/api/documents/doc_nonexistent/explain-clause", json={"clause_id": "clause_001"})
    assert res_clause.status_code == 404

    res_rel = client.post("/api/documents/doc_nonexistent/explain-relationship", json={"relationship_id": "rel_001"})
    assert res_rel.status_code == 404

    res_syn = client.post("/api/documents/doc_nonexistent/synthesize", json={})
    assert res_syn.status_code == 404


# 22. API endpoints return 404 for nonexistent clause or relationship
def test_api_explain_endpoints_404_clause_or_relationship(client):
    payload = {
        "title": "404 Test",
        "text": "1. Section One\nSingle clause agreement text here.",
    }
    upload_res = client.post("/api/documents/text", json=payload)
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["document_id"]

    res_clause = client.post(
        f"/api/documents/{doc_id}/explain-clause",
        json={"clause_id": "nonexistent_clause_id"},
    )
    assert res_clause.status_code == 404

    res_rel = client.post(
        f"/api/documents/{doc_id}/explain-relationship",
        json={"relationship_id": "nonexistent_rel_id"},
    )
    assert res_rel.status_code == 404


# 23. API endpoint returns status='llm_unavailable' when LLM service is unavailable
def test_api_llm_unavailable_handling(client):
    payload = {
        "title": "LLM Unavailable Test",
        "text": "1. Simple Section\nClause content here.",
    }
    upload_res = client.post("/api/documents/text", json=payload)
    assert upload_res.status_code == 200
    doc_data = upload_res.json()
    doc_id = doc_data["document_id"]
    clause_id = doc_data["clauses"][0]["clause_id"]

    # Simulate LLMUnavailableError when calling explain_clause
    with patch("backend.api.routes.get_llm_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.explain_clause.side_effect = LLMUnavailableError("Gemini API key missing.")
        mock_get_service.return_value = mock_service

        res = client.post(
            f"/api/documents/{doc_id}/explain-clause",
            json={"clause_id": clause_id},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "llm_unavailable"
        assert "Gemini API key missing" in data["message"]
        assert data["explanation"] is None
