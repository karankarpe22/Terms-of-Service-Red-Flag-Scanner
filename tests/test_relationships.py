"""Comprehensive test suite for Phase 5: Cross-Clause Relationship Analysis Engine.

Validates:
1. Relationship domain model serialization and validation
2. Candidate pair generation and pruning to avoid O(N^2) explosion
3. Explicit cross-reference resolution
4. EXCEPTS detection
5. QUALIFIES detection
6. DEPENDS_ON detection
7. TEMPORAL relationship detection
8. POTENTIAL_TENSION detection (cancellation vs auto-renewal; modification vs notice)
9. Strict same-document isolation
10. Different-document cross-contamination exclusion
11. Candidate-pair limits and pruning
12. Confidence threshold filtering
13. Verbatim evidence preservation with zero fabrication
14. Comprehensive synthetic ToS multi-clause relationships
15. API endpoint integration (/api/documents/{document_id}/relationships)
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.document.models import Clause, SourceLocation, Document
from backend.analysis.models import ClauseRelationship, RelationshipEvidence
from backend.analysis.relationships import (
    CrossClauseRelationshipEngine,
    detect_relationships,
)
from backend.core.constants import (
    RELATIONSHIP_TYPE_SUPPORTS,
    RELATIONSHIP_TYPE_QUALIFIES,
    RELATIONSHIP_TYPE_EXCEPTS,
    RELATIONSHIP_TYPE_OVERRIDES,
    RELATIONSHIP_TYPE_DEPENDS_ON,
    RELATIONSHIP_TYPE_TEMPORAL,
    RELATIONSHIP_TYPE_POTENTIAL_TENSION,
)
from backend.document.store import save_document


client = TestClient(app)


def _make_clause(
    clause_id: str,
    doc_id: str,
    index: int,
    title: str,
    text: str,
    primary_category: str = "General",
    attention_level: str = "Informational",
) -> Clause:
    return Clause(
        clause_id=clause_id,
        document_id=doc_id,
        clause_index=index,
        section_title=title,
        text=text,
        primary_category=primary_category,
        category=primary_category,
        attention_level=attention_level,
        source_location=SourceLocation(start_char=100 * index, end_char=100 * index + len(text)),
    )


# ---------------------------------------------------------------------------
# 1. Model Validation
# ---------------------------------------------------------------------------
def test_relationship_model_validation():
    """Verify that ClauseRelationship and RelationshipEvidence validate and serialize correctly."""
    evidence = RelationshipEvidence(
        source_text="Subscriptions renew automatically every month.",
        target_text="Users may cancel at any time.",
        trigger="cancel at any time vs advance renewal notice",
    )
    rel = ClauseRelationship(
        relationship_id="rel_test_001",
        document_id="doc_test_123",
        source_clause_id="doc_test_123_clause_001",
        target_clause_id="doc_test_123_clause_002",
        relationship_type=RELATIONSHIP_TYPE_POTENTIAL_TENSION,
        confidence=0.78,
        rationale="The clauses describe differing timing terms.",
        evidence=evidence,
        source_location="Offset 100-200",
        target_location="Offset 250-350",
    )

    data = rel.to_dict()
    assert data["relationship_id"] == "rel_test_001"
    assert data["relationship_type"] == RELATIONSHIP_TYPE_POTENTIAL_TENSION
    assert data["confidence"] == 0.78
    assert data["evidence"]["trigger"] == "cancel at any time vs advance renewal notice"
    assert data["source_location"] == "Offset 100-200"


# ---------------------------------------------------------------------------
# 2. Candidate Pair Generation & Limits
# ---------------------------------------------------------------------------
def test_candidate_pair_generation_pruning_and_limits():
    """Verify candidate pair generation does not explode and respects configurable caps."""
    doc_id = "doc_candidate_test"
    clauses = [
        _make_clause(f"c_{i}", doc_id, i + 1, f"Section {i+1}", f"Payment clause content number {i} regarding billing.")
        for i in range(15)
    ]

    engine = CrossClauseRelationshipEngine(
        max_candidates_per_clause=3,
        max_total_candidates=25,
    )
    candidates = engine.generate_candidate_pairs(clauses)

    assert len(candidates) <= 25
    assert len(candidates) > 0
    # Verify each candidate has match reasons
    for c_a, c_b, reasons, score in candidates:
        assert len(reasons) > 0
        assert score > 0.0


# ---------------------------------------------------------------------------
# 3. Explicit Cross-Reference Detection
# ---------------------------------------------------------------------------
def test_explicit_cross_reference_detection():
    """Verify explicit cross-references (e.g. 'Section 3') are resolved between clauses."""
    doc_id = "doc_xref_test"
    c1 = _make_clause(
        "c_1", doc_id, 1, "Section 3. Payment Terms",
        "All paid plans require monthly subscription billing.",
        primary_category="Payment",
    )
    c2 = _make_clause(
        "c_2", doc_id, 2, "Section 7. Cancellations",
        "As described in Section 3, all payments are subject to cancellation terms.",
        primary_category="Payment",
    )

    rels, meta = detect_relationships([c1, c2], document_id=doc_id)
    assert len(rels) >= 1
    types = [r.relationship_type for r in rels]
    assert (RELATIONSHIP_TYPE_DEPENDS_ON in types) or (RELATIONSHIP_TYPE_SUPPORTS in types)
    for r in rels:
        assert r.document_id == doc_id
        assert "Section 3" in r.evidence.source_text or "Section 3" in r.evidence.target_text


# ---------------------------------------------------------------------------
# 4. EXCEPTS Detection
# ---------------------------------------------------------------------------
def test_excepts_detection():
    """Verify explicit exception clauses are identified."""
    doc_id = "doc_excepts_test"
    c1 = _make_clause(
        "c_1", doc_id, 1, "Section 1. General Disclaimers",
        "Except as expressly provided in this agreement, no warranties are made.",
        primary_category="Liability",
    )
    c2 = _make_clause(
        "c_2", doc_id, 2, "Section 2. Service Warranty",
        "The service is provided on an as-is and as-available basis.",
        primary_category="Liability",
    )

    rels, _ = detect_relationships([c1, c2], document_id=doc_id)
    excepts = [r for r in rels if r.relationship_type == RELATIONSHIP_TYPE_EXCEPTS]
    assert len(excepts) >= 1
    assert "Except as expressly provided" in excepts[0].evidence.trigger or "except" in excepts[0].evidence.trigger.lower()


# ---------------------------------------------------------------------------
# 5. QUALIFIES Detection
# ---------------------------------------------------------------------------
def test_qualifies_detection():
    """Verify qualifying or scope-limiting provisos are detected."""
    doc_id = "doc_qualifies_test"
    c1 = _make_clause(
        "c_1", doc_id, 1, "Section 6. Liability Limitation",
        "To the extent permitted by applicable law, our aggregate liability is limited to fifty dollars.",
        primary_category="Liability",
    )
    c2 = _make_clause(
        "c_2", doc_id, 2, "Section 7. Damages",
        "We are not liable for incidental, indirect, or consequential damages.",
        primary_category="Liability",
    )

    rels, _ = detect_relationships([c1, c2], document_id=doc_id)
    qualifies = [r for r in rels if r.relationship_type == RELATIONSHIP_TYPE_QUALIFIES]
    assert len(qualifies) >= 1
    assert "To the extent permitted by" in qualifies[0].evidence.trigger or "to the extent permitted by" in qualifies[0].evidence.trigger.lower()


# ---------------------------------------------------------------------------
# 6. DEPENDS_ON Detection
# ---------------------------------------------------------------------------
def test_depends_on_detection():
    """Verify conditional operational dependencies are detected."""
    doc_id = "doc_depends_test"
    c1 = _make_clause(
        "c_1", doc_id, 1, "Section 2. Premium Services",
        "Your access to the platform is subject to the terms of payment and active subscription status.",
        primary_category="Account",
    )
    c2 = _make_clause(
        "c_2", doc_id, 2, "Section 3. Payment Processing",
        "All paid plans require automatic recurring monthly billing.",
        primary_category="Payment",
    )

    rels, _ = detect_relationships([c1, c2], document_id=doc_id)
    depends = [r for r in rels if r.relationship_type == RELATIONSHIP_TYPE_DEPENDS_ON]
    assert len(depends) >= 1
    assert "subject to" in depends[0].evidence.trigger.lower()


# ---------------------------------------------------------------------------
# 7. TEMPORAL Detection
# ---------------------------------------------------------------------------
def test_temporal_relationship_detection():
    """Verify interaction along timeline, notice windows, or retention horizons."""
    doc_id = "doc_temporal_test"
    c1 = _make_clause(
        "c_1", doc_id, 1, "Section 3. Billing",
        "Subscriptions renew automatically unless cancelled at least 30 days prior to the expiration date.",
        primary_category="Payment",
    )
    c2 = _make_clause(
        "c_2", doc_id, 2, "Section 8. Data Retention",
        "Upon termination of your account, we may retain your personal data for a period of 7 years.",
        primary_category="Privacy & Data",
    )

    rels, _ = detect_relationships([c1, c2], document_id=doc_id)
    temporal = [r for r in rels if r.relationship_type == RELATIONSHIP_TYPE_TEMPORAL]
    assert len(temporal) >= 1
    assert "timeline" in temporal[0].rationale or "notice" in temporal[0].rationale or "retention" in temporal[0].rationale


# ---------------------------------------------------------------------------
# 8. POTENTIAL_TENSION Detection
# ---------------------------------------------------------------------------
def test_potential_tension_auto_renewal_vs_cancel_anytime():
    """Verify potential tension between 'cancel at any time' and advance renewal notice."""
    doc_id = "doc_tension_cancel"
    c1 = _make_clause(
        "c_1", doc_id, 1, "Section 2. User Accounts",
        "You may terminate your account at any time through your account settings.",
        primary_category="Account",
    )
    c2 = _make_clause(
        "c_2", doc_id, 2, "Section 3. Auto-Renewal",
        "All paid plans are billed in advance. Your subscription will automatically renew unless you cancel it at least thirty (30) days prior to the expiration date. All subscription fees paid are non-refundable.",
        primary_category="Payment",
    )

    rels, _ = detect_relationships([c1, c2], document_id=doc_id)
    tensions = [r for r in rels if r.relationship_type == RELATIONSHIP_TYPE_POTENTIAL_TENSION]
    assert len(tensions) >= 1
    t = tensions[0]
    assert "cancellation timing" in t.rationale.lower()
    # Ensure neutral language, no legal conclusion
    assert "illegal" not in t.rationale.lower()
    assert "unlawful" not in t.rationale.lower()
    assert "unenforceable" not in t.rationale.lower()


def test_potential_tension_modification_vs_notice_commitment():
    """Verify potential tension between unilateral modification without notice and notice commitments."""
    doc_id = "doc_tension_mod"
    c1 = _make_clause(
        "c_1", doc_id, 1, "Section 4. Unilateral Modifications",
        "We reserve the right, at our sole discretion, to modify or replace these Terms at any time without prior individual notice.",
        primary_category="Changes",
    )
    c2 = _make_clause(
        "c_2", doc_id, 2, "Section 9. Material Change Notice",
        "We will notify users of any material changes via email 30 days prior to changes taking effect.",
        primary_category="Changes",
    )

    rels, _ = detect_relationships([c1, c2], document_id=doc_id)
    tensions = [r for r in rels if r.relationship_type == RELATIONSHIP_TYPE_POTENTIAL_TENSION]
    assert len(tensions) >= 1
    assert "without prior notice" in tensions[0].rationale.lower() or "modify" in tensions[0].rationale.lower()


# ---------------------------------------------------------------------------
# 9 & 10. Document Isolation Guarantee
# ---------------------------------------------------------------------------
def test_document_isolation_same_doc_success():
    """Verify that same-document clauses proceed successfully."""
    doc_id = "doc_isolated_1"
    c1 = _make_clause("c1", doc_id, 1, "Section 1", "Payment subscription details.", "Payment")
    c2 = _make_clause("c2", doc_id, 2, "Section 2", "Payment cancellation details.", "Payment")
    rels, meta = detect_relationships([c1, c2], document_id=doc_id)
    assert meta["candidate_pairs"] >= 1


def test_document_isolation_different_doc_exclusion():
    """Verify that passing clauses from different documents strictly raises ValueError."""
    c1 = _make_clause("c1", "doc_A", 1, "Section 1", "Payment details.", "Payment")
    c2 = _make_clause("c2", "doc_B", 2, "Section 2", "Privacy details.", "Privacy & Data")

    with pytest.raises(ValueError) as exc_info:
        detect_relationships([c1, c2], document_id="doc_A")
    assert "Document isolation violation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 11 & 12. Limits and Confidence Thresholds
# ---------------------------------------------------------------------------
def test_confidence_threshold_filtering():
    """Verify min_confidence filters out lower-confidence relationships."""
    doc_id = "doc_thresh_test"
    c1 = _make_clause("c1", doc_id, 1, "Section 1", "General terms of service.", "General")
    c2 = _make_clause("c2", doc_id, 2, "Section 2", "General user obligations.", "General")

    # High threshold should return 0 relationships
    rels, _ = detect_relationships([c1, c2], document_id=doc_id, min_confidence=0.99)
    assert len(rels) == 0


# ---------------------------------------------------------------------------
# 13 & 14. Verbatim Evidence Preservation & No Fabrication
# ---------------------------------------------------------------------------
def test_verbatim_evidence_preservation():
    """Verify that source and target evidence match original clause texts exactly."""
    doc_id = "doc_evidence_test"
    text1 = "You may terminate your account at any time through your account settings."
    text2 = "All paid plans are billed in advance. Your subscription will automatically renew unless you cancel it at least thirty (30) days prior."
    c1 = _make_clause("c1", doc_id, 1, "Section 1", text1, "Account")
    c2 = _make_clause("c2", doc_id, 2, "Section 2", text2, "Payment")

    rels, _ = detect_relationships([c1, c2], document_id=doc_id)
    assert len(rels) > 0
    for r in rels:
        # Exact verbatim checks
        assert r.evidence.source_text in (text1, text2)
        assert r.evidence.target_text in (text1, text2)
        assert len(r.evidence.trigger) > 0


# ---------------------------------------------------------------------------
# 15. Synthetic ToS Document Integration
# ---------------------------------------------------------------------------
def test_synthetic_tos_document_comprehensive_relationships():
    """Verify relationship detection across a realistic synthetic multi-clause ToS document."""
    doc_id = "synthetic_tos_full"
    clauses = [
        _make_clause(
            "syn_1", doc_id, 1, "1. Acceptance of Terms",
            "By using the service you agree to these Terms. If you do not agree, do not use the service.",
            primary_category="General",
        ),
        _make_clause(
            "syn_2", doc_id, 2, "2. User Accounts and Termination",
            "You may terminate your account at any time through your account settings. However, we reserve the right to suspend or terminate your account immediately, without notice or liability, for any reason whatsoever.",
            primary_category="Account",
        ),
        _make_clause(
            "syn_3", doc_id, 3, "3. Payment, Billing, and Auto-Renewal",
            "All paid plans are billed in advance on a recurring monthly or annual basis. Your subscription will automatically renew at the end of each billing cycle unless you cancel it at least thirty (30) days prior to the expiration date. All subscription fees paid are non-refundable.",
            primary_category="Payment",
        ),
        _make_clause(
            "syn_4", doc_id, 4, "4. Unilateral Modifications",
            "We reserve the right, at our sole discretion, to modify or replace these Terms and modify pricing at any time without prior individual notice.",
            primary_category="Changes",
        ),
        _make_clause(
            "syn_5", doc_id, 5, "5. User Content and License Grant",
            "By submitting content, you grant us a perpetual, irrevocable, worldwide, royalty-free license to use, reproduce, modify, and distribute your content.",
            primary_category="User Content",
        ),
        _make_clause(
            "syn_6", doc_id, 6, "6. Limitation of Liability",
            "To the maximum extent permitted by applicable law, in no event shall the company be liable for any indirect damages, and total liability shall not exceed fifty dollars ($50.00).",
            primary_category="Liability",
        ),
        _make_clause(
            "syn_7", doc_id, 7, "7. Dispute Resolution and Class Action Waiver",
            "Any dispute arising out of these Terms shall be resolved exclusively through binding individual arbitration. You waive any right to participate in a class action.",
            primary_category="Dispute",
        ),
        _make_clause(
            "syn_8", doc_id, 8, "8. Data Retention Following Account Closure",
            "Upon termination of your account, we may retain your personal data, usage logs, and content for a period of up to seven (7) years to comply with regulatory obligations.",
            primary_category="Privacy & Data",
        ),
    ]

    rels, metadata = detect_relationships(clauses, document_id=doc_id)
    assert len(rels) >= 2
    assert metadata["candidate_pairs"] > 0
    assert metadata["evaluated_pairs"] > 0

    detected_types = {r.relationship_type for r in rels}
    # Should identify at least POTENTIAL_TENSION (cancellation vs renewal) and TEMPORAL (closure vs retention / billing)
    assert RELATIONSHIP_TYPE_POTENTIAL_TENSION in detected_types
    assert (RELATIONSHIP_TYPE_TEMPORAL in detected_types) or (RELATIONSHIP_TYPE_QUALIFIES in detected_types)


# ---------------------------------------------------------------------------
# 16. API Endpoint Integration (/api/documents/{document_id}/relationships)
# ---------------------------------------------------------------------------
def test_api_relationships_endpoint_success():
    """Verify POST /api/documents/{document_id}/relationships returns valid structured response."""
    # 1. Ingest text to populate store
    doc_text = (
        "1. Acceptance of Terms\n"
        "By using the service, you agree to these Terms.\n\n"
        "2. Accounts\n"
        "You may terminate your account at any time through your account settings.\n\n"
        "3. Billing and Renewal\n"
        "Your subscription will automatically renew unless cancelled at least 30 days prior. Fees are non-refundable.\n"
    )
    submit_res = client.post("/api/documents/text", json={"text": doc_text, "title": "Test ToS"})
    assert submit_res.status_code == 200
    doc_id = submit_res.json()["document_id"]

    # 2. Call relationships endpoint
    rel_res = client.post(
        f"/api/documents/{doc_id}/relationships",
        json={"max_relationships": 50, "min_confidence": 0.50},
    )
    assert rel_res.status_code == 200
    body = rel_res.json()
    assert body["document_id"] == doc_id
    assert "relationships" in body
    assert "relationship_count" in body
    assert "analysis_metadata" in body
    assert body["analysis_metadata"]["candidate_pairs"] >= 1

    # Check relationship item structure
    if body["relationships"]:
        item = body["relationships"][0]
        assert "relationship_id" in item
        assert "relationship_type" in item
        assert "confidence" in item
        assert "rationale" in item
        assert "evidence" in item
        assert "source_text" in item["evidence"]
        assert "target_text" in item["evidence"]
        assert "trigger" in item["evidence"]


def test_api_relationships_endpoint_404_for_missing_doc():
    """Verify POST /api/documents/{document_id}/relationships returns 404 for unindexed doc."""
    res = client.post("/api/documents/non_existent_doc_id/relationships", json={})
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()
