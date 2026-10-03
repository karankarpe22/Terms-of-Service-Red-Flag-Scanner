import numpy as np
import pytest
from backend.config import settings
from backend.document.models import Clause, SourceLocation
from backend.retrieval.embedder import get_embedding_service, EmbeddingService
from backend.retrieval.vector_store import (
    FAISSVectorStore,
    DocumentNotIndexedError,
    get_vector_store,
)


def create_clause(
    clause_id: str,
    doc_id: str,
    text: str,
    category: str = "General",
    attention_level: str = "Informational",
) -> Clause:
    return Clause(
        clause_id=clause_id,
        document_id=doc_id,
        clause_index=1,
        section_title=f"Section {clause_id}",
        text=text,
        category=category,
        primary_category=category,
        secondary_categories=[],
        attention_level=attention_level,
        source_location=SourceLocation(source_type="text"),
    )


# ==============================================================================
# 1. Embedding Tests
# ==============================================================================

def test_embedding_model_loads_and_is_singleton():
    """Verify embedding service loads model and reuses the singleton instance."""
    s1 = get_embedding_service()
    s2 = get_embedding_service()
    assert s1 is s2
    assert isinstance(s1, EmbeddingService)


def test_embedding_dimensions_and_normalization():
    """Verify embeddings produce 384 dimensions and are unit normalized (L2 norm ≈ 1.0)."""
    service = get_embedding_service()
    text = "Subscriptions automatically renew every billing cycle."
    emb = service.encode(text)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)
    # Check L2 norm is approximately 1.0
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4


def test_embedding_batch_consistency():
    """Verify batch encoding produces same vector as individual encoding."""
    service = get_embedding_service()
    t1 = "All fees are non-refundable."
    t2 = "We collect your email address."

    batch_embs = service.encode([t1, t2])
    single_emb1 = service.encode(t1)

    assert batch_embs.shape == (2, 384)
    # Cosine similarity between same sentence encoded in batch vs single should be 1.0
    sim = float(np.dot(batch_embs[0], single_emb1))
    assert abs(sim - 1.0) < 1e-5


# ==============================================================================
# 2. FAISS Vector Store Tests
# ==============================================================================

def test_faiss_indexing_and_metadata_mapping():
    """Verify FAISS index creation, clause addition, and metadata preservation."""
    store = FAISSVectorStore()
    doc_id = "doc_test_meta"
    c1 = create_clause("c1", doc_id, "Paid plans renew automatically each month.", category="Payment")
    c2 = create_clause("c2", doc_id, "We use tracking cookies on our website.", category="Privacy & Data")

    store.index_document(doc_id, [c1, c2])

    assert store.has_document(doc_id)
    doc_index = store._indices[doc_id]
    assert doc_index.index.ntotal == 2
    assert len(doc_index.clauses) == 2
    assert doc_index.clauses[0].clause_id == "c1"
    assert doc_index.clauses[1].clause_id == "c2"


def test_faiss_semantic_search_and_similarity_scores():
    """Verify semantic search returns relevant clauses with accurate similarity scores."""
    store = FAISSVectorStore()
    doc_id = "doc_test_search"
    c1 = create_clause("c1", doc_id, "Subscriptions renew automatically without notice.", category="Payment")
    c2 = create_clause("c2", doc_id, "Arbitration is mandatory for all legal disputes.", category="Dispute")

    store.index_document(doc_id, [c1, c2])

    result = store.search("Does my subscription auto renew?", doc_id, top_k=2)

    assert result.document_id == doc_id
    assert len(result.results) >= 1
    top_result = result.results[0]
    assert top_result.clause_id == "c1"
    assert top_result.primary_category == "Payment"
    assert 0.0 <= top_result.similarity_score <= 1.0
    assert top_result.similarity_score > 0.40


def test_faiss_top_k_retrieval():
    """Verify top_k parameter strictly bounds the number of returned results."""
    store = FAISSVectorStore()
    doc_id = "doc_test_topk"
    clauses = [
        create_clause(f"c_{i}", doc_id, f"Payment condition number {i} regarding fees.")
        for i in range(10)
    ]
    store.index_document(doc_id, clauses)

    result = store.search("Payment condition", doc_id, top_k=3)
    assert len(result.results) == 3


# ==============================================================================
# 3. Document Isolation Tests
# ==============================================================================

def test_document_isolation_guarantee():
    """Verify that a query against Document A NEVER returns clauses belonging to Document B."""
    store = FAISSVectorStore()

    # Document A contains Payment clauses
    doc_a = "doc_payment_only"
    c_a1 = create_clause("c_a1", doc_a, "Monthly subscription fee is ten dollars.", category="Payment")
    c_a2 = create_clause("c_a2", doc_a, "Cancellation must occur 30 days prior.", category="Payment")
    store.index_document(doc_a, [c_a1, c_a2])

    # Document B contains Privacy clauses
    doc_b = "doc_privacy_only"
    c_b1 = create_clause("c_b1", doc_b, "We collect biometric data and geolocation logs.", category="Privacy & Data")
    store.index_document(doc_b, [c_b1])

    # Query Document A with a privacy query
    result_a = store.search("Do you track biometric geolocation data?", doc_a, top_k=5)

    # Result from Document A MUST NOT contain any clauses from Document B
    for res in result_a.results:
        assert res.document_id == doc_a
        assert res.clause_id != "c_b1"
        assert "biometric" not in res.text

    # Query Document B with the same query -> returns Document B's clause
    result_b = store.search("Do you track biometric geolocation data?", doc_b, top_k=5)
    assert any(r.clause_id == "c_b1" for r in result_b.results)


# ==============================================================================
# 4. Threshold & Sufficient Evidence Tests
# ==============================================================================

def test_similarity_threshold_and_sufficient_evidence():
    """Verify results above threshold are marked meets_threshold=True, and below threshold as False."""
    store = FAISSVectorStore()
    doc_id = "doc_thresh"
    c1 = create_clause("c1", doc_id, "Automatic subscription renewal every billing month.", category="Payment")
    store.index_document(doc_id, [c1])

    # 1. Query relevant to the clause with standard threshold
    res_relevant = store.search("Does it auto renew?", doc_id, threshold=0.40)
    assert res_relevant.sufficient_evidence is True
    assert res_relevant.results[0].meets_threshold is True

    # 2. Completely unrelated query (e.g. pet policy in software ToS)
    res_irrelevant = store.search("What is the refund policy for emotional support dogs?", doc_id, threshold=0.75)
    assert res_irrelevant.sufficient_evidence is False
    assert res_irrelevant.results[0].meets_threshold is False


# ==============================================================================
# 5. Category Filtering / User-Concern Prioritization
# ==============================================================================

def test_user_concern_category_filtering():
    """Verify that specifying concern categories filters results strictly to those categories."""
    store = FAISSVectorStore()
    doc_id = "doc_cat_filter"
    c_pay = create_clause("c_pay", doc_id, "Monthly renewal and cancellation terms apply.", category="Payment")
    c_priv = create_clause("c_priv", doc_id, "We share your information with marketing partners.", category="Privacy & Data")
    c_disp = create_clause("c_disp", doc_id, "Arbitration is the exclusive remedy for disputes.", category="Dispute")

    store.index_document(doc_id, [c_pay, c_priv, c_disp])

    # Search with category filter: ONLY Payment
    res_filtered = store.search(
        query="terms apply to your use",
        document_id=doc_id,
        categories=["Payment"],
        top_k=5,
    )

    for r in res_filtered.results:
        assert r.primary_category == "Payment"
        assert r.clause_id == "c_pay"


# ==============================================================================
# 6. Error Handling & Edge Cases
# ==============================================================================

def test_search_unindexed_document_raises_error():
    """Verify searching for a document that has not been indexed raises DocumentNotIndexedError."""
    store = FAISSVectorStore()
    with pytest.raises(DocumentNotIndexedError):
        store.search("test query", "doc_not_exists")


def test_hybrid_search_lexical_boost_on_short_queries():
    """Verify short conversational queries like 'payment related' match payment clauses with boosted scores."""
    store = FAISSVectorStore()
    doc_id = "doc_phonepe_short_q"
    c1 = create_clause("c1", doc_id, "By using PhonePe services, you agree to these Terms.")
    c2 = create_clause("c2", doc_id, "You must be 18 to register for a PhonePe account.")
    c3 = create_clause("c3", doc_id, "PhonePe provides payment aggregation and UPI payment services.", category="Payment")
    c3.section_title = "3. Payment Services & Wallet"
    c4 = create_clause("c4", doc_id, "We may charge convenience fees on certain payment transactions.", category="Payment")
    c4.section_title = "4. Charges and Fees"

    store.index_document(doc_id, [c1, c2, c3, c4])

    # Search with short conversational query
    res = store.search(query="payment related", document_id=doc_id, top_k=2)
    assert res.sufficient_evidence is True
    assert len(res.results) >= 1
    # Top result should be Clause 3 or 4 with meets_threshold=True
    assert res.results[0].clause_id in ("c3", "c4")
    assert res.results[0].meets_threshold is True
    assert res.results[0].similarity_score >= 0.45

