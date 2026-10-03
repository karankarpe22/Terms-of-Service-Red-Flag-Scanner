"""FAISS vector store and similarity retrieval module.

Vector storage uses FAISS (faiss-cpu) as the only vector store for the MVP.
Guarantees 100% document isolation by maintaining dedicated, isolated FAISS indices per document.
Supports configurable similarity thresholding, top-k retrieval, and user-concern category filtering.

Non-negotiable architectural rule:
FAISS is strictly a retrieval mechanism for semantically relevant clauses.
It does not determine legality, enforceability, or compliance.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import re
import faiss
import numpy as np

from backend.config import settings
from backend.document.models import Clause
from backend.retrieval.embedder import get_embedding_service, EmbeddingService

# Substantive query token filtering for lexical boosting in hybrid retrieval
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "related", "relating",
    "same", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "yourself",
}

# Domain synonym expansions for common consumer contract and ToS inquiries
CONTRACT_SYNONYM_MAP: Dict[str, List[str]] = {
    # Security, data protection, privacy
    "safe": ["secure", "security", "protect", "protection", "safeguard", "confidential"],
    "safety": ["secure", "security", "protect", "protection", "safeguard"],
    "secure": ["safe", "security", "protect", "safeguard", "confidential"],
    "security": ["safe", "secure", "protect", "protection", "safeguard"],
    "protect": ["safe", "secure", "security", "safeguard"],
    "protection": ["safe", "secure", "security", "safeguard"],
    "data": ["information", "privacy", "personal", "content", "records"],
    "privacy": ["data", "information", "personal", "confidential"],
    "private": ["privacy", "data", "confidential"],
    # Financial, payments, refunds
    "money": ["refund", "fee", "payment", "charge", "price", "billing", "cost"],
    "refund": ["money", "reimburse", "return", "payment", "charge", "fee"],
    "refunds": ["money", "reimburse", "return", "payment", "charge", "fee"],
    "pay": ["payment", "fee", "charge", "billing", "subscription", "cost"],
    "payment": ["fee", "charge", "billing", "subscription", "price", "refund"],
    # Termination, cancellation, account deletion
    "cancel": ["terminate", "discontinue", "end", "stop", "close", "cancellation"],
    "cancellation": ["cancel", "terminate", "discontinue", "end", "stop"],
    "delete": ["remove", "erase", "purge", "retention", "deletion"],
    "deletion": ["delete", "remove", "erase", "purge", "retention"],
    # Legal disputes, litigation, arbitration
    "sue": ["arbitration", "dispute", "court", "jury", "class action", "litigation"],
    "lawsuit": ["arbitration", "dispute", "court", "jury", "class action", "litigation"],
    "court": ["arbitration", "dispute", "jury", "class action", "litigation"],
}


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class DocumentNotIndexedError(VectorStoreError):
    """Raised when querying a document that has not been indexed in FAISS."""
    pass


@dataclass
class ClauseSearchResult:
    """Individual retrieved clause match from FAISS vector search."""
    clause_id: str
    document_id: str
    section_title: str
    text: str
    similarity_score: float
    primary_category: Optional[str] = None
    secondary_categories: List[str] = field(default_factory=list)
    attention_level: Optional[str] = None
    source_location: str = ""
    meets_threshold: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "document_id": self.document_id,
            "section_title": self.section_title,
            "text": self.text,
            "similarity_score": self.similarity_score,
            "primary_category": self.primary_category,
            "secondary_categories": self.secondary_categories,
            "attention_level": self.attention_level,
            "source_location": self.source_location,
            "meets_threshold": self.meets_threshold,
        }


@dataclass
class DocumentRetrievalResult:
    """Structured result of a semantic search over an indexed document."""
    query: str
    document_id: str
    threshold: float
    sufficient_evidence: bool
    total_matches: int
    results: List[ClauseSearchResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "document_id": self.document_id,
            "threshold": self.threshold,
            "sufficient_evidence": self.sufficient_evidence,
            "total_matches": self.total_matches,
            "results": [r.to_dict() for r in self.results],
        }


class FAISSDocumentIndex:
    """Dedicated, isolated FAISS index and clause metadata mapping for a single document."""

    def __init__(self, document_id: str, dimension: int = 384):
        self.document_id = document_id
        self.dimension = dimension
        # Inner Product on L2-normalized vectors calculates exact Cosine Similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.clauses: List[Clause] = []
        self.clause_id_to_pos: Dict[str, int] = {}

    def add_clauses(self, clauses: List[Clause], embeddings: np.ndarray) -> None:
        """Add clause embeddings to the index and record positional metadata."""
        if not clauses or len(clauses) == 0:
            return

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        # Ensure unit normalization
        faiss.normalize_L2(embeddings)

        start_pos = len(self.clauses)
        self.index.add(embeddings)

        for offset, clause in enumerate(clauses):
            pos = start_pos + offset
            self.clauses.append(clause)
            self.clause_id_to_pos[clause.clause_id] = pos

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        categories: Optional[List[str]] = None,
        threshold: float = 0.35,
        query_text: Optional[str] = None,
    ) -> List[ClauseSearchResult]:
        """Perform hybrid cosine similarity & lexical search against this document's indexed clauses."""
        if self.index.ntotal == 0:
            return []

        query_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)
        faiss.normalize_L2(query_vec)

        # Retrieve generous candidates so hybrid reranking can promote exact keyword matches
        candidate_k = min(self.index.ntotal, max(top_k * 4, 15))
        scores, indices = self.index.search(query_vec, candidate_k)

        # Extract substantive query tokens for lexical matching (with domain synonym expansion)
        query_tokens: List[str] = []
        synonym_tokens: List[str] = []
        if query_text:
            raw_tokens = re.findall(r"\b[a-zA-Z]{2,}\b", query_text.lower())
            query_tokens = [t for t in raw_tokens if t not in STOP_WORDS]
            expanded: Set[str] = set()
            for t in query_tokens:
                if t in CONTRACT_SYNONYM_MAP:
                    expanded.update(CONTRACT_SYNONYM_MAP[t])
            synonym_tokens = [s for s in expanded if s not in query_tokens]

        results: List[ClauseSearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.clauses):
                continue

            clause = self.clauses[idx]

            # Category filtering / user concern prioritization
            if categories:
                cat_match = (clause.primary_category in categories) or any(
                    sec in categories for sec in clause.secondary_categories
                )
                if not cat_match:
                    continue

            base_score = float(score)
            lexical_boost = 0.0
            if query_tokens or synonym_tokens:
                title_lower = (clause.section_title or "").lower()
                text_lower = clause.text.lower()

                # Direct token matches (highest weight)
                title_direct = sum(
                    1 for t in query_tokens if t in title_lower or any(w.startswith(t) for w in title_lower.split())
                )
                body_direct = sum(
                    1 for t in query_tokens if t in text_lower or any(w.startswith(t) for w in text_lower.split())
                )

                # Synonym token matches (supporting weight for domain terms)
                title_syn = sum(
                    1 for s in synonym_tokens if s in title_lower or any(w.startswith(s) for w in title_lower.split())
                )
                body_syn = sum(
                    1 for s in synonym_tokens if s in text_lower or any(w.startswith(s) for w in text_lower.split())
                )

                if title_direct > 0:
                    lexical_boost += 0.08 * min(title_direct, 2)
                elif title_syn > 0:
                    lexical_boost += 0.04 * min(title_syn, 2)

                if body_direct > 0:
                    lexical_boost += 0.04 * min(body_direct, 2)
                elif body_syn > 0:
                    lexical_boost += 0.02 * min(body_syn, 2)

                lexical_boost = min(lexical_boost, 0.15)

            sim_score = round(min(1.0, base_score + lexical_boost), 4)
            meets = bool(sim_score >= threshold)

            results.append(
                ClauseSearchResult(
                    clause_id=clause.clause_id,
                    document_id=self.document_id,
                    section_title=clause.section_title or "General",
                    text=clause.text,
                    similarity_score=sim_score,
                    primary_category=clause.primary_category,
                    secondary_categories=clause.secondary_categories,
                    attention_level=clause.attention_level,
                    source_location=clause.source_location.to_display_string(),
                    meets_threshold=meets,
                )
            )

        # Sort candidate results by similarity score descending so boosted matches rank at top
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results[:top_k]


class FAISSVectorStore:
    """Registry managing FAISS vector indexes across all ingested documents."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        default_threshold: Optional[float] = None,
        default_top_k: Optional[int] = None,
    ):
        self.embedder = embedding_service or get_embedding_service()
        self.default_threshold = (
            default_threshold
            if default_threshold is not None
            else settings.RETRIEVAL_SIMILARITY_THRESHOLD
        )
        self.default_top_k = (
            default_top_k if default_top_k is not None else settings.RETRIEVAL_TOP_K
        )
        self._indices: Dict[str, FAISSDocumentIndex] = {}

    def index_document(
        self,
        document_id: str,
        clauses: List[Clause],
        embeddings: Optional[np.ndarray] = None,
    ) -> int:
        """Build and register an isolated FAISS index for a document.
        
        Returns:
            Number of clauses successfully indexed into the document vector index.
        """
        if not clauses:
            return 0

        if embeddings is None:
            texts = [c.text for c in clauses]
            embeddings = self.embedder.encode(texts)

        doc_index = FAISSDocumentIndex(document_id=document_id, dimension=embeddings.shape[1])
        doc_index.add_clauses(clauses, embeddings)
        self._indices[document_id] = doc_index
        return len(clauses)

    def search(
        self,
        query: str,
        document_id: str,
        top_k: Optional[int] = None,
        categories: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ) -> DocumentRetrievalResult:
        """Search clauses of a specific document with semantic similarity and threshold filtering."""
        if document_id not in self._indices:
            raise DocumentNotIndexedError(
                f"Document '{document_id}' has not been indexed in the vector store."
            )

        doc_index = self._indices[document_id]
        k = top_k or self.default_top_k
        t = threshold if threshold is not None else self.default_threshold

        query_emb = self.embedder.encode(query)
        results = doc_index.search(
            query_embedding=query_emb,
            top_k=k,
            categories=categories,
            threshold=t,
            query_text=query,
        )

        sufficient_evidence = any(r.meets_threshold for r in results)

        return DocumentRetrievalResult(
            query=query,
            document_id=document_id,
            threshold=t,
            sufficient_evidence=sufficient_evidence,
            total_matches=len(results),
            results=results,
        )

    def has_document(self, document_id: str) -> bool:
        """Check if a document has been indexed in the vector store."""
        return document_id in self._indices

    def clear(self) -> None:
        """Clear all indexed documents (used for test resets)."""
        self._indices.clear()


_VECTOR_STORE_INSTANCE: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    """Return or initialize the singleton FAISSVectorStore."""
    global _VECTOR_STORE_INSTANCE
    if _VECTOR_STORE_INSTANCE is None:
        _VECTOR_STORE_INSTANCE = FAISSVectorStore()
    return _VECTOR_STORE_INSTANCE
