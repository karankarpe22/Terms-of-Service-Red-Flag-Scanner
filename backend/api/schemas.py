from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(..., description="Service health status", examples=["healthy"])
    service: str = Field(..., description="Service identifier", examples=["tos-red-flag-scanner"])
    version: str = Field(..., description="Service version", examples=["0.1.0"])


class TextSubmitRequest(BaseModel):
    """Payload for submitting raw Terms of Service text."""
    title: Optional[str] = Field(None, description="Optional document title")
    text: str = Field(..., min_length=10, description="Terms of Service raw text content")


class ClauseResponse(BaseModel):
    """Structured clause representation."""
    clause_id: str
    document_id: str
    section_title: Optional[str] = "General"
    clause_index: int
    text: str
    category: Optional[str] = Field(default=None, description="Primary category alias")
    primary_category: Optional[str] = Field(default=None, description="Primary PRD category")
    secondary_categories: List[str] = Field(default_factory=list, description="Secondary PRD categories")
    category_confidence: Optional[float] = Field(default=None, description="Classification model score")
    attention_level: Optional[str] = Field(default=None, description="Attention level")
    attention_reasons: List[str] = Field(default_factory=list, description="Transparent attention indicators")
    source_location: str


class DocumentUploadResponse(BaseModel):
    """Response after document ingestion and clause segmentation."""
    document_id: str
    filename: Optional[str] = None
    source_type: str = "text"
    total_clauses: int
    message: str
    clauses: List[ClauseResponse] = []


class RelationshipResponse(BaseModel):
    """Structured relationship between two clauses (Phase 3)."""
    relationship_id: str
    source_clause_id: str
    target_clause_id: str
    relationship_type: str
    reason: str
    confidence: float


class FindingResponse(BaseModel):
    """Detailed finding for high/medium attention clauses (Phase 3)."""
    finding_id: str
    clause: ClauseResponse
    plain_language_explanation: str
    what_to_check: str
    related_clauses: List[ClauseResponse] = []
    relationships: List[RelationshipResponse] = []


class AnalysisRequest(BaseModel):
    """Request to trigger analysis with selected user priorities (Phase 3)."""
    concern_categories: List[str] = Field(
        default=[],
        description="User-selected concern categories to prioritize in findings",
    )


class AnalysisResponse(BaseModel):
    """Complete document analysis result (Phase 3)."""
    document_id: str
    total_clauses: int
    findings: List[FindingResponse]
    disclaimer: str


class QueryRequest(BaseModel):
    """Request payload for document-grounded Q&A (Phase 4)."""
    question: str = Field(..., min_length=3, description="User question about the document")


class QueryResponse(BaseModel):
    """Response payload for document-grounded Q&A (Phase 4)."""
    question: str
    answer: str
    supporting_clauses: List[ClauseResponse] = []
    abstained: bool = False
    disclaimer: str


class SearchRequest(BaseModel):
    """Payload for semantic search against an indexed document."""
    query: str = Field(..., min_length=2, description="Semantic query text")
    top_k: Optional[int] = Field(default=5, ge=1, le=50, description="Max clauses to return")
    categories: Optional[List[str]] = Field(default=None, description="Optional category filter list")
    threshold: Optional[float] = Field(default=None, description="Optional custom similarity threshold")


class SearchResultItem(BaseModel):
    """Individual search result item."""
    clause_id: str
    document_id: str
    section_title: str
    text: str
    similarity_score: float
    primary_category: Optional[str] = None
    secondary_categories: List[str] = []
    attention_level: Optional[str] = None
    source_location: str = ""
    meets_threshold: bool = True


class SearchResponse(BaseModel):
    """Structured semantic search response."""
    query: str
    document_id: str
    threshold: float
    sufficient_evidence: bool
    total_matches: int
    results: List[SearchResultItem] = []


class RelationshipRequest(BaseModel):
    """Request payload for cross-clause relationship analysis (Phase 5)."""
    max_relationships: Optional[int] = Field(
        default=100, ge=1, le=500, description="Maximum number of relationships to return"
    )
    min_confidence: Optional[float] = Field(
        default=0.50, ge=0.0, le=1.0, description="Minimum algorithmic confidence threshold"
    )


class RelationshipEvidenceResponse(BaseModel):
    """Evidence sub-object for API responses."""
    source_text: str
    target_text: str
    trigger: str


class RelationshipItemResponse(BaseModel):
    """Individual detected relationship item."""
    relationship_id: str
    document_id: str
    source_clause_id: str
    target_clause_id: str
    relationship_type: str
    confidence: float
    rationale: str
    evidence: RelationshipEvidenceResponse
    source_location: str = ""
    target_location: str = ""


class RelationshipAnalysisMetadata(BaseModel):
    """Summary metrics of candidate generation and evaluation."""
    candidate_pairs: int
    evaluated_pairs: int
    rules_used: List[str] = []


class RelationshipResponse(BaseModel):
    """Structured response containing detected cross-clause relationships."""
    document_id: str
    relationships: List[RelationshipItemResponse] = []
    relationship_count: int
    analysis_metadata: RelationshipAnalysisMetadata


class ExplainClauseRequest(BaseModel):
    """Request payload for generating an evidence-grounded clause explanation."""
    clause_id: str = Field(..., min_length=1, description="Unique identifier of the clause to explain")


class ExplainRelationshipRequest(BaseModel):
    """Request payload for explaining an identified cross-clause relationship."""
    relationship_id: str = Field(
        ..., min_length=1, description="Identifier of the detected relationship"
    )


class SynthesizeRequest(BaseModel):
    """Request payload for synthesizing multiple clauses and relationships."""
    clause_ids: Optional[List[str]] = Field(
        default=None, description="Optional list of specific clause IDs to synthesize"
    )
    relationship_ids: Optional[List[str]] = Field(
        default=None, description="Optional list of relationship IDs to include"
    )
    user_concerns: Optional[List[str]] = Field(
        default=None, description="Optional user concern categories to prioritize"
    )


class EvidenceReferenceItem(BaseModel):
    """Citation and verbatim quoted text in explanation response."""
    clause_id: str
    source_location: str
    quoted_text: str


class ExplanationData(BaseModel):
    """Core explanation content generated by Gemini or Mock service."""
    summary: str
    attention_explanation: str = ""
    relationship_explanation: str = ""
    key_points: List[str] = []
    evidence_references: List[EvidenceReferenceItem] = []
    evidence_sufficient: bool = True
    uncertainty: str = ""
    disclaimer: str = "Informational document analysis only; not legal advice."


class ExplanationResponse(BaseModel):
    """API response envelope for evidence-grounded explanations."""
    status: str = Field(
        ...,
        description="'success', 'validation_failed', or 'llm_unavailable'",
    )
    document_id: str
    operation: str
    explanation: Optional[ExplanationData] = None
    message: Optional[str] = None
    disclaimer: str = "Informational document analysis only; not legal advice."


class QuestionRequest(BaseModel):
    """Request payload for evidence-grounded document Q&A (Phase 8)."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question about the document",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of relevant clauses to retrieve (safe upper bound: 10)",
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional similarity threshold override for retrieval",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Question cannot be empty or contain only whitespace.")
        if len(trimmed) > 1000:
            raise ValueError("Question exceeds maximum length of 1000 characters.")
        return trimmed


class QuestionSourceResponse(BaseModel):
    """Source clause citation and verified verbatim excerpt in Q&A response."""
    clause_id: str
    section_title: str
    source_location: str
    quoted_text: str


class QuestionResponse(BaseModel):
    """API response envelope for document Q&A."""
    status: str = Field(
        ...,
        description="'success', 'insufficient_evidence', 'validation_failed', or 'llm_unavailable'",
    )
    document_id: str
    question: str
    answer: Optional[str] = None
    evidence_sufficient: bool = True
    confidence: float = 0.0
    sources: List[QuestionSourceResponse] = []
    uncertainty: str = ""
    message: Optional[str] = None
    disclaimer: str = "Informational document analysis only; not legal advice."

