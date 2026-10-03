from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File
from backend.api.schemas import (
    HealthResponse,
    DocumentUploadResponse,
    TextSubmitRequest,
    ClauseResponse,
    AnalysisRequest,
    AnalysisResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    RelationshipRequest,
    RelationshipResponse,
    RelationshipItemResponse,
    RelationshipEvidenceResponse,
    RelationshipAnalysisMetadata,
    ExplainClauseRequest,
    ExplainRelationshipRequest,
    SynthesizeRequest,
    ExplanationResponse,
    ExplanationData,
    EvidenceReferenceItem,
    QuestionRequest,
    QuestionResponse,
    QuestionSourceResponse,
)

from backend.document.extractor import (
    extract_from_pdf,
    extract_from_text,
    InvalidDocumentError,
    UnsupportedDocumentError,
)
from backend.document.segmenter import segment_document
from backend.document.store import save_document, get_document
from backend.analysis.classifier import classify_clauses
from backend.analysis.attention import assign_attention_levels
from backend.analysis.relationships import detect_relationships
from backend.retrieval.vector_store import get_vector_store, DocumentNotIndexedError
from backend.llm import get_llm_service, LLMUnavailableError, LLMServiceError, ExplanationOutput
from backend.core.constants import SYSTEM_DISCLAIMER

router = APIRouter(prefix="/api", tags=["API"])


def _to_clause_response(c) -> ClauseResponse:
    """Helper to convert a Clause domain model into a ClauseResponse schema."""
    return ClauseResponse(
        clause_id=c.clause_id,
        document_id=c.document_id,
        section_title=c.section_title or "General",
        clause_index=c.clause_index,
        text=c.text,
        category=c.category,
        primary_category=c.primary_category,
        secondary_categories=c.secondary_categories,
        category_confidence=c.category_confidence,
        attention_level=c.attention_level,
        attention_reasons=c.attention_reasons,
        source_location=c.source_location.to_display_string(),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check endpoint confirming that backend service is running."""
    return HealthResponse(
        status="healthy",
        service="tos-red-flag-scanner",
        version="0.1.0",
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Upload and process a machine-readable PDF document.

    Extracts text, segments clauses, runs classification and attention detection,
    and stores the document in memory.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Only machine-readable PDF files are accepted.",
        )

    try:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        document = extract_from_pdf(content, filename=file.filename)
        clauses = segment_document(document)
        classify_clauses(clauses)
        assign_attention_levels(clauses)
        get_vector_store().index_document(document.document_id, clauses)
        save_document(document)

        clause_responses = [_to_clause_response(c) for c in clauses]

        return DocumentUploadResponse(
            document_id=document.document_id,
            filename=document.filename,
            source_type=document.source_type,
            total_clauses=len(clauses),
            message="Document extracted, segmented, and indexed successfully.",
            clauses=clause_responses,
        )

    except HTTPException:
        raise
    except UnsupportedDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(exc)}")


@router.post("/documents/text", response_model=DocumentUploadResponse)
async def submit_text(request: TextSubmitRequest) -> DocumentUploadResponse:
    """Submit raw Terms of Service text.

    Cleans text, segments clauses, runs classification and attention detection,
    builds FAISS vector index, and stores the document in memory.
    """
    try:
        document = extract_from_text(request.text, title=request.title)
        clauses = segment_document(document)
        classify_clauses(clauses)
        assign_attention_levels(clauses)
        get_vector_store().index_document(document.document_id, clauses)
        save_document(document)

        clause_responses = [_to_clause_response(c) for c in clauses]

        return DocumentUploadResponse(
            document_id=document.document_id,
            filename=document.filename,
            source_type=document.source_type,
            total_clauses=len(clauses),
            message="Raw text processed, segmented, and indexed successfully.",
            clauses=clause_responses,
        )

    except HTTPException:
        raise
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(exc)}")


@router.get("/documents/{document_id}", response_model=DocumentUploadResponse)
async def get_document_by_id(document_id: str) -> DocumentUploadResponse:
    """Retrieve processed document and its segmented clauses by document ID."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{document_id}' not found.",
        )

    clause_responses = [_to_clause_response(c) for c in doc.clauses]

    return DocumentUploadResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        source_type=doc.source_type,
        total_clauses=len(doc.clauses),
        message="Document retrieved successfully.",
        clauses=clause_responses,
    )


@router.post("/documents/{document_id}/search", response_model=SearchResponse)
async def search_document(document_id: str, request: SearchRequest) -> SearchResponse:
    """Perform semantic search across an indexed document's clauses using FAISS."""
    store = get_vector_store()
    if not store.has_document(document_id):
        doc = get_document(document_id)
        if not doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID '{document_id}' not found.",
            )
        store.index_document(document_id, doc.clauses)

    try:
        retrieval_res = store.search(
            query=request.query,
            document_id=document_id,
            top_k=request.top_k,
            categories=request.categories,
            threshold=request.threshold,
        )

        return SearchResponse(
            query=retrieval_res.query,
            document_id=retrieval_res.document_id,
            threshold=retrieval_res.threshold,
            sufficient_evidence=retrieval_res.sufficient_evidence,
            total_matches=retrieval_res.total_matches,
            results=[SearchResultItem(**r.to_dict()) for r in retrieval_res.results],
        )

    except DocumentNotIndexedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(exc)}")


@router.post("/documents/{document_id}/relationships", response_model=RelationshipResponse)
async def get_document_relationships(
    document_id: str,
    request: Optional[RelationshipRequest] = None,
) -> RelationshipResponse:
    """Analyze and retrieve cross-clause relationships for an ingested document."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{document_id}' not found.",
        )

    req = request or RelationshipRequest()
    try:
        relationships, metadata = detect_relationships(
            clauses=doc.clauses,
            document_id=document_id,
            max_relationships=req.max_relationships,
            min_confidence=req.min_confidence,
        )

        rel_items = [
            RelationshipItemResponse(
                relationship_id=r.relationship_id,
                document_id=r.document_id,
                source_clause_id=r.source_clause_id,
                target_clause_id=r.target_clause_id,
                relationship_type=r.relationship_type,
                confidence=r.confidence,
                rationale=r.rationale,
                evidence=RelationshipEvidenceResponse(
                    source_text=r.evidence.source_text,
                    target_text=r.evidence.target_text,
                    trigger=r.evidence.trigger,
                ),
                source_location=r.source_location,
                target_location=r.target_location,
            )
            for r in relationships
        ]

        return RelationshipResponse(
            document_id=document_id,
            relationships=rel_items,
            relationship_count=len(rel_items),
            analysis_metadata=RelationshipAnalysisMetadata(**metadata),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Relationship analysis error: {str(exc)}")


def _to_explanation_data(output: ExplanationOutput) -> ExplanationData:
    """Map internal ExplanationOutput to API schema ExplanationData."""
    return ExplanationData(
        summary=output.summary,
        attention_explanation=output.attention_explanation,
        relationship_explanation=output.relationship_explanation,
        key_points=output.key_points,
        evidence_references=[
            EvidenceReferenceItem(
                clause_id=ref.clause_id,
                source_location=ref.source_location,
                quoted_text=ref.quoted_text,
            )
            for ref in output.evidence_references
        ],
        evidence_sufficient=output.evidence_sufficient,
        uncertainty=output.uncertainty,
        disclaimer=output.disclaimer or SYSTEM_DISCLAIMER,
    )


@router.post("/documents/{document_id}/explain-clause", response_model=ExplanationResponse)
async def explain_clause(
    document_id: str,
    request: ExplainClauseRequest,
) -> ExplanationResponse:
    """Generate an evidence-grounded explanation for a single clause."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found.")

    target_clause = next((c for c in doc.clauses if c.clause_id == request.clause_id), None)
    if not target_clause:
        raise HTTPException(
            status_code=404,
            detail=f"Clause '{request.clause_id}' not found in document '{document_id}'.",
        )

    try:
        service = get_llm_service()
        output = service.explain_clause(target_clause)
        status = (
            "validation_failed"
            if not output.evidence_sufficient and "validation failed" in (output.uncertainty or "").lower()
            else "success"
        )
        return ExplanationResponse(
            status=status,
            document_id=document_id,
            operation="explain-clause",
            explanation=_to_explanation_data(output),
        )
    except (LLMUnavailableError, LLMServiceError) as exc:
        return ExplanationResponse(
            status="llm_unavailable",
            document_id=document_id,
            operation="explain-clause",
            explanation=None,
            message=str(exc),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Clause explanation error: {str(exc)}")


@router.post("/documents/{document_id}/explain-relationship", response_model=ExplanationResponse)
async def explain_relationship(
    document_id: str,
    request: ExplainRelationshipRequest,
) -> ExplanationResponse:
    """Generate an evidence-grounded explanation for a cross-clause relationship."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found.")

    relationships, _ = detect_relationships(clauses=doc.clauses, document_id=document_id)
    rel = next((r for r in relationships if r.relationship_id == request.relationship_id), None)
    if not rel:
        raise HTTPException(
            status_code=404,
            detail=f"Relationship '{request.relationship_id}' not found in document '{document_id}'.",
        )

    src_clause = next((c for c in doc.clauses if c.clause_id == rel.source_clause_id), None)
    tgt_clause = next((c for c in doc.clauses if c.clause_id == rel.target_clause_id), None)
    if not src_clause or not tgt_clause:
        raise HTTPException(
            status_code=404,
            detail=f"Source or target clause for relationship '{request.relationship_id}' not found.",
        )

    try:
        service = get_llm_service()
        output = service.explain_relationship(rel, src_clause, tgt_clause)
        status = (
            "validation_failed"
            if not output.evidence_sufficient and "validation failed" in (output.uncertainty or "").lower()
            else "success"
        )
        return ExplanationResponse(
            status=status,
            document_id=document_id,
            operation="explain-relationship",
            explanation=_to_explanation_data(output),
        )
    except (LLMUnavailableError, LLMServiceError) as exc:
        return ExplanationResponse(
            status="llm_unavailable",
            document_id=document_id,
            operation="explain-relationship",
            explanation=None,
            message=str(exc),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Relationship explanation error: {str(exc)}")


@router.post("/documents/{document_id}/synthesize", response_model=ExplanationResponse)
async def synthesize_document(
    document_id: str,
    request: Optional[SynthesizeRequest] = None,
) -> ExplanationResponse:
    """Synthesize clauses and relationships into an evidence-grounded summary."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found.")

    req = request or SynthesizeRequest()
    clauses = doc.clauses
    if req.clause_ids:
        clause_id_set = set(req.clause_ids)
        clauses = [c for c in doc.clauses if c.clause_id in clause_id_set]
        if not clauses:
            raise HTTPException(
                status_code=404,
                detail=f"None of the requested clause IDs were found in document '{document_id}'.",
            )

    relationships, _ = detect_relationships(clauses=doc.clauses, document_id=document_id)
    if req.relationship_ids:
        rel_id_set = set(req.relationship_ids)
        relationships = [r for r in relationships if r.relationship_id in rel_id_set]

    try:
        service = get_llm_service()
        output = service.synthesize_clauses(
            clauses=clauses,
            relationships=relationships,
            user_concerns=req.user_concerns,
        )
        status = (
            "validation_failed"
            if not output.evidence_sufficient and "validation failed" in (output.uncertainty or "").lower()
            else "success"
        )
        return ExplanationResponse(
            status=status,
            document_id=document_id,
            operation="synthesize",
            explanation=_to_explanation_data(output),
        )
    except (LLMUnavailableError, LLMServiceError) as exc:
        return ExplanationResponse(
            status="llm_unavailable",
            document_id=document_id,
            operation="synthesize",
            explanation=None,
            message=str(exc),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document synthesis error: {str(exc)}")



@router.post("/documents/{document_id}/analyze", response_model=AnalysisResponse)
async def analyze_document(document_id: str, request: AnalysisRequest):
    """Placeholder endpoint for running analysis on a document."""
    raise HTTPException(
        status_code=501,
        detail="Analysis orchestration will be implemented in Phase 5.",
    )


@router.post("/documents/{document_id}/ask", response_model=QuestionResponse)
async def ask_document(
    document_id: str,
    request: QuestionRequest,
) -> QuestionResponse:
    """Answer user questions about the analyzed Terms using retrieved evidence clauses (Phase 8)."""
    doc = get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID '{document_id}' not found.",
        )

    # 1. Semantic Retrieval using Document-Isolated FAISS index
    try:
        vector_store = get_vector_store()
        retrieval_res = vector_store.search(
            document_id=document_id,
            query=request.question,
            top_k=request.top_k or 5,
            threshold=request.threshold,
        )
    except DocumentNotIndexedError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(exc)}")

    # 2. Strict Evidence Sufficiency Check
    candidate_results = [r for r in retrieval_res.results if r.meets_threshold]
    if not retrieval_res.sufficient_evidence or not candidate_results:
        return QuestionResponse(
            status="insufficient_evidence",
            document_id=document_id,
            question=request.question,
            answer=None,
            evidence_sufficient=False,
            confidence=0.0,
            sources=[],
            uncertainty="The supplied Terms do not contain enough evidence to answer this question.",
            message="The supplied Terms do not contain enough evidence to answer this question.",
            disclaimer=SYSTEM_DISCLAIMER,
        )

    # 3. Prepare structured evidence
    clause_map = {c.clause_id: c for c in doc.clauses}
    matched_clauses = [clause_map[r.clause_id] for r in candidate_results if r.clause_id in clause_map]
    score_map = {r.clause_id: r.similarity_score for r in candidate_results}

    if not matched_clauses:
        return QuestionResponse(
            status="insufficient_evidence",
            document_id=document_id,
            question=request.question,
            answer=None,
            evidence_sufficient=False,
            confidence=0.0,
            sources=[],
            uncertainty="The supplied Terms do not contain enough evidence to answer this question.",
            message="The supplied Terms do not contain enough evidence to answer this question.",
            disclaimer=SYSTEM_DISCLAIMER,
        )

    # 4. Generate & Validate Answer with Gemini or Mock
    try:
        llm_service = get_llm_service()
        output = llm_service.answer_question(
            question=request.question,
            clauses=matched_clauses,
            scores=score_map,
        )

        status = "success"
        if not output.evidence_sufficient and "validation failed" in (output.uncertainty or "").lower():
            status = "validation_failed"
        elif not output.evidence_sufficient:
            status = "insufficient_evidence"

        return QuestionResponse(
            status=status,
            document_id=document_id,
            question=request.question,
            answer=output.answer if output.evidence_sufficient else None,
            evidence_sufficient=output.evidence_sufficient,
            confidence=output.confidence if output.evidence_sufficient else 0.0,
            sources=[
                QuestionSourceResponse(
                    clause_id=s.clause_id,
                    section_title=s.section_title,
                    source_location=s.source_location,
                    quoted_text=s.quoted_text,
                )
                for s in output.sources
            ] if output.evidence_sufficient else [],
            uncertainty=output.uncertainty,
            message=None if output.evidence_sufficient else (output.uncertainty or "The supplied Terms do not contain enough evidence to answer this question."),
            disclaimer=output.disclaimer or SYSTEM_DISCLAIMER,
        )

    except (LLMUnavailableError, LLMServiceError) as exc:
        # LLM unavailable or service error: graceful degradation returning retrieved sources
        return QuestionResponse(
            status="llm_unavailable",
            document_id=document_id,
            question=request.question,
            answer=None,
            evidence_sufficient=True,
            confidence=0.0,
            sources=[
                QuestionSourceResponse(
                    clause_id=r.clause_id,
                    section_title=r.section_title,
                    source_location=r.source_location,
                    quoted_text=r.text[:120] + "..." if len(r.text) > 120 else r.text,
                )
                for r in candidate_results[:request.top_k]
            ],
            uncertainty="AI answer generation is currently unavailable.",
            message="AI answer generation is currently unavailable.",
            disclaimer=SYSTEM_DISCLAIMER,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Answer generation error: {str(exc)}")


@router.post("/documents/{document_id}/query", response_model=QuestionResponse)
async def query_document(
    document_id: str,
    request: QuestionRequest,
) -> QuestionResponse:
    """Document-grounded Q&A endpoint (alias for /ask)."""
    return await ask_document(document_id=document_id, request=request)



@router.get("/documents/{document_id}/findings")
async def get_findings(document_id: str):
    """Placeholder endpoint for retrieving findings of an analyzed document."""
    raise HTTPException(
        status_code=501,
        detail="Findings retrieval will be implemented in Phase 5.",
    )
