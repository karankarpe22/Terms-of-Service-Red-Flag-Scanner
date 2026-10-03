"""Structured output models and evidence validation schemas for Gemini LLM responses.

Ensures that all LLM outputs adhere to a rigid, deterministic structure.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class EvidenceReference(BaseModel):
    """Verbatim quotation and location citation referencing a supplied clause."""
    clause_id: str = Field(..., description="Exact ID of the referenced clause from supplied evidence")
    source_location: str = Field(..., description="Document location of the clause, e.g. Page 2 or Offset 100-200")
    quoted_text: str = Field(..., description="Verbatim excerpt directly quoted from the clause text")


class ExplanationOutput(BaseModel):
    """Structured, evidence-grounded output format produced by Gemini or Mock service."""
    summary: str = Field(..., description="Concise, plain-language summary of what the provision(s) state")
    attention_explanation: str = Field(
        default="",
        description="Neutral explanation of why this provision warrants closer human review",
    )
    relationship_explanation: str = Field(
        default="",
        description="Neutral explanation of how the provisions interact, sequence, or differ in conditions",
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Bullet points highlighting substantive contractual conditions",
    )
    evidence_references: List[EvidenceReference] = Field(
        default_factory=list,
        description="Direct quotes and citations traceable to supplied clauses",
    )
    evidence_sufficient: bool = Field(
        default=True,
        description="Whether the supplied text contains sufficient factual evidence for the explanation",
    )
    uncertainty: str = Field(
        default="",
        description="Notes on ambiguity, missing terms, or evidence limits (no assumptions made)",
    )
    disclaimer: str = Field(
        default="Informational document analysis only; not legal advice.",
        description="Mandatory informational disclaimer",
    )


class ValidationResult(BaseModel):
    """Result of deterministic evidence and hallucination validation."""
    is_valid: bool = Field(..., description="True if output passed all grounding and safety checks")
    validation_errors: List[str] = Field(default_factory=list, description="Descriptions of validation failures")
    output: ExplanationOutput = Field(..., description="Validated or sanitized output payload")


class QuestionSource(BaseModel):
    """Source clause citation and verified verbatim excerpt for Q&A."""
    clause_id: str = Field(..., description="Exact ID of the referenced clause from retrieved evidence")
    section_title: str = Field(..., description="Section title of the clause, e.g. 'Payment, Billing, and Auto-Renewal'")
    source_location: str = Field(..., description="Document location of the clause, e.g. Page 1 or Offset 100-200")
    quoted_text: str = Field(..., description="Verbatim excerpt directly quoted from the clause text")


class QuestionAnswerOutput(BaseModel):
    """Structured, evidence-grounded answer model produced by Gemini or Mock service for Q&A."""
    answer: str = Field(..., description="Concise, plain-language answer grounded strictly in supplied evidence")
    evidence_sufficient: bool = Field(
        default=True,
        description="Whether the supplied evidence contains sufficient factual basis to answer the question",
    )
    confidence: float = Field(
        default=0.85,
        description="Algorithmic/model output indicator. NOT legal certainty or probability of enforceability.",
    )
    sources: List[QuestionSource] = Field(
        default_factory=list,
        description="Direct source clauses and quotes supporting the answer",
    )
    uncertainty: str = Field(
        default="",
        description="Ambiguity or missing contractual terms (distinguish fact from unknown)",
    )
    disclaimer: str = Field(
        default="Informational document analysis only; not legal advice.",
        description="Mandatory informational disclaimer",
    )


class QAValidationResult(BaseModel):
    """Result of deterministic evidence validation for Q&A answers."""
    is_valid: bool = Field(..., description="True if Q&A output passed all grounding, quote, and safety checks")
    validation_errors: List[str] = Field(default_factory=list, description="Descriptions of validation failures")
    output: QuestionAnswerOutput = Field(..., description="Validated or sanitized Q&A output payload")

