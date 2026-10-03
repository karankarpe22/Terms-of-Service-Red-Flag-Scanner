"""Domain models for cross-clause relationships and candidate pairs.

Preserves deterministic evidence linking two contractual clauses within the same document.
Confidence represents an algorithmic scoring heuristic, not a legal probability.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RelationshipEvidence(BaseModel):
    """Textual evidence substantiating a detected cross-clause relationship."""
    source_text: str = Field(..., description="Substantive text or excerpt from the source clause")
    target_text: str = Field(..., description="Substantive text or excerpt from the target clause")
    trigger: str = Field(..., description="Identified keyword, phrase, or cross-reference linking the clauses")


class ClauseRelationship(BaseModel):
    """Structured representation of a relationship between two clauses in the same document."""
    relationship_id: str = Field(..., description="Unique relationship identifier, e.g. rel_12345678_001")
    document_id: str = Field(..., description="Parent document identifier ensuring document isolation")
    source_clause_id: str = Field(..., description="Clause ID of the primary/source participant")
    target_clause_id: str = Field(..., description="Clause ID of the secondary/target participant")
    relationship_type: str = Field(
        ...,
        description="Canonical type: SUPPORTS, QUALIFIES, EXCEPTS, OVERRIDES, DEPENDS_ON, TEMPORAL, POTENTIAL_TENSION",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Algorithmic confidence score (not a statistical or legal probability)",
    )
    rationale: str = Field(..., description="Neutral explanation of how the two clauses interact")
    evidence: RelationshipEvidence = Field(..., description="Direct evidence excerpts and trigger phrase")
    source_location: str = Field(default="", description="Display location of source clause")
    target_location: str = Field(default="", description="Display location of target clause")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation matching API response specification."""
        return {
            "relationship_id": self.relationship_id,
            "document_id": self.document_id,
            "source_clause_id": self.source_clause_id,
            "target_clause_id": self.target_clause_id,
            "relationship_type": self.relationship_type,
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "evidence": {
                "source_text": self.evidence.source_text,
                "target_text": self.evidence.target_text,
                "trigger": self.evidence.trigger,
            },
            "source_location": self.source_location,
            "target_location": self.target_location,
        }


class CandidatePair(BaseModel):
    """A prioritized candidate pair of clauses slated for deterministic rule evaluation."""
    source_clause_id: str
    target_clause_id: str
    reasons: List[str] = Field(default_factory=list, description="Reasons pair was selected (e.g. category, keyword, semantic)")
    priority_score: float = Field(default=0.0, description="Priority score for candidate ranking")
