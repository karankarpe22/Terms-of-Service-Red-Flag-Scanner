"""Domain data models for document ingestion, clause segmentation, and classification.

Preserves exact traceability between segmented clauses and original raw document text.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class SourceLocation(BaseModel):
    """Tracks exact location of extracted clause in the original document."""
    page_number: Optional[int] = Field(default=None, description="1-based starting page number")
    end_page_number: Optional[int] = Field(default=None, description="1-based ending page number if clause spans pages")
    start_char: Optional[int] = Field(default=None, description="Character offset start in raw text")
    end_char: Optional[int] = Field(default=None, description="Character offset end in raw text")
    source_type: str = Field(default="text", description="'pdf' or 'text'")

    def to_display_string(self) -> str:
        """Formatted human-readable string for UI display."""
        if self.page_number is not None:
            if self.end_page_number and self.end_page_number != self.page_number:
                page_str = f"Pages {self.page_number}-{self.end_page_number}"
            else:
                page_str = f"Page {self.page_number}"
            if self.start_char is not None and self.end_char is not None:
                return f"{page_str} (chars {self.start_char}-{self.end_char})"
            return page_str
        elif self.start_char is not None and self.end_char is not None:
            return f"Offset {self.start_char}-{self.end_char}"
        return "Pasted Text"

    def __str__(self) -> str:
        return self.to_display_string()


class Clause(BaseModel):
    """Internal representation of a single extracted contractual clause."""
    clause_id: str = Field(..., description="Stable, unique identifier e.g. doc_123_clause_001")
    document_id: str = Field(..., description="Parent document identifier")
    clause_index: int = Field(..., description="1-based sequential position in document")
    section_title: Optional[str] = Field(default="General", description="Associated section or clause heading")
    text: str = Field(..., description="Substantive text of the clause")
    raw_text: Optional[str] = Field(default=None, description="Verbatim raw text before cleaning")
    category: Optional[str] = Field(default=None, description="Primary category alias")
    primary_category: Optional[str] = Field(default=None, description="Primary PRD category")
    secondary_categories: List[str] = Field(default_factory=list, description="Secondary PRD categories")
    category_confidence: Optional[float] = Field(default=None, description="Model score between 0.0 and 1.0")
    attention_level: Optional[str] = Field(default=None, description="Attention level (High, Medium, Low, Informational)")
    attention_reasons: List[str] = Field(default_factory=list, description="Transparent indicators explaining attention level")
    source_location: SourceLocation = Field(default_factory=SourceLocation)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PageInfo(BaseModel):
    """Metadata and raw text for an extracted document page."""
    page_number: int = Field(..., description="1-based page number")
    text: str = Field(..., description="Raw text of this page")
    char_start: int = Field(..., description="Starting character offset in full document text")
    char_end: int = Field(..., description="Ending character offset in full document text")


class Document(BaseModel):
    """Internal representation of an ingested Terms of Service document."""
    document_id: str = Field(..., description="Unique document ID")
    filename: Optional[str] = Field(default=None, description="Original filename if uploaded")
    source_type: str = Field(default="text", description="'pdf' or 'text'")
    raw_text: str = Field(default="", description="Complete raw extracted text preserving exact original")
    cleaned_text: str = Field(default="", description="Cleaned document text")
    pages: List[PageInfo] = Field(default_factory=list, description="Per-page extracted text and offsets")
    clauses: List[Clause] = Field(default_factory=list, description="Segmented clauses")
    metadata: Dict[str, Any] = Field(default_factory=dict)
