"""Generative AI layer: Gemini API client, prompts, and output validation."""
from backend.llm.models import ExplanationOutput, EvidenceReference, ValidationResult
from backend.llm.validator import EvidenceValidator
from backend.llm.client import (
    GeminiService,
    MockGeminiService,
    get_llm_service,
    LLMServiceError,
    LLMUnavailableError,
    LLMValidationError,
)

__all__ = [
    "ExplanationOutput",
    "EvidenceReference",
    "ValidationResult",
    "EvidenceValidator",
    "GeminiService",
    "MockGeminiService",
    "get_llm_service",
    "LLMServiceError",
    "LLMUnavailableError",
    "LLMValidationError",
]
