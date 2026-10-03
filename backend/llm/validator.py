"""Evidence validation and hallucination protection module for Gemini responses.

Enforces deterministic grounding checks:
1. Referenced clause IDs must exist in supplied evidence.
2. Referenced source locations must match supplied clause metadata.
3. Quoted evidence snippets must be exact or normalized substrings of corresponding clause texts.
4. Secondary guardrail: verifies that Gemini-generated explanatory fields do not assert legal conclusions
   (e.g., 'is illegal', 'violates GDPR', 'is unenforceable', 'is void', 'violating the law').
   NOTE: In accordance with project rules, this check applies strictly to generated explanatory text,
   NEVER to original source clause text or verbatim quotations.
"""
import re
from typing import List, Dict, Any, Set
from backend.document.models import Clause
from backend.llm.models import (
    ExplanationOutput,
    ValidationResult,
    EvidenceReference,
    QuestionAnswerOutput,
    QAValidationResult,
    QuestionSource,
)


# Secondary guardrail: patterns asserting definitive legal conclusions in AI explanations
FORBIDDEN_LEGAL_CONCLUSIONS = [
    re.compile(r"\b(?:is|are)\s+(?:clearly\s+|definitely\s+|strictly\s+)?(?:illegal|unlawful)\b", re.I),
    re.compile(r"\b(?:violates|in violation of)\s+(?:the\s+law|gdpr|statute|regulation)\b", re.I),
    re.compile(r"\b(?:is|are)\s+(?:legally\s+)?(?:unenforceable|invalid|void(?:\s+and\s+without\s+effect)?)\b", re.I),
    re.compile(r"\bbreaking\s+the\s+law\b", re.I),
    re.compile(r"\bcourt\s+will\s+(?:strike\s+down|invalidate)\b", re.I),
]


def _normalize_ws(s: str) -> str:
    """Collapse excess whitespace and strip invisible characters for robust substring comparison."""
    if not s:
        return ""
    # Strip zero-width and invisible formatting characters (Word Joiner, ZWSP, ZWNJ, ZWJ, BOM, soft hyphen, etc.)
    s = re.sub(r"[\u200b-\u200d\u2060\ufeff\u00ad\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", s)
    # Standardize quotes, backticks and smart quotes
    s = re.sub(r"[\u2018\u2019\u201a\u201b`´]", "'", s)
    s = re.sub(r"[\u201c\u201d\u201e\u201f«»]", '"', s)
    # Standardize dashes and hyphens
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    # Standardize all whitespace variants
    s = re.sub(r"[\s\u00a0\u2000-\u200a\u202f\u205f\u3000]+", " ", s)
    return s.strip().lower()


class EvidenceValidator:
    """Deterministic validator verifying factual grounding and hallucination prevention."""

    @staticmethod
    def validate(
        output: ExplanationOutput,
        supplied_clauses: List[Clause],
    ) -> ValidationResult:
        """Validate generated ExplanationOutput against supplied source clauses.

        Args:
            output: Generated explanation from Gemini or Mock.
            supplied_clauses: List of Clause domain objects provided as evidence.

        Returns:
            ValidationResult with validation status, errors, and sanitized output.
        """
        errors: List[str] = []

        # Index supplied clauses
        supplied_map: Dict[str, Clause] = {c.clause_id: c for c in supplied_clauses}
        supplied_ids: Set[str] = set(supplied_map.keys())

        # 1. Validate Referenced Clause IDs
        for ref in output.evidence_references:
            if ref.clause_id not in supplied_ids:
                errors.append(
                    f"Unsupported clause ID in evidence reference: '{ref.clause_id}'. "
                    f"Valid IDs were: {sorted(list(supplied_ids))}."
                )

        # 2. Validate Source Locations & Verbatim Quotations
        for ref in output.evidence_references:
            if ref.clause_id in supplied_map:
                clause = supplied_map[ref.clause_id]
                expected_loc = clause.source_location.to_display_string()

                # Verify location matches if cited
                if ref.source_location:
                    norm_cited = _normalize_ws(ref.source_location)
                    norm_expected = _normalize_ws(expected_loc)
                    if norm_cited and norm_cited != norm_expected:
                        errors.append(
                            f"Location mismatch for clause '{ref.clause_id}': cited '{ref.source_location}', "
                            f"expected '{expected_loc}'."
                        )
                else:
                    ref.source_location = expected_loc

                # Verify quoted text is an exact or normalized substring of the original clause
                norm_quote = _normalize_ws(ref.quoted_text).strip(" .…'\"")
                norm_source = _normalize_ws(clause.text)

                if norm_quote and norm_quote not in norm_source:
                    # Fallback: check alphanumeric words to tolerate minor punctuation/whitespace variations
                    punct_quote = re.sub(r"[^\w\s]", "", norm_quote)
                    punct_source = re.sub(r"[^\w\s]", "", norm_source)
                    if not (punct_quote and punct_quote in punct_source):
                        errors.append(
                            f"Fabricated or modified quotation in clause '{ref.clause_id}': "
                            f"quoted snippet \"{ref.quoted_text[:80]}...\" was not found in source clause text."
                        )

        # 3. Secondary Guardrail: Check generated explanatory fields for forbidden legal conclusions
        explanatory_texts = [
            output.summary,
            output.attention_explanation,
            output.relationship_explanation,
            output.uncertainty,
        ] + output.key_points

        for text_block in explanatory_texts:
            if not text_block:
                continue
            for pat in FORBIDDEN_LEGAL_CONCLUSIONS:
                match = pat.search(text_block)
                if match:
                    errors.append(
                        f"Forbidden legal conclusion detected in explanation: '{match.group(0)}'. "
                        f"The system must remain informational and neutral without asserting legal determinations."
                    )

        # 4. Construct Validation Outcome
        if errors:
            # Controlled failure state: do NOT fabricate replacement explanation
            sanitized = ExplanationOutput(
                summary="The generated explanation could not be validated against the supplied document evidence.",
                attention_explanation="",
                relationship_explanation="",
                key_points=[],
                evidence_references=[],
                evidence_sufficient=False,
                uncertainty=f"Evidence validation failed with {len(errors)} error(s): " + "; ".join(errors),
                disclaimer=output.disclaimer,
            )
            return ValidationResult(
                is_valid=False,
                validation_errors=errors,
                output=sanitized,
            )

        return ValidationResult(
            is_valid=True,
            validation_errors=[],
            output=output,
        )

    @staticmethod
    def validate_qa(
        output: QuestionAnswerOutput,
        supplied_clauses: List[Clause],
    ) -> QAValidationResult:
        """Validate generated QuestionAnswerOutput against supplied source clauses.

        Args:
            output: Generated answer from Gemini or Mock.
            supplied_clauses: List of Clause domain objects provided as retrieved evidence.

        Returns:
            QAValidationResult with validation status, errors, and sanitized output.
        """
        errors: List[str] = []

        supplied_map: Dict[str, Clause] = {c.clause_id: c for c in supplied_clauses}
        supplied_ids: Set[str] = set(supplied_map.keys())

        # 1. Validate Referenced Clause IDs
        for src in output.sources:
            if src.clause_id not in supplied_ids:
                errors.append(
                    f"Unsupported clause ID in source citation: '{src.clause_id}'. "
                    f"Retrieved candidate IDs were: {sorted(list(supplied_ids))}."
                )

        # 2. Validate Section Titles, Locations & Verbatim Quotations
        for src in output.sources:
            if src.clause_id in supplied_map:
                clause = supplied_map[src.clause_id]
                expected_loc = clause.source_location.to_display_string()
                expected_title = clause.section_title or "General"

                # Check section title matching
                if src.section_title:
                    if _normalize_ws(src.section_title) != _normalize_ws(expected_title):
                        errors.append(
                            f"Section title mismatch for clause '{src.clause_id}': cited '{src.section_title}', "
                            f"expected '{expected_title}'."
                        )
                else:
                    src.section_title = expected_title

                # Check source location matching
                if src.source_location:
                    if _normalize_ws(src.source_location) != _normalize_ws(expected_loc):
                        errors.append(
                            f"Location mismatch for clause '{src.clause_id}': cited '{src.source_location}', "
                            f"expected '{expected_loc}'."
                        )
                else:
                    src.source_location = expected_loc

                # Check verbatim or normalized quote existence
                norm_quote = _normalize_ws(src.quoted_text).strip(" .…'\"")
                norm_source = _normalize_ws(clause.text)

                if not norm_quote:
                    errors.append(
                        f"Empty quotation cited for clause '{src.clause_id}'."
                    )
                elif norm_quote not in norm_source:
                    # Fallback: check alphanumeric words to tolerate minor punctuation/whitespace variations
                    punct_quote = re.sub(r"[^\w\s]", "", norm_quote)
                    punct_source = re.sub(r"[^\w\s]", "", norm_source)
                    if not (punct_quote and punct_quote in punct_source):
                        errors.append(
                            f"Fabricated or modified quotation in clause '{src.clause_id}': "
                            f"quoted excerpt \"{src.quoted_text[:80]}...\" was not found in source clause text."
                        )

        # 3. Secondary Guardrail: Check generated answer text for forbidden legal conclusions
        answer_text = (output.answer or "") + " " + (output.uncertainty or "")
        for pat in FORBIDDEN_LEGAL_CONCLUSIONS:
            match = pat.search(answer_text)
            if match:
                errors.append(
                    f"Forbidden legal conclusion detected in answer: '{match.group(0)}'. "
                    f"The system must remain informational and neutral without asserting legal determinations."
                )

        # 4. Construct Validation Outcome
        if errors:
            sanitized = QuestionAnswerOutput(
                answer="The generated answer could not be validated against the supplied document evidence.",
                evidence_sufficient=False,
                confidence=0.0,
                sources=[],
                uncertainty=f"Evidence validation failed with {len(errors)} error(s): " + "; ".join(errors),
                disclaimer=output.disclaimer,
            )
            return QAValidationResult(
                is_valid=False,
                validation_errors=errors,
                output=sanitized,
            )

        return QAValidationResult(
            is_valid=True,
            validation_errors=[],
            output=output,
        )

