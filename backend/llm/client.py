"""Google Gemini API service and deterministic mock service using the official google-genai SDK.

Non-negotiable rules:
- Uses ONLY the official modern Google GenAI SDK (google-genai).
- Three-state execution behavior:
  1. GEMINI_MOCK_MODE=true -> MockGeminiService
  2. GEMINI_MOCK_MODE=false + valid GEMINI_API_KEY -> Real Gemini client
  3. GEMINI_MOCK_MODE=false + missing API key -> LLMUnavailableError (mapped to 'llm_unavailable')
- Fully environment-configurable GEMINI_MODEL_NAME.
- All outputs are deterministically validated for evidence grounding before returning.
"""
import json
import logging
from typing import List, Optional, Dict, Any

from backend.config import settings
from backend.document.models import Clause
from backend.analysis.models import ClauseRelationship
from backend.llm.models import (
    ExplanationOutput,
    EvidenceReference,
    ValidationResult,
    QuestionAnswerOutput,
    QuestionSource,
    QAValidationResult,
)
from backend.llm.prompts import (
    SYSTEM_INSTRUCTION,
    QA_SYSTEM_INSTRUCTION,
    build_clause_explanation_prompt,
    build_relationship_explanation_prompt,
    build_synthesis_prompt,
    build_qa_prompt,
)
from backend.llm.validator import EvidenceValidator


logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Base exception for LLM operations."""
    pass


class LLMUnavailableError(LLMServiceError):
    """Raised when the LLM service is unavailable (e.g. missing API key, network timeout)."""
    pass


class LLMValidationError(LLMServiceError):
    """Raised when generated explanation fails factual evidence grounding validation."""
    def __init__(self, message: str, validation_result: ValidationResult):
        super().__init__(message)
        self.validation_result = validation_result


class GeminiService:
    """Production service communicating with Google Gemini API via official google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.GEMINI_TEMPERATURE
        self.max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else settings.GEMINI_MAX_OUTPUT_TOKENS
        )
        self._client = None

    def _get_client(self):
        """Lazily initialize official google-genai Client."""
        if not self.api_key or not self.api_key.strip():
            raise LLMUnavailableError(
                "Gemini API key is not configured and GEMINI_MOCK_MODE is false. "
                "AI explanation is unavailable."
            )

        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                raise LLMUnavailableError(f"Failed to initialize Google GenAI Client: {str(exc)}") from exc

        return self._client

    def _call_gemini(self, prompt: str) -> ExplanationOutput:
        """Call Gemini model with structured JSON schema and parse response."""
        client = self._get_client()

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_json_schema=ExplanationOutput.model_json_schema(),
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )

            logger.info("Calling Gemini model '%s' with prompt length %d", self.model_name, len(prompt))
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise LLMServiceError("Gemini returned an empty response.")

            parsed_json = json.loads(raw_text)
            return ExplanationOutput(**parsed_json)

        except LLMUnavailableError:
            raise
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON output: %s", str(exc))
            raise LLMServiceError(f"Malformed JSON returned by Gemini: {str(exc)}") from exc
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str:
                raise LLMUnavailableError("Gemini API rate limit exceeded.") from exc
            if "timeout" in err_str or "deadline" in err_str:
                raise LLMUnavailableError("Gemini API request timed out.") from exc
            if "unauthenticated" in err_str or "api_key" in err_str or "401" in err_str:
                raise LLMUnavailableError("Invalid or unauthorized Gemini API key.") from exc
            raise LLMServiceError(f"Gemini API call failed: {str(exc)}") from exc

    def explain_clause(self, clause: Clause) -> ExplanationOutput:
        """Generate evidence-grounded explanation for a single clause."""
        prompt = build_clause_explanation_prompt(clause)
        raw_output = self._call_gemini(prompt)
        val = EvidenceValidator.validate(raw_output, [clause])
        if not val.is_valid:
            logger.warning("Evidence validation failed for clause '%s': %s", clause.clause_id, val.validation_errors)
        return val.output

    def explain_relationship(
        self,
        rel: ClauseRelationship,
        src_clause: Clause,
        tgt_clause: Clause,
    ) -> ExplanationOutput:
        """Generate evidence-grounded explanation for a clause relationship."""
        prompt = build_relationship_explanation_prompt(rel, src_clause, tgt_clause)
        raw_output = self._call_gemini(prompt)
        val = EvidenceValidator.validate(raw_output, [src_clause, tgt_clause])
        if not val.is_valid:
            logger.warning(
                "Evidence validation failed for relationship '%s': %s",
                rel.relationship_id,
                val.validation_errors,
            )
        return val.output

    def synthesize_clauses(
        self,
        clauses: List[Clause],
        relationships: List[ClauseRelationship],
        user_concerns: Optional[List[str]] = None,
    ) -> ExplanationOutput:
        """Synthesize multiple clauses and relationships into an executive summary."""
        if not clauses:
            return ExplanationOutput(
                summary="No clauses were supplied for synthesis.",
                evidence_sufficient=False,
                uncertainty="No document evidence available to synthesize.",
            )

        # For large documents, prioritize substantive attention clauses and top relationships
        selected_clauses = clauses
        if len(clauses) > 20:
            high_att = [c for c in clauses if c.attention_level == "High Attention"]
            med_att = [c for c in clauses if c.attention_level == "Medium Attention"]
            low_att = [c for c in clauses if c.attention_level == "Low Attention"]
            info = [c for c in clauses if c.attention_level not in ("High Attention", "Medium Attention", "Low Attention")]

            prioritized = high_att + med_att + low_att
            if len(prioritized) < 20:
                prioritized += info[:(20 - len(prioritized))]
            selected_clauses = prioritized[:20]

        selected_rels = relationships
        if len(relationships) > 15:
            selected_rels = sorted(relationships, key=lambda r: getattr(r, "confidence", 0.0), reverse=True)[:15]

        prompt = build_synthesis_prompt(selected_clauses, selected_rels, user_concerns)
        raw_output = self._call_gemini(prompt)
        val = EvidenceValidator.validate(raw_output, clauses)
        if not val.is_valid:
            logger.warning("Evidence validation failed during synthesis: %s", val.validation_errors)
        return val.output

    def _call_gemini_qa(self, prompt: str) -> QuestionAnswerOutput:
        """Call Gemini model with structured JSON schema for Q&A and parse response."""
        client = self._get_client()

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                system_instruction=QA_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_json_schema=QuestionAnswerOutput.model_json_schema(),
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )

            logger.info("Calling Gemini Q&A with prompt length %d", len(prompt))
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            raw_text = response.text
            if not raw_text or not raw_text.strip():
                raise LLMServiceError("Gemini returned an empty response.")

            parsed_json = json.loads(raw_text)
            return QuestionAnswerOutput(**parsed_json)

        except LLMUnavailableError:
            raise
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini Q&A JSON output: %s", str(exc))
            raise LLMServiceError(f"Malformed JSON returned by Gemini: {str(exc)}") from exc
        except Exception as exc:
            err_str = str(exc).lower()
            if "quota" in err_str or "rate" in err_str or "429" in err_str:
                raise LLMUnavailableError("Gemini API rate limit exceeded.") from exc
            if "timeout" in err_str or "deadline" in err_str:
                raise LLMUnavailableError("Gemini API request timed out.") from exc
            if "unauthenticated" in err_str or "api_key" in err_str or "401" in err_str:
                raise LLMUnavailableError("Invalid or unauthorized Gemini API key.") from exc
            raise LLMServiceError(f"Gemini API call failed: {str(exc)}") from exc

    def answer_question(
        self,
        question: str,
        clauses: List[Clause],
        scores: Optional[Dict[str, float]] = None,
    ) -> QuestionAnswerOutput:
        """Answer user question using only retrieved evidence clauses."""
        if not clauses:
            return QuestionAnswerOutput(
                answer="The supplied Terms do not contain enough evidence to answer this question.",
                evidence_sufficient=False,
                confidence=0.0,
                sources=[],
                uncertainty="No document evidence available to answer this question.",
            )

        prompt = build_qa_prompt(question, clauses, scores)
        raw_output = self._call_gemini_qa(prompt)
        val = EvidenceValidator.validate_qa(raw_output, clauses)
        if not val.is_valid:
            logger.warning("Evidence validation failed for Q&A: %s", val.validation_errors)
        return val.output



class MockGeminiService:
    """Deterministic, offline mock service for automated testing and development without live API keys."""

    def __init__(self, **kwargs):
        self.model_name = "mock-gemini-service"

    def explain_clause(self, clause: Clause) -> ExplanationOutput:
        """Generate grounded mock explanation based strictly on supplied clause."""
        short_quote = clause.text[:80] + "..." if len(clause.text) > 80 else clause.text
        # Extract first 40 characters for a verified verbatim quote
        first_sentence = clause.text.split(".")[0]
        quote = first_sentence if len(first_sentence) > 10 else clause.text[:50]

        return ExplanationOutput(
            summary=(
                f"The clause '{clause.section_title or clause.clause_id}' outlines terms regarding "
                f"{clause.primary_category or 'general service usage'}."
            ),
            attention_explanation=(
                f"This provision warrants user attention because of contractual indicators: "
                f"{', '.join(clause.attention_reasons) if clause.attention_reasons else 'standard terms'}."
            ),
            relationship_explanation="",
            key_points=[
                f"Applies to {clause.primary_category or 'General'} terms.",
                f"Identified with attention level: {clause.attention_level or 'Informational'}.",
            ],
            evidence_references=[
                EvidenceReference(
                    clause_id=clause.clause_id,
                    source_location=clause.source_location.to_display_string(),
                    quoted_text=quote,
                )
            ],
            evidence_sufficient=True,
            uncertainty="",
            disclaimer="Informational document analysis only; not legal advice.",
        )

    def explain_relationship(
        self,
        rel: ClauseRelationship,
        src_clause: Clause,
        tgt_clause: Clause,
    ) -> ExplanationOutput:
        """Generate grounded mock relationship explanation."""
        first_src_sentence = src_clause.text.split(".")[0]
        first_tgt_sentence = tgt_clause.text.split(".")[0]
        src_quote = first_src_sentence if len(first_src_sentence) > 10 else src_clause.text[:50]
        tgt_quote = first_tgt_sentence if len(first_tgt_sentence) > 10 else tgt_clause.text[:50]

        return ExplanationOutput(
            summary=(
                f"Cross-clause interaction of type '{rel.relationship_type}' between "
                f"'{src_clause.section_title or src_clause.clause_id}' and '{tgt_clause.section_title or tgt_clause.clause_id}'."
            ),
            attention_explanation="Users should be aware of how these two provisions interact across the agreement.",
            relationship_explanation=rel.rationale,
            key_points=[
                f"Source provision: {src_clause.section_title or src_clause.clause_id}",
                f"Target provision: {tgt_clause.section_title or tgt_clause.clause_id}",
                f"Interaction type: {rel.relationship_type}",
            ],
            evidence_references=[
                EvidenceReference(
                    clause_id=src_clause.clause_id,
                    source_location=src_clause.source_location.to_display_string(),
                    quoted_text=src_quote,
                ),
                EvidenceReference(
                    clause_id=tgt_clause.clause_id,
                    source_location=tgt_clause.source_location.to_display_string(),
                    quoted_text=tgt_quote,
                ),
            ],
            evidence_sufficient=True,
            uncertainty="",
            disclaimer="Informational document analysis only; not legal advice.",
        )

    def synthesize_clauses(
        self,
        clauses: List[Clause],
        relationships: List[ClauseRelationship],
        user_concerns: Optional[List[str]] = None,
    ) -> ExplanationOutput:
        """Generate grounded mock synthesis across multiple clauses."""
        if not clauses:
            return ExplanationOutput(
                summary="No clauses were supplied for synthesis.",
                evidence_sufficient=False,
                uncertainty="No document evidence available to synthesize.",
            )

        refs = []
        for c in clauses[:3]:
            sent = c.text.split(".")[0]
            quote = sent if len(sent) > 10 else c.text[:40]
            refs.append(
                EvidenceReference(
                    clause_id=c.clause_id,
                    source_location=c.source_location.to_display_string(),
                    quoted_text=quote,
                )
            )

        concerns_text = f" (focusing on {', '.join(user_concerns)})" if user_concerns else ""
        return ExplanationOutput(
            summary=f"Synthesized analysis of {len(clauses)} clauses and {len(relationships)} relationships{concerns_text}.",
            attention_explanation="Several provisions describe key operational, billing, and account boundaries.",
            relationship_explanation=f"Identified {len(relationships)} structural cross-clause interactions.",
            key_points=[
                f"Analyzed {len(clauses)} contractual clauses.",
                f"Identified {len(relationships)} cross-clause relationships.",
            ],
            evidence_references=refs,
            evidence_sufficient=True,
            uncertainty="",
            disclaimer="Informational document analysis only; not legal advice.",
        )

    def answer_question(
        self,
        question: str,
        clauses: List[Clause],
        scores: Optional[Dict[str, float]] = None,
    ) -> QuestionAnswerOutput:
        """Generate grounded mock answer using retrieved clauses."""
        if not clauses:
            return QuestionAnswerOutput(
                answer="The supplied Terms do not contain enough evidence to answer this question.",
                evidence_sufficient=False,
                confidence=0.0,
                sources=[],
                uncertainty="No document evidence available to answer this question.",
            )

        q_lower = question.lower()

        # Check if question is asking for a legal conclusion
        is_legal_question = any(term in q_lower for term in ["illegal", "unlawful", "law", "gdpr", "enforceable", "valid", "compliant"])

        # Find best matching clause among retrieved clauses
        target_clause = None
        for c in clauses:
            c_text_lower = c.text.lower()
            if "renew" in q_lower or "subscription" in q_lower:
                if "renew" in c_text_lower or "subscription" in c_text_lower or "billing" in c_text_lower:
                    target_clause = c
                    break
            elif "data" in q_lower or "retention" in q_lower or "close" in q_lower or "closure" in q_lower:
                if "retain" in c_text_lower or "retention" in c_text_lower or "personal data" in c_text_lower:
                    target_clause = c
                    break
            elif "terminate" in q_lower or "suspend" in q_lower or "account" in q_lower:
                if "terminate" in c_text_lower or "suspend" in c_text_lower or "account" in c_text_lower:
                    target_clause = c
                    break
            elif "change" in q_lower or "modify" in q_lower or "notice" in q_lower:
                if "modify" in c_text_lower or "replace" in c_text_lower or "pricing" in c_text_lower:
                    target_clause = c
                    break

        if target_clause is None:
            target_clause = clauses[0]

        # Extract verified verbatim quote from target_clause
        clause_text = target_clause.text
        sentences = [s.strip() for s in clause_text.split(".") if len(s.strip()) > 15]
        default_quote = sentences[0] if sentences else clause_text[:60]

        # Build direct, neutral answer
        if "renew" in q_lower or "subscription" in q_lower:
            quote = (
                "Your subscription will automatically renew at the end of each billing cycle unless you cancel it at least thirty (30) days prior to the expiration date."
                if "Your subscription will automatically renew" in clause_text
                else default_quote
            )
            if is_legal_question:
                answer = (
                    "ANSWER\n"
                    "The supplied Terms state that subscriptions automatically renew unless cancelled at least thirty (30) days in advance. "
                    "Whether this provision is legally valid or enforceable cannot be determined from this document analysis.\n\n"
                    "WHAT THE TERMS SAY\n"
                    "The document specifies automatic renewal conditions and a cancellation notice period. "
                    "This system provides informational document analysis and does not make determinations of legality or enforceability."
                )
            else:
                answer = (
                    "ANSWER\n"
                    "Yes, your subscription will renew automatically at the end of each billing cycle unless cancelled at least thirty (30) days prior to expiration.\n\n"
                    "WHAT THE TERMS SAY\n"
                    "The Terms state that all paid plans are billed in advance on a recurring basis and will automatically renew unless cancelled at least 30 days before the period ends. Subscription fees are stated to be non-refundable."
                )
        elif ("data" in q_lower or "close" in q_lower or "closure" in q_lower) and ("retain" in clause_text.lower() or "retention" in clause_text.lower()):
            quote = (
                "Upon termination of your account, we may retain your personal data, usage logs, and content for a period of up to seven (7) years to comply with regulatory obligations and enforce our agreements."
                if "Upon termination of your account" in clause_text
                else default_quote
            )
            if is_legal_question:
                answer = (
                    "ANSWER\n"
                    "The supplied Terms state that personal data may be retained for up to seven (7) years following account termination. "
                    "Whether this retention policy complies with regulations such as GDPR cannot be determined from this document analysis.\n\n"
                    "WHAT THE TERMS SAY\n"
                    "The document states data may be kept for up to 7 years to comply with obligations. This system does not evaluate regulatory compliance."
                )
            else:
                answer = (
                    "ANSWER\n"
                    "The company may retain your personal data, usage logs, and content for up to seven (7) years after your account is closed.\n\n"
                    "WHAT THE TERMS SAY\n"
                    "Upon account termination, the Terms state the company retains data, logs, and content for regulatory compliance and agreement enforcement. The document does not state an immediate deletion right."
                )
        elif "terminate" in q_lower or "account" in q_lower:
            quote = (
                "we reserve the right to suspend or terminate your account immediately, without notice or liability, for any reason whatsoever."
                if "we reserve the right to suspend or terminate your account immediately" in clause_text
                else default_quote
            )
            answer = (
                "ANSWER\n"
                "Yes, the company reserves the right to suspend or terminate your account immediately without notice or liability for any reason.\n\n"
                "WHAT THE TERMS SAY\n"
                "While users may terminate their account at any time through settings, the company explicitly reserves the right to terminate accounts immediately without prior notice."
            )
        elif "change" in q_lower or "modify" in q_lower or "notice" in q_lower:
            quote = (
                "We reserve the right, at our sole discretion, to modify or replace these Terms and modify pricing at any time without prior individual notice."
                if "We reserve the right, at our sole discretion, to modify or replace these Terms" in clause_text
                else default_quote
            )
            answer = (
                "ANSWER\n"
                "Yes, the company states it can modify the Terms and pricing at any time without prior individual notice.\n\n"
                "WHAT THE TERMS SAY\n"
                "The agreement reserves unilateral discretion to change terms or pricing without prior notice, and continued use constitutes acceptance."
            )
        else:
            quote = default_quote
            answer = (
                f"ANSWER\n"
                f"Based on the supplied Terms, the agreement addresses this topic in the '{target_clause.section_title or 'General'}' provision.\n\n"
                f"WHAT THE TERMS SAY\n"
                f"The agreement states: \"{quote}\"."
            )

        source_obj = QuestionSource(
            clause_id=target_clause.clause_id,
            section_title=target_clause.section_title or "General",
            source_location=target_clause.source_location.to_display_string(),
            quoted_text=quote,
        )

        output = QuestionAnswerOutput(
            answer=answer,
            evidence_sufficient=True,
            confidence=0.88,
            sources=[source_obj],
            uncertainty="",
            disclaimer="Informational document analysis only; not legal advice.",
        )

        val = EvidenceValidator.validate_qa(output, clauses)
        return val.output



_LLM_SERVICE_INSTANCE = None


def get_llm_service() -> Any:
    """Return configured LLM service respecting strict three-state mode control.

    1. GEMINI_MOCK_MODE=true -> MockGeminiService
    2. GEMINI_MOCK_MODE=false -> GeminiService (will raise LLMUnavailableError if API key missing)
    """
    global _LLM_SERVICE_INSTANCE
    if settings.GEMINI_MOCK_MODE:
        if not isinstance(_LLM_SERVICE_INSTANCE, MockGeminiService):
            logger.info("Initializing MockGeminiService (GEMINI_MOCK_MODE=true)")
            _LLM_SERVICE_INSTANCE = MockGeminiService()
    else:
        if not isinstance(_LLM_SERVICE_INSTANCE, GeminiService):
            logger.info("Initializing production GeminiService with model '%s'", settings.GEMINI_MODEL_NAME)
            _LLM_SERVICE_INSTANCE = GeminiService()

    return _LLM_SERVICE_INSTANCE

