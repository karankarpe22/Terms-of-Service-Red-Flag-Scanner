"""Prompt definitions and instruction builders for Google Gemini API.

Non-negotiable rules:
- Primary scope control: Prompt as an objective document analyst, not a lawyer.
- Never state that a clause is illegal, invalid, unenforceable, compliant, or non-compliant.
- Explicit instruction: "If the supplied evidence does not support a factual statement, do not make that statement."
- Reason solely over supplied structured evidence.
"""
from typing import List, Optional
import json
from backend.document.models import Clause
from backend.analysis.models import ClauseRelationship


SYSTEM_INSTRUCTION = (
    "You are an objective informational document-analysis assistant analyzing Terms of Service agreements. "
    "You are NOT a lawyer and you DO NOT provide legal advice. "
    "NEVER assert, state, or imply that any clause or contract is illegal, unlawful, invalid, enforceable, "
    "unenforceable, compliant, or in violation of any law (including GDPR, consumer protection, or contract law). "
    "NEVER claim to determine whether a provision would be upheld or struck down in court. "
    "Use strictly neutral, factual language such as 'The Terms state that...', 'This clause describes...', "
    "or 'This provision may warrant user attention because...'.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. Use ONLY the supplied evidence clauses provided in the prompt. Do NOT use outside knowledge about companies or legal standards.\n"
    "2. Do NOT invent, assume, or infer missing contractual terms, rights, refund rules, or deletion policies not explicitly stated in the evidence.\n"
    "3. Do NOT invent clause IDs, section numbers, or page numbers. Use only the exact IDs and locations provided.\n"
    "4. All quoted evidence excerpts in 'evidence_references' must be exact verbatim quotes from the supplied clause text.\n"
    "5. If the supplied evidence does not support a factual statement, do not make that statement.\n"
    "6. If the supplied evidence is insufficient to address a query or determine a condition, explicitly set evidence_sufficient to false and describe the limitation in uncertainty."
)


def build_clause_explanation_prompt(clause: Clause) -> str:
    """Construct prompt for explaining a single clause."""
    reasons_str = ", ".join(clause.attention_reasons) if clause.attention_reasons else "None noted"
    evidence_payload = {
        "task": "EXPLAIN_CLAUSE",
        "clause": {
            "clause_id": clause.clause_id,
            "section_title": clause.section_title or "General",
            "category": clause.primary_category or clause.category or "General",
            "attention_level": clause.attention_level or "Informational",
            "attention_indicators": reasons_str,
            "source_location": clause.source_location.to_display_string(),
            "text": clause.text,
        },
    }

    return (
        f"Analyze and explain the following Terms of Service clause based solely on the supplied evidence.\n"
        f"Explain in simple, plain language what the clause states and why its contractual terms warrant closer human attention.\n"
        f"Include verbatim evidence references using the exact clause_id and source_location provided.\n\n"
        f"SUPPLIED EVIDENCE:\n{json.dumps(evidence_payload, indent=2)}"
    )


def build_relationship_explanation_prompt(
    rel: ClauseRelationship,
    src_clause: Clause,
    tgt_clause: Clause,
) -> str:
    """Construct prompt for explaining an interaction or potential tension between two clauses."""
    evidence_payload = {
        "task": "EXPLAIN_RELATIONSHIP",
        "relationship_type": rel.relationship_type,
        "rationale": rel.rationale,
        "trigger": rel.evidence.trigger,
        "source_clause": {
            "clause_id": src_clause.clause_id,
            "section_title": src_clause.section_title or "General",
            "category": src_clause.primary_category or "General",
            "attention_level": src_clause.attention_level or "Informational",
            "source_location": src_clause.source_location.to_display_string(),
            "text": src_clause.text,
        },
        "target_clause": {
            "clause_id": tgt_clause.clause_id,
            "section_title": tgt_clause.section_title or "General",
            "category": tgt_clause.primary_category or "General",
            "attention_level": tgt_clause.attention_level or "Informational",
            "source_location": tgt_clause.source_location.to_display_string(),
            "text": tgt_clause.text,
        },
    }

    return (
        f"Analyze and explain the cross-clause relationship between the two supplied clauses.\n"
        f"Relationship Type: {rel.relationship_type}.\n"
        f"Explain how these provisions interact, qualify each other, operate across time, or present differing conditions (potential tension).\n"
        f"Maintain strictly neutral language and do NOT claim that the clauses are legally contradictory or invalid.\n"
        f"Ground your explanation in both clauses and cite verbatim quotes in evidence_references.\n\n"
        f"SUPPLIED EVIDENCE:\n{json.dumps(evidence_payload, indent=2)}"
    )


def build_synthesis_prompt(
    clauses: List[Clause],
    relationships: List[ClauseRelationship],
    user_concerns: Optional[List[str]] = None,
) -> str:
    """Construct prompt for synthesizing multiple retrieved clauses and their relationships."""
    clauses_payload = [
        {
            "clause_id": c.clause_id,
            "section_title": c.section_title or "General",
            "category": c.primary_category or "General",
            "attention_level": c.attention_level or "Informational",
            "source_location": c.source_location.to_display_string(),
            "text": c.text,
        }
        for c in clauses
    ]

    relationships_payload = [
        {
            "relationship_id": r.relationship_id,
            "source_clause_id": r.source_clause_id,
            "target_clause_id": r.target_clause_id,
            "relationship_type": r.relationship_type,
            "rationale": r.rationale,
            "trigger": r.evidence.trigger,
        }
        for r in relationships
    ]

    evidence_payload = {
        "task": "MULTI_CLAUSE_SYNTHESIS",
        "user_concerns": user_concerns or [],
        "clauses": clauses_payload,
        "relationships": relationships_payload,
    }

    concerns_instruction = ""
    if user_concerns:
        concerns_instruction = (
            f"The user has expressed particular interest in these areas: {', '.join(user_concerns)}. "
            f"Synthesize relevant clauses and relationships in these areas more prominently, "
            f"without altering underlying facts or fabricating risks.\n"
        )

    return (
        f"Synthesize the supplied clauses and detected relationships into an evidence-grounded summary.\n"
        f"{concerns_instruction}"
        f"Highlight key takeaways in key_points and explain noteworthy interactions in relationship_explanation.\n"
        f"Every finding must be grounded in the supplied clause texts and cited in evidence_references.\n\n"
        f"SUPPLIED EVIDENCE:\n{json.dumps(evidence_payload, indent=2)}"
    )


QA_SYSTEM_INSTRUCTION = (
    "You are answering a question about a Terms of Service document.\n"
    "Use ONLY the supplied document evidence.\n"
    "Do not use outside knowledge.\n"
    "Do not invent contractual provisions.\n"
    "Do not invent section numbers, dates, policies, rights, obligations, or facts.\n"
    "If the evidence does not answer the question, explicitly state that the supplied evidence is insufficient and set evidence_sufficient to false.\n"
    "Do not make legal conclusions.\n"
    "Do not determine legality, illegality, enforceability, validity, regulatory compliance, or statutory violations.\n"
    "Distinguish what the document explicitly states from any uncertainty.\n\n"
    "Format answers in a concise, direct, evidence-grounded structure:\n"
    "ANSWER\n[plain-language answer]\n\n"
    "WHAT THE TERMS SAY\n[short explanation of what the text explicitly states]\n\n"
    "Every cited source in 'sources' must use the exact clause_id, section_title, and source_location from the supplied evidence, "
    "and quoted_text must be an exact verbatim excerpt from that clause."
)


def build_qa_prompt(
    question: str,
    clauses: List[Clause],
    scores: Optional[dict] = None,
) -> str:
    """Construct structured prompt for answering questions using retrieved evidence clauses."""
    scores = scores or {}
    evidence_payload = {
        "question": question,
        "evidence": [
            {
                "clause_id": c.clause_id,
                "section_title": c.section_title or "General",
                "text": c.text,
                "category": c.primary_category or c.category or "General",
                "attention_level": c.attention_level or "Informational",
                "source_location": c.source_location.to_display_string(),
                "similarity_score": round(scores.get(c.clause_id, 0.0), 3),
            }
            for c in clauses
        ],
    }

    return (
        f"Answer the following question about the Terms of Service using ONLY the supplied evidence clauses.\n"
        f"Do not guess, do not use external facts, and do not make legal conclusions.\n\n"
        f"SUPPLIED EVIDENCE:\n{json.dumps(evidence_payload, indent=2)}"
    )

