"""Evaluation script for Phase 8 Evidence-Grounded Document Q&A and Evidence Validator.

Evaluates:
1. Q&A Source/Citation Accuracy (Citation Precision, Recall, Accuracy)
2. Unsupported-Question Abstention (Abstention Precision, Recall, Rate)
3. Hallucination Resistance on Adversarial Questions (Unsupported claim rate, fabricated citation rate)
4. EvidenceValidator Unit Rejection Benchmark (Rejection rate & False acceptance rate on adversarial inputs)
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import settings
from backend.document.cleaner import clean_text
from backend.document.segmenter import segment_document
from backend.document.models import Document, Clause, SourceLocation
from backend.document.store import save_document
from backend.retrieval.vector_store import get_vector_store
from backend.llm.client import MockGeminiService
from backend.llm.models import (
    ExplanationOutput,
    EvidenceReference,
    QuestionAnswerOutput,
    QuestionSource,
)
from backend.llm.validator import EvidenceValidator
from backend.evaluation.evaluator import evaluate_qa_grounding

client = TestClient(app)

EVAL_DOC_IDS = [
    "doc_saas_cloud",
    "doc_ecommerce_marketplace",
    "doc_social_media",
    "doc_developer_api",
]


def ensure_documents_indexed() -> None:
    """Ensure all evaluation documents are present in memory and vector store."""
    docs_dir = BASE_DIR / "evaluation" / "data" / "documents"
    vector_store = get_vector_store()

    for doc_id in EVAL_DOC_IDS:
        raw_text = (docs_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        cleaned_text = clean_text(raw_text)
        doc = Document(
            document_id=doc_id,
            filename=f"{doc_id}.txt",
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            source_type="text",
        )
        clauses = segment_document(doc)
        save_document(doc)
        vector_store.index_document(doc_id, clauses)


def run_qa_system_evaluation(
    qa_gt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate end-to-end Q&A across the ground-truth questions."""
    gt_file = qa_gt_path or (BASE_DIR / "evaluation" / "data" / "qa_ground_truth.json")
    ensure_documents_indexed()

    import backend.llm.client as client_mod
    original_mock = settings.GEMINI_MOCK_MODE
    settings.GEMINI_MOCK_MODE = True
    client_mod._LLM_SERVICE_INSTANCE = None

    try:
        with open(gt_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = data["questions"]
        question_eval_records: List[Dict[str, Any]] = []

        for q in questions:
            q_id = q["question_id"]
            doc_id = q["document_id"]
            q_text = q["question"]
            expected_ans = q["expected_answerable"]
            expected_sources = q["expected_source_clause_ids"]
            required_facts = q.get("required_facts", [])
            forbidden_claims = q.get("forbidden_claims", [])

            # Call the Q&A API endpoint
            response = client.post(
                f"/api/documents/{doc_id}/ask",
                json={"question": q_text, "top_k": 5, "threshold": 0.45},
            )
            res_data = response.json() if response.status_code == 200 else {}
            system_status = res_data.get("status", "error")
            answer_text = res_data.get("answer")
            cited_sources = [s["clause_id"] for s in res_data.get("sources", [])]

            # Check for forbidden / hallucinated claims
            has_unsupported_claim = False
            if answer_text:
                ans_lower = answer_text.lower()
                for fc in forbidden_claims:
                    if fc.lower() in ans_lower:
                        has_unsupported_claim = True
                        break

            # Check for fabricated citations (cited clause ID does not exist in document or is completely irrelevant)
            has_fabricated_citation = False
            vector_store = get_vector_store()
            indexed_doc = vector_store._indices.get(doc_id)
            valid_doc_clause_ids = set(indexed_doc.clause_id_to_pos.keys()) if indexed_doc else set()

            for c_id in cited_sources:
                if c_id not in valid_doc_clause_ids:
                    has_fabricated_citation = True
                    break

            question_eval_records.append({
                "question_id": q_id,
                "document_id": doc_id,
                "question": q_text,
                "category": q.get("category", ""),
                "expected_answerable": expected_ans,
                "system_status": system_status,
                "answer": answer_text,
                "expected_source_clause_ids": expected_sources,
                "cited_clause_ids": cited_sources,
                "has_unsupported_claim": has_unsupported_claim,
                "has_fabricated_citation": has_fabricated_citation,
                "uncertainty": res_data.get("uncertainty"),
            })

        # Compute aggregate metrics
        metrics = evaluate_qa_grounding(question_eval_records)
        metrics["detailed_evaluations"] = question_eval_records
        return metrics
    finally:
        settings.GEMINI_MOCK_MODE = original_mock
        client_mod._LLM_SERVICE_INSTANCE = None


def run_adversarial_validator_evaluation() -> Dict[str, Any]:
    """Test EvidenceValidator with adversarial inputs to measure rejection rate and false acceptance rate."""
    sample_clause = Clause(
        clause_id="clause_test_001",
        document_id="doc_saas_cloud",
        clause_index=5,
        section_title="5. Subscription Billing and Auto-Renewal",
        text="Your subscription will automatically renew at the end of each billing cycle unless you cancel it at least thirty (30) days prior to the expiration date.",
        raw_text="Your subscription will automatically renew at the end of each billing cycle unless you cancel it at least thirty (30) days prior to the expiration date.",
        source_location=SourceLocation(start_char=100, end_char=260, source_type="text"),
    )

    test_cases = [
        # 1. Valid verbatim quote
        {
            "name": "valid_verbatim_quote",
            "should_pass": True,
            "output": ExplanationOutput(
                summary="The subscription renews automatically unless cancelled 30 days prior.",
                evidence_sufficient=True,
                confidence=0.90,
                evidence_references=[
                    EvidenceReference(
                        clause_id="clause_test_001",
                        section_title="5. Subscription Billing and Auto-Renewal",
                        source_location="Characters 100-260",
                        quoted_text="automatically renew at the end of each billing cycle",
                    )
                ],
            ),
        },
        # 2. Fabricated quote
        {
            "name": "fabricated_quote",
            "should_pass": False,
            "output": ExplanationOutput(
                summary="The company promises a free refund anytime.",
                evidence_sufficient=True,
                confidence=0.85,
                evidence_references=[
                    EvidenceReference(
                        clause_id="clause_test_001",
                        section_title="5. Subscription Billing and Auto-Renewal",
                        source_location="Characters 100-260",
                        quoted_text="we provide full refunds within 90 days guaranteed",
                    )
                ],
            ),
        },
        # 3. Altered quote
        {
            "name": "altered_quote",
            "should_pass": False,
            "output": ExplanationOutput(
                summary="The cancellation window is 10 days.",
                evidence_sufficient=True,
                confidence=0.80,
                evidence_references=[
                    EvidenceReference(
                        clause_id="clause_test_001",
                        section_title="5. Subscription Billing and Auto-Renewal",
                        source_location="Characters 100-260",
                        quoted_text="unless you cancel it at least ten (10) days prior",
                    )
                ],
            ),
        },
        # 4. Wrong clause ID
        {
            "name": "wrong_clause_id",
            "should_pass": False,
            "output": ExplanationOutput(
                summary="Unknown clause reference.",
                evidence_sufficient=True,
                confidence=0.75,
                evidence_references=[
                    EvidenceReference(
                        clause_id="clause_non_existent_999",
                        section_title="Nonexistent Section",
                        source_location="Characters 0-50",
                        quoted_text="automatically renew at the end",
                    )
                ],
            ),
        },
        # 5. Wrong source location
        {
            "name": "wrong_source_location",
            "should_pass": False,
            "output": ExplanationOutput(
                summary="Incorrect character location cited.",
                evidence_sufficient=True,
                confidence=0.80,
                evidence_references=[
                    EvidenceReference(
                        clause_id="clause_test_001",
                        section_title="5. Subscription Billing and Auto-Renewal",
                        source_location="Characters 9999-99999",
                        quoted_text="automatically renew at the end of each billing cycle",
                    )
                ],
            ),
        },
        # 6. Unsupported legal conclusion in explanation
        {
            "name": "unsupported_legal_conclusion",
            "should_pass": False,
            "output": ExplanationOutput(
                summary="This auto-renewal clause is strictly illegal under consumer protection statutes.",
                evidence_sufficient=True,
                confidence=0.90,
                evidence_references=[
                    EvidenceReference(
                        clause_id="clause_test_001",
                        section_title="5. Subscription Billing and Auto-Renewal",
                        source_location="Characters 100-260",
                        quoted_text="automatically renew at the end of each billing cycle",
                    )
                ],
            ),
        },
    ]

    total_adversarial = sum(1 for tc in test_cases if not tc["should_pass"])
    correctly_rejected = 0
    false_acceptances = 0
    valid_accepted = 0
    valid_rejected = 0

    results_detail = []
    for tc in test_cases:
        res = EvidenceValidator.validate(tc["output"], [sample_clause])
        passed = res.is_valid

        if not tc["should_pass"]:
            if not passed:
                correctly_rejected += 1
            else:
                false_acceptances += 1
        else:
            if passed:
                valid_accepted += 1
            else:
                valid_rejected += 1

        results_detail.append({
            "test_case": tc["name"],
            "expected_pass": tc["should_pass"],
            "validator_passed": passed,
            "errors": res.validation_errors,
        })

    rejection_rate = correctly_rejected / total_adversarial if total_adversarial > 0 else 0.0
    false_acceptance_rate = false_acceptances / total_adversarial if total_adversarial > 0 else 0.0

    return {
        "total_test_cases": len(test_cases),
        "adversarial_cases": total_adversarial,
        "correctly_rejected": correctly_rejected,
        "false_acceptances": false_acceptances,
        "rejection_rate": round(rejection_rate, 4),
        "false_acceptance_rate": round(false_acceptance_rate, 4),
        "valid_cases_accepted": valid_accepted,
        "cases_detail": results_detail,
    }


def run_full_qa_evaluation() -> Dict[str, Any]:
    """Execute both Q&A grounding evaluation and EvidenceValidator benchmark."""
    qa_metrics = run_qa_system_evaluation()
    validator_metrics = run_adversarial_validator_evaluation()
    return {
        "qa_grounding": qa_metrics,
        "evidence_validator_benchmark": validator_metrics,
    }


if __name__ == "__main__":
    res = run_full_qa_evaluation()
    qa = res["qa_grounding"]
    val = res["evidence_validator_benchmark"]

    print("=== EVIDENCE-GROUNDED Q&A EVALUATION ===")
    print(f"Total Questions Evaluated: {qa['total_questions']}")
    print(f"Answerable Questions:      {qa['answerable_questions']}")
    print(f"Unsupported Questions:     {qa['unsupported_questions']}")

    print("\nAbstention Metrics:")
    ab = qa["abstention_metrics"]
    print(f"  Precision: {ab['abstention_precision'] * 100:.2f}%")
    print(f"  Recall:    {ab['abstention_recall'] * 100:.2f}%")
    print(f"  F1 Score:  {ab['abstention_f1'] * 100:.2f}%")
    print(f"  Abstention Rate on Unsupported: {ab['abstention_rate_on_unsupported'] * 100:.2f}%")
    print(f"  Correctly Abstained: {ab['correctly_abstained']} / {ab['unsupported_total']}")

    print("\nCitation Metrics:")
    cm = qa["citation_metrics"]
    print(f"  Citation Precision: {cm['citation_precision'] * 100:.2f}%")
    print(f"  Citation Recall:    {cm['citation_recall'] * 100:.2f}%")
    print(f"  Citation Accuracy:  {cm['citation_accuracy'] * 100:.2f}%")

    print("\nHallucination Resistance:")
    hm = qa["hallucination_metrics"]
    print(f"  Unsupported Claim Rate:   {hm['unsupported_claim_rate'] * 100:.2f}% ({hm['unsupported_claim_count']} occurrences)")
    print(f"  Fabricated Citation Rate: {hm['fabricated_citation_rate'] * 100:.2f}% ({hm['fabricated_citation_count']} occurrences)")

    print("\nEvidence Validator Adversarial Benchmark:")
    print(f"  Adversarial Test Cases:    {val['adversarial_cases']}")
    print(f"  Validation Rejection Rate: {val['rejection_rate'] * 100:.2f}% ({val['correctly_rejected']} / {val['adversarial_cases']})")
    print(f"  False Acceptance Rate:     {val['false_acceptance_rate'] * 100:.2f}% ({val['false_acceptances']} / {val['adversarial_cases']})")
