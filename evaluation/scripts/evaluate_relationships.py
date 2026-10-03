"""Evaluation script for Phase 5 Cross-Clause Relationship Detection Engine.

Evaluates the deterministic relationship detection engine against manually labeled
cross-clause pairs across 7 canonical relationship types:
- SUPPORTS
- QUALIFIES
- EXCEPTS
- OVERRIDES
- DEPENDS_ON
- TEMPORAL
- POTENTIAL_TENSION (textual and structural tension, NOT legal contradiction)

Calculates:
- Overall Pair Detection Precision, Recall, F1-score
- Type Classification Accuracy on detected pairs
- Per-relationship-type Precision, Recall, F1-score, and Support
- Specific False Positive and False Negative examples for error analysis
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

from backend.document.cleaner import clean_text
from backend.document.segmenter import segment_document
from backend.document.models import Document
from backend.document.store import save_document
from backend.analysis.classifier import get_classifier
from backend.analysis.attention import assign_attention_levels
from backend.analysis.relationships import detect_relationships
from backend.evaluation.evaluator import evaluate_relationships

EVAL_DOC_IDS = [
    "doc_saas_cloud",
    "doc_ecommerce_marketplace",
    "doc_social_media",
    "doc_developer_api",
]


def run_relationship_evaluation(
    ground_truth_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute cross-clause relationship evaluation against ground truth."""
    gt_file = ground_truth_path or (BASE_DIR / "evaluation" / "data" / "ground_truth.json")
    docs_dir = BASE_DIR / "evaluation" / "data" / "documents"

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_relationships = data["relationships"]
    classifier = get_classifier()

    all_predicted_pairs: List[Dict[str, Any]] = []
    doc_relationships_map: Dict[str, List[Any]] = {}

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
        classifier.classify_clauses(clauses)
        assign_attention_levels(clauses)
        save_document(doc)

        # Run relationship analysis
        detected_rels, _ = detect_relationships(clauses, document_id=doc_id)
        doc_relationships_map[doc_id] = detected_rels

        for rel in detected_rels:
            all_predicted_pairs.append({
                "document_id": doc_id,
                "source_clause_id": rel.source_clause_id,
                "target_clause_id": rel.target_clause_id,
                "relationship_type": rel.relationship_type,
                "confidence": rel.confidence,
                "rationale": rel.rationale,
            })

    # Evaluate using evaluator utility
    eval_res = evaluate_relationships(
        predicted_pairs=all_predicted_pairs,
        ground_truth_pairs=gt_relationships,
    )

    eval_res["evaluation_note"] = (
        "The relationship engine detects structural and textual relationships "
        "and potential tensions between clauses in the same document. It does NOT "
        "determine legal enforceability or formal legal contradiction."
    )

    return eval_res


if __name__ == "__main__":
    res = run_relationship_evaluation()
    print("=== CROSS-CLAUSE RELATIONSHIP EVALUATION ===")
    print(f"Ground Truth Pairs: {res['ground_truth_count']}")
    print(f"Predicted Pairs:    {res['predicted_count']}")
    print("\nPair Detection Metrics:")
    pd = res["pair_detection"]
    print(f"  Precision: {pd['precision'] * 100:.2f}%")
    print(f"  Recall:    {pd['recall'] * 100:.2f}%")
    print(f"  F1 Score:  {pd['f1_score'] * 100:.2f}%")
    print(f"  TP: {pd['true_positives']} | FP: {pd['false_positives']} | FN: {pd['false_negatives']}")

    print("\nType Classification on Detected Pairs:")
    tc = res["type_classification"]
    print(f"  Correct Type Matches: {tc['detected_pairs_with_correct_type']}")
    print(f"  Type Accuracy:        {tc['type_accuracy_on_detected'] * 100:.2f}%")

    print("\nPer-Type Breakdown:")
    for t, m in res["per_type"].items():
        if m["support"] > 0 or m["true_positives"] > 0 or m["false_positives"] > 0:
            print(f"  {t:<18} | Prec: {m['precision']:.2f} | Rec: {m['recall']:.2f} | F1: {m['f1_score']:.2f} | Support: {m['support']}")
