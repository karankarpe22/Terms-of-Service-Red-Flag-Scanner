"""Evaluation script for Phase 4 FAISS Semantic Retrieval.

Evaluates retrieval quality against manually labeled retrieval queries across multiple
similarity thresholds: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55].

Calculates:
- Recall@1, Recall@3, Recall@5
- Mean Reciprocal Rank (MRR)
- Precision@1, Precision@3, Precision@5
- Evidence sufficiency / Abstention rate per threshold
- Threshold sensitivity analysis table (preserving 0.45 as engineering baseline)
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
from backend.retrieval.vector_store import get_vector_store
from backend.evaluation.evaluator import evaluate_retrieval

EVAL_DOC_IDS = [
    "doc_saas_cloud",
    "doc_ecommerce_marketplace",
    "doc_social_media",
    "doc_developer_api",
]

TEST_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]


def setup_and_index_documents() -> None:
    """Ensure all 4 evaluation documents are segmented and indexed in FAISS."""
    docs_dir = BASE_DIR / "evaluation" / "data" / "documents"
    vector_store = get_vector_store()

    for doc_id in EVAL_DOC_IDS:
        txt_path = docs_dir / f"{doc_id}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing evaluation document: {txt_path}")

        raw_text = txt_path.read_text(encoding="utf-8")
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


def run_retrieval_evaluation(
    retrieval_gt_path: Optional[Path] = None,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Execute retrieval evaluation over all queries and test thresholds."""
    gt_file = retrieval_gt_path or (BASE_DIR / "evaluation" / "data" / "retrieval_ground_truth.json")
    threshold_list = thresholds or TEST_THRESHOLDS

    # Ensure evaluation indices are ready
    setup_and_index_documents()

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    vector_store = get_vector_store()

    threshold_comparisons: Dict[str, Any] = {}
    query_details_by_threshold: Dict[str, List[Dict[str, Any]]] = {}

    for thresh in threshold_list:
        thresh_key = f"{thresh:.2f}"
        query_eval_records = []
        sufficient_count = 0

        for q in queries:
            doc_id = q["document_id"]
            query_str = q["query"]
            relevant_ids = q["relevant_clause_ids"]

            # Perform FAISS search at this threshold
            search_res = vector_store.search(
                document_id=doc_id,
                query=query_str,
                top_k=5,
                threshold=thresh,
            )

            # Retrieved clauses that meet threshold
            retrieved_ids = [r.clause_id for r in search_res.results if r.meets_threshold]
            if search_res.sufficient_evidence:
                sufficient_count += 1

            query_eval_records.append({
                "query": query_str,
                "document_id": doc_id,
                "relevant_clause_ids": relevant_ids,
                "retrieved_clause_ids": retrieved_ids,
                "sufficient_evidence": search_res.sufficient_evidence,
                "top_similarity": search_res.results[0].similarity_score if search_res.results else 0.0,
            })

        # Calculate standard metrics
        metrics = evaluate_retrieval(query_eval_records, k_values=[1, 3, 5])
        metrics["threshold"] = thresh
        metrics["sufficient_evidence_rate"] = round(sufficient_count / len(queries), 4)
        metrics["abstention_rate"] = round(1.0 - (sufficient_count / len(queries)), 4)

        threshold_comparisons[thresh_key] = metrics
        query_details_by_threshold[thresh_key] = query_eval_records

    # Identify retrieval misses at baseline threshold (0.45)
    baseline_records = query_details_by_threshold.get("0.45", [])
    retrieval_misses = []
    for r in baseline_records:
        rel = set(r["relevant_clause_ids"])
        ret = set(r["retrieved_clause_ids"][:5])
        if not (rel & ret):
            retrieval_misses.append({
                "query": r["query"],
                "document_id": r["document_id"],
                "expected": list(rel),
                "top_similarity": r["top_similarity"],
                "sufficient_evidence": r["sufficient_evidence"],
            })

    return {
        "query_count": len(queries),
        "baseline_threshold": 0.45,
        "tested_thresholds": threshold_list,
        "threshold_comparisons": threshold_comparisons,
        "baseline_metrics": threshold_comparisons.get("0.45", {}),
        "retrieval_misses_at_baseline": retrieval_misses,
        "interpretation": (
            "Comparing retrieval metrics across thresholds reveals the trade-off between recall and "
            "abstention. At lower thresholds (e.g. 0.30-0.35), recall is high but low-similarity matches "
            "are accepted. At higher thresholds (e.g. 0.50-0.55), precision improves but abstention increases."
        ),
    }


if __name__ == "__main__":
    res = run_retrieval_evaluation()
    print("=== FAISS RETRIEVAL EVALUATION ===")
    print(f"Total Queries: {res['query_count']}")
    print("\nThreshold Sensitivity Comparison:")
    print(f"{'Threshold':<10} | {'Recall@1':<10} | {'Recall@3':<10} | {'Recall@5':<10} | {'MRR':<8} | {'Suff. Rate':<10} | {'Abstain Rate':<12}")
    print("-" * 78)
    for t_str, m in res["threshold_comparisons"].items():
        print(
            f"{t_str:<10} | "
            f"{m['recall_at_k']['recall@1']:<10.2f} | "
            f"{m['recall_at_k']['recall@3']:<10.2f} | "
            f"{m['recall_at_k']['recall@5']:<10.2f} | "
            f"{m['mrr']:<8.2f} | "
            f"{m['sufficient_evidence_rate']:<10.2f} | "
            f"{m['abstention_rate']:<12.2f}"
        )
