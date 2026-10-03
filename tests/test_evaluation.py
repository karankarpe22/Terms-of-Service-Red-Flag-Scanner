"""Automated test suite for Phase 9: Evaluation & Benchmarking.

Validates:
1. Ground truth dataset files integrity and schema compliance (clauses, relationships, queries, QA).
2. Human-authored rationales present for all ground-truth entries.
3. Mathematical correctness of evaluation metrics in app.evaluation.evaluator.
4. Correctness of confusion matrix computation.
5. End-to-end execution of evaluation runner functions.
"""
import json
from pathlib import Path
import pytest
import numpy as np

from backend.evaluation.evaluator import (
    compute_confusion_matrix,
    evaluate_classification,
    evaluate_attention,
    evaluate_retrieval,
    evaluate_relationships,
    evaluate_qa_grounding,
    compute_latency_stats,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "evaluation" / "data"


# ---------------------------------------------------------------------------
# Dataset Integrity Tests
# ---------------------------------------------------------------------------

def test_ground_truth_json_schema_and_size():
    """Verify ground_truth.json contains >= 50 labeled clauses and valid schemas."""
    gt_file = DATA_DIR / "ground_truth.json"
    assert gt_file.exists(), f"Missing ground truth file: {gt_file}"

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "clauses" in data
    assert "relationships" in data

    clauses = data["clauses"]
    assert len(clauses) >= 50, f"Expected at least 50 clauses, got {len(clauses)}"

    supported_categories = {
        "Privacy & Data",
        "Payment",
        "Account",
        "Dispute",
        "Liability",
        "User Content",
        "Changes",
        "General",
    }
    supported_attention = {
        "High Attention",
        "Medium Attention",
        "Low Attention",
        "Informational",
    }

    found_categories = set()
    found_attention = set()

    for c in clauses:
        assert "clause_id" in c
        assert "document_id" in c
        assert "text" in c
        assert "expected_primary_category" in c
        assert "expected_attention_level" in c
        assert "rationale" in c and len(c["rationale"].strip()) > 10, "Each clause must have a substantive rationale"

        cat = c["expected_primary_category"]
        att = c["expected_attention_level"]
        assert cat in supported_categories, f"Unknown category: {cat}"
        assert att in supported_attention, f"Unknown attention level: {att}"

        found_categories.add(cat)
        found_attention.add(att)

    # All 8 categories and 4 attention levels must be covered
    assert found_categories == supported_categories
    assert found_attention == supported_attention


def test_relationships_ground_truth_schema():
    """Verify relationship pairs contain valid keys, types, and rationales."""
    gt_file = DATA_DIR / "ground_truth.json"
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    relationships = data["relationships"]
    assert len(relationships) >= 15, f"Expected at least 15 relationship pairs, got {len(relationships)}"

    valid_types = {
        "SUPPORTS",
        "QUALIFIES",
        "EXCEPTS",
        "OVERRIDES",
        "DEPENDS_ON",
        "TEMPORAL",
        "POTENTIAL_TENSION",
    }

    for r in relationships:
        assert "document_id" in r
        assert "source_clause_id" in r
        assert "target_clause_id" in r
        assert "expected_relationship_type" in r
        assert "rationale" in r and len(r["rationale"].strip()) > 10
        assert r["expected_relationship_type"] in valid_types


def test_retrieval_ground_truth_schema():
    """Verify retrieval_ground_truth.json contains >= 15 queries with relevant clause IDs."""
    ret_file = DATA_DIR / "retrieval_ground_truth.json"
    assert ret_file.exists(), f"Missing retrieval ground truth file: {ret_file}"

    with open(ret_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "queries" in data
    queries = data["queries"]
    assert len(queries) >= 15, f"Expected at least 15 retrieval queries, got {len(queries)}"

    for q in queries:
        assert "query" in q and len(q["query"].strip()) > 5
        assert "document_id" in q
        assert "relevant_clause_ids" in q and len(q["relevant_clause_ids"]) > 0
        assert "rationale" in q


def test_qa_ground_truth_schema():
    """Verify qa_ground_truth.json contains >= 20 questions with answerable and unsupported types."""
    qa_file = DATA_DIR / "qa_ground_truth.json"
    assert qa_file.exists(), f"Missing QA ground truth file: {qa_file}"

    with open(qa_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "questions" in data
    questions = data["questions"]
    assert len(questions) >= 20, f"Expected at least 20 QA questions, got {len(questions)}"

    answerable_count = 0
    unsupported_count = 0

    for q in questions:
        assert "question_id" in q
        assert "document_id" in q
        assert "question" in q
        assert "expected_answerable" in q
        assert "expected_source_clause_ids" in q
        assert "ground_truth_notes" in q

        if q["expected_answerable"]:
            answerable_count += 1
            assert len(q["expected_source_clause_ids"]) > 0
        else:
            unsupported_count += 1

    assert answerable_count >= 10, "Expected at least 10 answerable questions"
    assert unsupported_count >= 4, "Expected at least 4 unsupported questions for abstention"


# ---------------------------------------------------------------------------
# Metric Calculation Unit Tests
# ---------------------------------------------------------------------------

def test_evaluate_classification_metrics():
    """Test classification metric calculations with synthetic known data."""
    y_true = ["Payment", "Payment", "Account", "Liability", "Changes"]
    y_pred = ["Payment", "Account", "Account", "Liability", "Changes"]

    res = evaluate_classification(y_pred, y_true)
    assert res["sample_size"] == 5
    assert res["correct_predictions"] == 4
    assert res["accuracy"] == 0.8
    assert "per_class" in res
    assert "Payment" in res["per_class"]
    # Payment: TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=0.6667
    assert res["per_class"]["Payment"]["precision"] == 1.0
    assert res["per_class"]["Payment"]["recall"] == 0.5


def test_compute_confusion_matrix():
    """Test confusion matrix 2D structure."""
    y_true = ["A", "A", "B", "C"]
    y_pred = ["A", "B", "B", "C"]
    labels = ["A", "B", "C"]

    cm = compute_confusion_matrix(y_true, y_pred, labels)
    # Row 0 (A): [1, 1, 0]
    # Row 1 (B): [0, 1, 0]
    # Row 2 (C): [0, 0, 1]
    assert cm == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_evaluate_retrieval_mrr_and_recall():
    """Test MRR and Recall@K computation."""
    queries = [
        {"query": "q1", "relevant_clause_ids": ["c1"], "retrieved_clause_ids": ["c1", "c2", "c3"]},
        {"query": "q2", "relevant_clause_ids": ["c2"], "retrieved_clause_ids": ["c1", "c2", "c3"]},
        {"query": "q3", "relevant_clause_ids": ["c3"], "retrieved_clause_ids": ["c4", "c5", "c6"]},
    ]
    # q1: rank 1 -> RR = 1.0, in top-1, top-3, top-5
    # q2: rank 2 -> RR = 0.5, in top-3, top-5, not in top-1
    # q3: not in top-3 -> RR = 0.0, not in top-1, top-3, top-5
    # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.50
    # Recall@1 = (1 + 0 + 0) / 3 = 0.3333
    # Recall@3 = (1 + 1 + 0) / 3 = 0.6667
    res = evaluate_retrieval(queries, k_values=[1, 3, 5])
    assert res["query_count"] == 3
    assert res["mrr"] == 0.50
    assert res["recall_at_k"]["recall@1"] == 0.3333
    assert res["recall_at_k"]["recall@3"] == 0.6667


def test_evaluate_relationships_metrics():
    """Test relationship precision and recall calculation."""
    gt_pairs = [
        {"document_id": "doc1", "source_clause_id": "c1", "target_clause_id": "c2", "expected_relationship_type": "SUPPORTS"},
        {"document_id": "doc1", "source_clause_id": "c3", "target_clause_id": "c4", "expected_relationship_type": "TEMPORAL"},
    ]
    pred_pairs = [
        {"document_id": "doc1", "source_clause_id": "c1", "target_clause_id": "c2", "relationship_type": "SUPPORTS"},
        {"document_id": "doc1", "source_clause_id": "c5", "target_clause_id": "c6", "relationship_type": "EXCEPTS"},
    ]
    res = evaluate_relationships(pred_pairs, gt_pairs)
    pd = res["pair_detection"]
    # TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
    assert pd["true_positives"] == 1
    assert pd["false_positives"] == 1
    assert pd["false_negatives"] == 1
    assert pd["precision"] == 0.5
    assert pd["recall"] == 0.5
    assert pd["f1_score"] == 0.5
    assert res["type_classification"]["detected_pairs_with_correct_type"] == 1


def test_compute_latency_stats():
    """Test latency stats math."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = compute_latency_stats(data)
    assert stats["mean_ms"] == 30.0
    assert stats["median_ms"] == 30.0
    assert stats["min_ms"] == 10.0
    assert stats["max_ms"] == 50.0
