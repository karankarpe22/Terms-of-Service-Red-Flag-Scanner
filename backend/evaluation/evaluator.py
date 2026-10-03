"""Metric calculation and evaluation utility module.

Computes exact evaluation metrics across:
- Clause classification (Accuracy, Macro/Per-class Precision, Recall, F1, Confusion Matrix)
- Attention level detection (Accuracy, Macro/Per-class Precision, Recall, F1, Confusion Matrix)
- Semantic retrieval (Recall@K, Precision@K, MRR, threshold sensitivity sweep)
- Cross-clause relationship detection (Precision, Recall, F1, FP/FN breakdown)
- Q&A grounding & citation accuracy (Citation Precision/Recall, Abstention, Hallucination resistance)
- Latency statistics (Mean, Median, Min, Max)

Academic reporting guidelines:
- Zero fabrication: metrics are strictly computed from actual inputs and ground truths.
- All metrics are returned in structured dictionaries for JSON serialization.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> List[List[int]]:
    """Compute confusion matrix where rows are ground truth and columns are predictions."""
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1
    return matrix


def evaluate_classification(
    predictions: List[str],
    ground_truth: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate multi-class classification metrics: Accuracy, Macro/Per-class Precision, Recall, F1.

    Args:
        predictions: List of predicted category strings.
        ground_truth: List of expected ground-truth category strings.
        labels: Optional explicit list of target classes.

    Returns:
        Dictionary containing overall accuracy, macro metrics, per-class metrics, and confusion matrix.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Length of predictions and ground_truth must match.")

    total = len(ground_truth)
    if total == 0:
        return {
            "sample_size": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": {"labels": [], "matrix": []},
        }

    # Determine unique classes if not supplied
    if labels is None:
        unique_labels = sorted(list(set(ground_truth) | set(predictions)))
    else:
        unique_labels = list(labels)

    # Accuracy
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = float(correct / total)

    # Per-class TP, FP, FN, Support
    per_class: Dict[str, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in unique_labels:
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
        support = sum(1 for g in ground_truth if g == label)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

        # Include all classes present in ground truth in macro average
        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    cm = compute_confusion_matrix(ground_truth, predictions, unique_labels)

    return {
        "sample_size": total,
        "correct_predictions": correct,
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": unique_labels,
            "matrix": cm,
        },
    }


def evaluate_attention(
    predictions: List[str],
    ground_truth: List[str],
    levels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate attention level detection metrics and confusion matrix.

    Attention level represents a deterministic user-attention heuristic, NOT a legal-risk score.
    """
    default_levels = [
        "High Attention",
        "Medium Attention",
        "Low Attention",
        "Informational",
    ]
    labels = levels or default_levels
    metrics = evaluate_classification(predictions, ground_truth, labels=labels)
    metrics["heuristic_note"] = (
        "Attention level reflects deterministic contractual indicators (e.g., auto-renewal, "
        "arbitration, liability disclaimers) requiring closer human review, NOT legal enforceability."
    )
    return metrics


def evaluate_retrieval(
    queries_eval: List[Dict[str, Any]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Calculate retrieval metrics: Recall@K, Precision@K, and Mean Reciprocal Rank (MRR).

    Args:
        queries_eval: List of dicts, each with:
            - 'query': str
            - 'relevant_clause_ids': List[str] (ground truth)
            - 'retrieved_clause_ids': List[str] (ordered by rank)

    Returns:
        Dictionary with Recall@1, Recall@3, Recall@5, MRR, and sample count.
    """
    k_values = k_values or [1, 3, 5]
    if not queries_eval:
        return {
            "query_count": 0,
            "mrr": 0.0,
            "recall_at_k": {f"recall@{k}": 0.0 for k in k_values},
            "precision_at_k": {f"precision@{k}": 0.0 for k in k_values},
        }

    recalls_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
    precisions_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
    reciprocal_ranks: List[float] = []

    for item in queries_eval:
        relevant = set(item.get("relevant_clause_ids", []))
        retrieved = item.get("retrieved_clause_ids", [])

        # If no relevant clauses exist in ground truth, skip or handle
        if not relevant:
            continue

        # MRR calculation
        rr = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Recall@K and Precision@K
        for k in k_values:
            top_k = retrieved[:k]
            hits = sum(1 for cid in top_k if cid in relevant)
            rec = hits / len(relevant)
            prec = hits / k if k > 0 else 0.0
            recalls_at_k[k].append(rec)
            precisions_at_k[k].append(prec)

    res: Dict[str, Any] = {
        "query_count": len(queries_eval),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0,
        "recall_at_k": {
            f"recall@{k}": round(float(np.mean(recalls_at_k[k])), 4) if recalls_at_k[k] else 0.0
            for k in k_values
        },
        "precision_at_k": {
            f"precision@{k}": round(float(np.mean(precisions_at_k[k])), 4) if precisions_at_k[k] else 0.0
            for k in k_values
        },
    }
    return res


def evaluate_relationships(
    predicted_pairs: List[Dict[str, Any]],
    ground_truth_pairs: List[Dict[str, Any]],
    relationship_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate cross-clause relationship detection Precision, Recall, and F1-score.

    A relationship match requires matching (document_id, source_clause_id, target_clause_id).
    Type agreement is also evaluated.
    """
    default_types = [
        "SUPPORTS",
        "QUALIFIES",
        "EXCEPTS",
        "OVERRIDES",
        "DEPENDS_ON",
        "TEMPORAL",
        "POTENTIAL_TENSION",
    ]
    types = relationship_types or default_types

    def pair_key(p: Dict[str, Any]) -> Tuple[str, str, str]:
        return (
            p.get("document_id", ""),
            p.get("source_clause_id", ""),
            p.get("target_clause_id", ""),
        )

    gt_map = {pair_key(p): p.get("expected_relationship_type") or p.get("relationship_type") for p in ground_truth_pairs}
    pred_map = {pair_key(p): p.get("relationship_type") for p in predicted_pairs}

    gt_keys = set(gt_map.keys())
    pred_keys = set(pred_map.keys())

    # Overall Pair Detection
    tp_pairs = pred_keys & gt_keys
    fp_pairs = pred_keys - gt_keys
    fn_pairs = gt_keys - pred_keys

    pair_prec = len(tp_pairs) / len(pred_keys) if pred_keys else 0.0
    pair_rec = len(tp_pairs) / len(gt_keys) if gt_keys else 0.0
    pair_f1 = (2 * pair_prec * pair_rec / (pair_prec + pair_rec)) if (pair_prec + pair_rec) > 0 else 0.0

    # Type-aware exact matches
    type_matches = sum(1 for k in tp_pairs if pred_map[k] == gt_map[k])
    type_accuracy_on_detected = (type_matches / len(tp_pairs)) if tp_pairs else 0.0

    # Per-relationship-type metrics
    per_type: Dict[str, Dict[str, Any]] = {}
    for r_type in types:
        gt_type_keys = {k for k, v in gt_map.items() if v == r_type}
        pred_type_keys = {k for k, v in pred_map.items() if v == r_type}

        tp = len(pred_type_keys & gt_type_keys)
        fp = len(pred_type_keys - gt_type_keys)
        fn = len(gt_type_keys - pred_type_keys)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_type[r_type] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": len(gt_type_keys),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    return {
        "ground_truth_count": len(gt_keys),
        "predicted_count": len(pred_keys),
        "pair_detection": {
            "precision": round(pair_prec, 4),
            "recall": round(pair_rec, 4),
            "f1_score": round(pair_f1, 4),
            "true_positives": len(tp_pairs),
            "false_positives": len(fp_pairs),
            "false_negatives": len(fn_pairs),
        },
        "type_classification": {
            "detected_pairs_with_correct_type": type_matches,
            "type_accuracy_on_detected": round(type_accuracy_on_detected, 4),
        },
        "per_type": per_type,
        "false_positive_examples": [
            {"source": k[1], "target": k[2], "doc": k[0], "predicted_type": pred_map[k]}
            for k in list(fp_pairs)[:5]
        ],
        "false_negative_examples": [
            {"source": k[1], "target": k[2], "doc": k[0], "expected_type": gt_map[k]}
            for k in list(fn_pairs)[:5]
        ],
    }


def evaluate_qa_grounding(
    qa_eval_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate Q&A citation accuracy, abstention precision/recall, and hallucination rates.

    Args:
        qa_eval_results: List of evaluated question results with fields:
            - 'question_id': str
            - 'expected_answerable': bool
            - 'system_status': str ('success', 'insufficient_evidence', 'validation_failed')
            - 'expected_source_clause_ids': List[str]
            - 'cited_clause_ids': List[str]
            - 'has_unsupported_claim': bool
            - 'has_fabricated_citation': bool

    Returns:
        Structured dictionary of citation, abstention, and hallucination metrics.
    """
    total_q = len(qa_eval_results)
    if total_q == 0:
        return {}

    # 1. Abstention Evaluation
    # Ground truth: unanswerable / unsupported questions
    unsupported_items = [r for r in qa_eval_results if not r.get("expected_answerable", True)]
    answerable_items = [r for r in qa_eval_results if r.get("expected_answerable", True)]

    # System abstains when status == "insufficient_evidence" or answer is None
    abstained_items = [
        r for r in qa_eval_results
        if r.get("system_status") == "insufficient_evidence" or r.get("answer") is None
    ]

    tp_abstain = sum(1 for r in unsupported_items if r.get("system_status") == "insufficient_evidence" or r.get("answer") is None)
    fp_abstain = sum(1 for r in answerable_items if r.get("system_status") == "insufficient_evidence" or r.get("answer") is None)
    fn_abstain = len(unsupported_items) - tp_abstain

    abstain_prec = tp_abstain / len(abstained_items) if abstained_items else 0.0
    abstain_rec = tp_abstain / len(unsupported_items) if unsupported_items else 0.0
    abstain_f1 = (2 * abstain_prec * abstain_rec / (abstain_prec + abstain_rec)) if (abstain_prec + abstain_rec) > 0 else 0.0

    # 2. Citation Accuracy (Evaluated on Answerable Questions that were answered)
    citation_precisions: List[float] = []
    citation_recalls: List[float] = []
    exact_citation_matches = 0

    answered_items = [r for r in answerable_items if r.get("answer") is not None]

    for item in answered_items:
        expected_sources = set(item.get("expected_source_clause_ids", []))
        cited_sources = set(item.get("cited_clause_ids", []))

        if not expected_sources and not cited_sources:
            continue

        tp = len(cited_sources & expected_sources)
        prec = tp / len(cited_sources) if cited_sources else 0.0
        rec = tp / len(expected_sources) if expected_sources else 0.0

        citation_precisions.append(prec)
        citation_recalls.append(rec)

        if cited_sources == expected_sources or (expected_sources.issubset(cited_sources)):
            exact_citation_matches += 1

    mean_citation_prec = float(np.mean(citation_precisions)) if citation_precisions else 0.0
    mean_citation_rec = float(np.mean(citation_recalls)) if citation_recalls else 0.0
    citation_acc = exact_citation_matches / len(answered_items) if answered_items else 0.0

    # 3. Hallucination Resistance
    unsupported_claim_count = sum(1 for r in qa_eval_results if r.get("has_unsupported_claim", False))
    fabricated_citation_count = sum(1 for r in qa_eval_results if r.get("has_fabricated_citation", False))

    unsupported_claim_rate = unsupported_claim_count / total_q
    fabricated_citation_rate = fabricated_citation_count / total_q

    return {
        "total_questions": total_q,
        "answerable_questions": len(answerable_items),
        "unsupported_questions": len(unsupported_items),
        "abstention_metrics": {
            "unsupported_total": len(unsupported_items),
            "correctly_abstained": tp_abstain,
            "incorrectly_answered": fn_abstain,
            "false_abstentions": fp_abstain,
            "abstention_precision": round(abstain_prec, 4),
            "abstention_recall": round(abstain_rec, 4),
            "abstention_f1": round(abstain_f1, 4),
            "abstention_rate_on_unsupported": round(abstain_rec, 4),
        },
        "citation_metrics": {
            "answered_questions_evaluated": len(answered_items),
            "citation_precision": round(mean_citation_prec, 4),
            "citation_recall": round(mean_citation_rec, 4),
            "citation_accuracy": round(citation_acc, 4),
        },
        "hallucination_metrics": {
            "unsupported_claim_count": unsupported_claim_count,
            "unsupported_claim_rate": round(unsupported_claim_rate, 4),
            "fabricated_citation_count": fabricated_citation_count,
            "fabricated_citation_rate": round(fabricated_citation_rate, 4),
        },
    }


def compute_latency_stats(latencies_ms: List[float]) -> Dict[str, float]:
    """Calculate mean, median, min, max, p95 from latency observations in milliseconds."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "median_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "p95_ms": 0.0}

    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
    }
