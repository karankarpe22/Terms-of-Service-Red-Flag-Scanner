"""Evaluation script for Phase 3 Clause Classification.

Evaluates the hybrid classifier (keyword rules + SentenceTransformers all-MiniLM-L6-v2)
against the manually annotated ground-truth dataset across all 8 categories:
1. Privacy & Data
2. Payment
3. Account
4. Dispute
5. Liability
6. User Content
7. Changes
8. General

Calculates:
- Overall Accuracy
- Macro-averaged Precision, Recall, F1-score
- Per-category Precision, Recall, F1-score, Support, TP, FP, FN
- Multi-class Confusion Matrix (saved as text table and PNG plot)
- Class imbalance explanation
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.analysis.classifier import get_classifier
from backend.document.models import Clause
from backend.evaluation.evaluator import evaluate_classification


def format_confusion_matrix_text(cm_dict: Dict[str, Any]) -> str:
    """Format confusion matrix as an aligned, human-readable ASCII table."""
    labels = cm_dict["labels"]
    matrix = cm_dict["matrix"]

    col_width = max(max(len(l) for l in labels), 12) + 2
    header = f"{'True \\ Pred':<{col_width}}" + "".join(f"{l:>{col_width}}" for l in labels) + f"{'Total':>{col_width}}\n"
    divider = "-" * len(header) + "\n"

    rows = []
    for i, true_label in enumerate(labels):
        row_str = f"{true_label:<{col_width}}"
        row_total = sum(matrix[i])
        for j in range(len(labels)):
            row_str += f"{matrix[i][j]:>{col_width}}"
        row_str += f"{row_total:>{col_width}}"
        rows.append(row_str)

    # Column totals (predictions)
    col_totals = [sum(matrix[i][j] for i in range(len(labels))) for j in range(len(labels))]
    total_str = f"{'Pred Total':<{col_width}}" + "".join(f"{t:>{col_width}}" for t in col_totals) + f"{sum(col_totals):>{col_width}}"

    return f"{divider}{header}{divider}" + "\n".join(rows) + f"\n{divider}{total_str}\n{divider}"


def plot_confusion_matrix(cm_dict: Dict[str, Any], output_path: Path) -> None:
    """Render and save confusion matrix as a high-resolution PNG using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        labels = cm_dict["labels"]
        matrix = np.array(cm_dict["matrix"])

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix[i, j]
                ax.text(
                    j, i, str(val),
                    va="center", ha="center",
                    color="white" if val > matrix.max() / 2 else "black",
                    fontweight="bold" if i == j else "normal"
                )

        fig.colorbar(cax)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="left")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Category", labelpad=12, fontweight="bold")
        ax.set_ylabel("True Category", labelpad=12, fontweight="bold")
        ax.set_title("Clause Classification Confusion Matrix", pad=20, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"Warning: Could not generate PNG confusion matrix: {exc}", file=sys.stderr)


def run_classifier_evaluation(
    ground_truth_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute classification evaluation against ground truth clauses."""
    gt_file = ground_truth_path or (BASE_DIR / "evaluation" / "data" / "ground_truth.json")
    out_dir = results_dir or (BASE_DIR / "evaluation" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_clauses = data["clauses"]
    classifier = get_classifier()

    # Reconstruct Clause domain objects
    clauses = [
        Clause(
            clause_id=item["clause_id"],
            document_id=item["document_id"],
            clause_index=item.get("clause_index", idx + 1),
            section_title=item.get("section_title", "General"),
            text=item["text"],
            raw_text=item["text"],
        )
        for idx, item in enumerate(gt_clauses)
    ]

    # Batch classify
    pred_records = classifier.classify_clauses(clauses)
    predictions = [r["primary_category"] for r in pred_records]
    ground_truth = [item["expected_primary_category"] for item in gt_clauses]

    # All 8 supported target categories
    all_categories = [
        "Privacy & Data",
        "Payment",
        "Account",
        "Dispute",
        "Liability",
        "User Content",
        "Changes",
        "General",
    ]

    eval_results = evaluate_classification(
        predictions=predictions,
        ground_truth=ground_truth,
        labels=all_categories,
    )

    # Add error examples for qualitative review
    mismatches = []
    for gt_item, pred_record in zip(gt_clauses, pred_records):
        pred_cat = pred_record["primary_category"]
        expected_cat = gt_item["expected_primary_category"]
        if pred_cat != expected_cat:
            mismatches.append({
                "clause_id": gt_item["clause_id"],
                "document_id": gt_item["document_id"],
                "section_title": gt_item.get("section_title", ""),
                "expected": expected_cat,
                "predicted": pred_cat,
                "confidence": pred_record["category_confidence"],
                "snippet": gt_item["text"][:120] + "...",
                "rationale": gt_item.get("rationale", ""),
            })

    eval_results["mismatches"] = mismatches
    eval_results["class_imbalance_note"] = (
        "Dataset contains 74 clauses distributed relatively evenly across 8 classes "
        "(8-12 clauses per class: Account=12, Privacy & Data=11, Liability=10, Changes=9, "
        "Dispute=8, Payment=8, User Content=8, General=8). Minor variance reflects natural ToS structure."
    )

    # Save text confusion matrix
    cm_text = format_confusion_matrix_text(eval_results["confusion_matrix"])
    (out_dir / "confusion_matrix_classification.txt").write_text(cm_text, encoding="utf-8")

    # Save PNG plot
    plot_confusion_matrix(eval_results["confusion_matrix"], out_dir / "confusion_matrix_classification.png")

    return eval_results


if __name__ == "__main__":
    res = run_classifier_evaluation()
    print("=== CLAUSE CLASSIFICATION EVALUATION ===")
    print(f"Sample Size: {res['sample_size']}")
    print(f"Accuracy:    {res['accuracy'] * 100:.2f}%")
    print(f"Macro Prec:  {res['macro_precision'] * 100:.2f}%")
    print(f"Macro Rec:   {res['macro_recall'] * 100:.2f}%")
    print(f"Macro F1:    {res['macro_f1'] * 100:.2f}%")
    print("\nPer-Category Metrics:")
    for cat, m in res["per_class"].items():
        print(f"  {cat:<16} | Prec: {m['precision']:.2f} | Rec: {m['recall']:.2f} | F1: {m['f1_score']:.2f} | Support: {m['support']}")
    print(f"\nMismatches: {len(res['mismatches'])}")
