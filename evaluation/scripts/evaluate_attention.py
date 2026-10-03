"""Evaluation script for Attention Level Detection.

Evaluates the deterministic attention level detector against ground-truth annotations across:
- High Attention
- Medium Attention
- Low Attention
- Informational

Calculates:
- Overall Accuracy
- Macro-averaged Precision, Recall, F1-score
- Per-level Precision, Recall, F1-score, Support, TP, FP, FN
- Attention Confusion Matrix (saved as text table and PNG plot)
- Heuristic transparency explanation (NOT a legal-risk score)
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

from backend.analysis.attention import detect_attention_level
from backend.evaluation.evaluator import evaluate_attention


def format_confusion_matrix_text(cm_dict: Dict[str, Any]) -> str:
    """Format confusion matrix as an aligned, human-readable ASCII table."""
    labels = cm_dict["labels"]
    matrix = cm_dict["matrix"]

    col_width = max(max(len(l) for l in labels), 15) + 2
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

        fig, ax = plt.subplots(figsize=(8, 6))
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
        ax.set_xticklabels(labels, rotation=30, ha="left")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Attention Level", labelpad=12, fontweight="bold")
        ax.set_ylabel("True Attention Level", labelpad=12, fontweight="bold")
        ax.set_title("Attention Level Confusion Matrix", pad=20, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"Warning: Could not generate PNG attention confusion matrix: {exc}", file=sys.stderr)


def run_attention_evaluation(
    ground_truth_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute attention level evaluation against ground truth clauses."""
    gt_file = ground_truth_path or (BASE_DIR / "evaluation" / "data" / "ground_truth.json")
    out_dir = results_dir or (BASE_DIR / "evaluation" / "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_clauses = data["clauses"]

    predictions = []
    predicted_reasons = []
    for item in gt_clauses:
        level, reasons = detect_attention_level(item["text"])
        predictions.append(level)
        predicted_reasons.append(reasons)

    ground_truth = [item["expected_attention_level"] for item in gt_clauses]

    levels = [
        "High Attention",
        "Medium Attention",
        "Low Attention",
        "Informational",
    ]

    eval_results = evaluate_attention(
        predictions=predictions,
        ground_truth=ground_truth,
        levels=levels,
    )

    # Collect mismatches for qualitative error analysis
    mismatches = []
    for gt_item, pred_lvl, reasons in zip(gt_clauses, predictions, predicted_reasons):
        exp_lvl = gt_item["expected_attention_level"]
        if pred_lvl != exp_lvl:
            mismatches.append({
                "clause_id": gt_item["clause_id"],
                "document_id": gt_item["document_id"],
                "section_title": gt_item.get("section_title", ""),
                "expected": exp_lvl,
                "predicted": pred_lvl,
                "reasons": reasons,
                "snippet": gt_item["text"][:120] + "...",
                "rationale": gt_item.get("rationale", ""),
            })

    eval_results["mismatches"] = mismatches

    # Save text confusion matrix
    cm_text = format_confusion_matrix_text(eval_results["confusion_matrix"])
    (out_dir / "confusion_matrix_attention.txt").write_text(cm_text, encoding="utf-8")

    # Save PNG plot
    plot_confusion_matrix(eval_results["confusion_matrix"], out_dir / "confusion_matrix_attention.png")

    return eval_results


if __name__ == "__main__":
    res = run_attention_evaluation()
    print("=== ATTENTION LEVEL EVALUATION ===")
    print(f"Sample Size: {res['sample_size']}")
    print(f"Accuracy:    {res['accuracy'] * 100:.2f}%")
    print(f"Macro Prec:  {res['macro_precision'] * 100:.2f}%")
    print(f"Macro Rec:   {res['macro_recall'] * 100:.2f}%")
    print(f"Macro F1:    {res['macro_f1'] * 100:.2f}%")
    print("\nPer-Level Metrics:")
    for lvl, m in res["per_class"].items():
        print(f"  {lvl:<16} | Prec: {m['precision']:.2f} | Rec: {m['recall']:.2f} | F1: {m['f1_score']:.2f} | Support: {m['support']}")
    print(f"\nMismatches: {len(res['mismatches'])}")
