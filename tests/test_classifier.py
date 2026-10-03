import pytest
from backend.document.models import Clause, SourceLocation
from backend.analysis.classifier import classify_clause, classify_clauses, get_classifier
from backend.core.constants import SUPPORTED_CATEGORIES


def create_sample_clause(text: str, clause_id: str = "test_c1") -> Clause:
    return Clause(
        clause_id=clause_id,
        document_id="doc_test",
        clause_index=1,
        section_title="Test Section",
        text=text,
        source_location=SourceLocation(source_type="text"),
    )


def test_classifier_all_seven_categories_with_keywords():
    """Verify high-precision classification across all 7 supported PRD categories."""
    samples = {
        "Privacy & Data": "We collect personal data and cookies, and share information with third parties.",
        "Payment": "All subscription fees are non-refundable and auto-renew every billing cycle.",
        "Account": "We reserve the right to suspend or terminate your account for unauthorized access.",
        "Dispute": "All legal disputes shall be resolved through mandatory binding individual arbitration.",
        "Liability": "In no event shall aggregate liability exceed $50, and we disclaim all warranties.",
        "User Content": "You grant us a perpetual, irrevocable, royalty-free license to use your user content.",
        "Changes": "We reserve the right to modify these terms at any time without prior notice.",
    }

    for expected_cat, text in samples.items():
        clause = create_sample_clause(text, clause_id=f"c_{expected_cat}")
        result = classify_clause(clause)

        assert result["primary_category"] == expected_cat, (
            f"Expected {expected_cat} for text '{text}', but got {result['primary_category']}"
        )
        assert 0.0 <= result["category_confidence"] <= 1.0
        assert clause.primary_category == expected_cat
        assert clause.category == expected_cat


def test_classifier_semantic_similarity_without_exact_keywords():
    """Verify that semantic similarity correctly classifies text with non-keyword paraphrasing."""
    # Paraphrased Privacy (gathers browsing trails, partner corporations)
    c1 = create_sample_clause("We gather personal browsing trails and distribute them to partner corporations.")
    r1 = classify_clause(c1)
    assert r1["primary_category"] == "Privacy & Data"

    # Paraphrased Payment (monies remitted, unrecoverable, refresh cycle)
    c2 = create_sample_clause("Monies remitted for access to our digital tools are unrecoverable and refresh each cycle.")
    r2 = classify_clause(c2)
    assert r2["primary_category"] == "Payment"

    # Paraphrased Liability (financial damages capped at zero)
    c3 = create_sample_clause("Our financial compensation for damages is strictly capped at zero dollars.")
    r3 = classify_clause(c3)
    assert r3["primary_category"] == "Liability"


def test_classifier_multi_category_detection():
    """Verify that a clause containing multiple category concepts populates secondary_categories."""
    # Combines recurring subscription billing (Payment) with unilateral terms modification (Changes)
    text = (
        "Subscription fees automatically renew each month, and we reserve the right "
        "to modify these terms and change pricing at any time without notice."
    )
    clause = create_sample_clause(text)
    result = classify_clause(clause)

    assert result["primary_category"] in ("Payment", "Changes")
    assert any(cat in ("Payment", "Changes") for cat in result["secondary_categories"])


def test_classifier_general_unsupported_category():
    """Verify that neutral or irrelevant text falls back to 'General' with low confidence."""
    text = "Welcome to our website. Feel free to contact our help desk with any inquiries."
    clause = create_sample_clause(text)
    result = classify_clause(clause)

    assert result["primary_category"] == "General"
    assert result["secondary_categories"] == []


def test_classifier_batch_output_structure():
    """Verify batch classification updates all Clause instances and returns structured records."""
    clauses = [
        create_sample_clause("Payment is due upon invoice and subscriptions renew automatically.", "c_1"),
        create_sample_clause("We may modify these terms at any time.", "c_2"),
    ]
    results = classify_clauses(clauses)

    assert len(results) == 2
    for res in results:
        assert "clause_id" in res
        assert "primary_category" in res
        assert "secondary_categories" in res
        assert "category_confidence" in res
        assert isinstance(res["category_confidence"], float)
