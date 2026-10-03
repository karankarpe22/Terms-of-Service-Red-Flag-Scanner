import pytest
from backend.document.models import Clause, SourceLocation
from backend.analysis.attention import (
    detect_attention_level,
    assign_attention_levels,
)
from backend.core.constants import (
    ATTENTION_LEVEL_HIGH,
    ATTENTION_LEVEL_MEDIUM,
    ATTENTION_LEVEL_LOW,
    ATTENTION_LEVEL_INFORMATIONAL,
)


def create_sample_clause(text: str, clause_id: str = "att_c1") -> Clause:
    return Clause(
        clause_id=clause_id,
        document_id="doc_att_test",
        clause_index=1,
        section_title="Attention Test",
        text=text,
        source_location=SourceLocation(source_type="text"),
    )


def test_attention_high_multiple_strong_indicators():
    """Verify that multiple strong indicators trigger High Attention."""
    # 1. Payment: Auto-renew + non-refundable + 30 days cancellation deadline
    p1 = (
        "Your subscription will automatically renew at the end of each billing cycle. "
        "All subscription fees are non-refundable unless cancelled at least thirty (30) days prior."
    )
    lvl1, r1 = detect_attention_level(p1)
    assert lvl1 == ATTENTION_LEVEL_HIGH
    assert len(r1) >= 2

    # 2. Dispute: Mandatory arbitration + class-action waiver
    p2 = (
        "Any dispute shall be resolved through final and binding individual arbitration. "
        "You expressly waive any right to participate in a class action lawsuit."
    )
    lvl2, r2 = detect_attention_level(p2)
    assert lvl2 == ATTENTION_LEVEL_HIGH
    assert len(r2) >= 2

    # 3. User Content: Perpetual irrevocable license + right to sublicense/distribute
    p3 = (
        "You grant us a perpetual, irrevocable, royalty-free license to use, reproduce, "
        "and sublicensable right to commercialize your user content in any media."
    )
    lvl3, r3 = detect_attention_level(p3)
    assert lvl3 == ATTENTION_LEVEL_HIGH

    # 4. Changes: Unilateral modification without notice + deemed acceptance
    p4 = (
        "We reserve the right to modify these terms at any time without notice. "
        "Your continued use of the service constitutes binding acceptance of all changes."
    )
    lvl4, r4 = detect_attention_level(p4)
    assert lvl4 == ATTENTION_LEVEL_HIGH

    # 5. Account: Terminate without notice + without liability
    p5 = (
        "We reserve the right to terminate your account immediately without notice "
        "and without any liability for any reason whatsoever."
    )
    lvl5, r5 = detect_attention_level(p5)
    assert lvl5 == ATTENTION_LEVEL_HIGH


def test_attention_medium_single_strong_indicator():
    """Verify that a single strong indicator triggers Medium Attention."""
    # Only arbitration, without class action waiver
    text = "Any controversy arising out of these terms shall be settled by binding arbitration."
    lvl, reasons = detect_attention_level(text)
    assert lvl == ATTENTION_LEVEL_MEDIUM
    assert len(reasons) == 1

    # Only limitation of liability cap
    text2 = "In no event shall aggregate liability exceed $50."
    lvl2, reasons2 = detect_attention_level(text2)
    assert lvl2 == ATTENTION_LEVEL_MEDIUM
    assert len(reasons2) == 1


def test_attention_low_weak_indicator():
    """Verify that general or weak indicators trigger Low Attention."""
    text = "The platform is provided on an as-is and as-available basis without warranty of any kind."
    lvl, reasons = detect_attention_level(text)
    assert lvl == ATTENTION_LEVEL_LOW
    assert len(reasons) >= 1


def test_attention_informational_ordinary_text():
    """Verify that standard informational text receives Informational attention level."""
    text = (
        "By accessing or using our service, you agree to be bound by these Terms of Service. "
        "If you do not agree to these terms, please do not use the website."
    )
    lvl, reasons = detect_attention_level(text)
    assert lvl == ATTENTION_LEVEL_INFORMATIONAL
    assert "Standard informational clause" in reasons[0]


def test_no_legal_conclusions_in_attention_reasons():
    """Verify non-negotiable rule: reasons must never state that a clause is illegal or unenforceable."""
    prohibited_terms = [
        "illegal",
        "unlawful",
        "legally invalid",
        "unenforceable",
        "legally risky",
        "legally compliant",
    ]

    all_test_clauses = [
        "We may terminate your account at any time without notice or liability.",
        "You waive all rights to a class action lawsuit.",
        "Subscriptions auto-renew every month and are non-refundable.",
        "You grant us a perpetual irrevocable license to your photos.",
    ]

    for t in all_test_clauses:
        lvl, reasons = detect_attention_level(t)
        combined_reasons = " ".join(reasons).lower()
        for prohibited in prohibited_terms:
            assert prohibited not in combined_reasons, (
                f"Prohibited legal term '{prohibited}' found in attention reasons: {combined_reasons}"
            )


def test_assign_attention_levels_batch():
    """Verify assign_attention_levels updates all Clause instances in place."""
    clauses = [
        create_sample_clause("All subscriptions auto-renew every month.", "c1"),
        create_sample_clause("Welcome to our service.", "c2"),
    ]
    assign_attention_levels(clauses)

    assert clauses[0].attention_level in (ATTENTION_LEVEL_MEDIUM, ATTENTION_LEVEL_HIGH)
    assert clauses[1].attention_level == ATTENTION_LEVEL_INFORMATIONAL
    assert len(clauses[0].attention_reasons) > 0
