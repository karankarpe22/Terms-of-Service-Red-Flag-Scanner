from pathlib import Path
import pytest
from backend.document.extractor import extract_from_text, extract_from_pdf
from backend.document.segmenter import segment_document
from tests.test_extractor import create_pdf_with_text


def test_segment_numbered_sections():
    """Verify standard numbered sections e.g. '1.', '2.', '3.' are parsed into distinct clauses."""
    text = (
        "1. Acceptance of Terms\n"
        "By accessing our platform, you agree to these terms.\n\n"
        "2. Accounts\n"
        "You must maintain the confidentiality of your credentials.\n\n"
        "3. Termination\n"
        "We may terminate your account at our discretion."
    )
    doc = extract_from_text(text)
    clauses = segment_document(doc)

    assert len(clauses) == 3
    assert clauses[0].section_title.startswith("1.")
    assert "By accessing our platform" in clauses[0].text
    assert clauses[1].section_title.startswith("2.")
    assert "confidentiality of your credentials" in clauses[1].text
    assert clauses[2].section_title.startswith("3.")
    assert "terminate your account" in clauses[2].text


def test_segment_subsection_numbering():
    """Verify hierarchical subsections e.g. 1.1, 1.2, (a), (b) are segmented."""
    text = (
        "SECTION 1: USER CONDUCT\n\n"
        "1.1 Permitted Uses\n"
        "You may use the service only for lawful purposes.\n\n"
        "1.2 Prohibited Activities\n"
        "(a) You shall not reverse engineer the application.\n"
        "(b) You shall not deploy scrapers or spiders."
    )
    doc = extract_from_text(text)
    clauses = segment_document(doc)

    assert len(clauses) >= 3
    # Check that subsections were captured
    clause_titles = [c.section_title for c in clauses]
    assert any("1.1" in t for t in clause_titles)
    assert any("1.2" in t for t in clause_titles)
    assert any("reverse engineer" in c.text for c in clauses)
    assert any("deploy scrapers" in c.text for c in clauses)


def test_segment_lettered_and_uppercase_sections():
    """Verify lettered headings ('A.', 'B.') and standalone uppercase headings."""
    text = (
        "A. ELIGIBILITY AND REGISTRATION\n"
        "You must be at least 18 years old to use this service.\n\n"
        "B. PAYMENT TERMS\n"
        "All fees are charged in advance and are non-refundable.\n\n"
        "DISPUTE RESOLUTION\n\n"
        "All claims must be arbitrated on an individual basis."
    )
    doc = extract_from_text(text)
    clauses = segment_document(doc)

    assert len(clauses) == 3
    assert "A." in clauses[0].section_title
    assert "18 years old" in clauses[0].text
    assert "B." in clauses[1].section_title
    assert "charged in advance" in clauses[1].text
    assert "DISPUTE RESOLUTION" in clauses[2].section_title
    assert "arbitrated" in clauses[2].text


def test_segment_irregular_document():
    """Verify irregular documents with no formal headers still segment cleanly by paragraphs."""
    text = (
        "This is an informal terms agreement with no section numbering at all. "
        "Users should read this carefully before proceeding.\n\n"
        "Second paragraph details our cancellation rules and non-refund policy. "
        "Any cancellation requires thirty days notice.\n\n"
        "Third paragraph limits company liability to ten dollars."
    )
    doc = extract_from_text(text)
    clauses = segment_document(doc)

    assert len(clauses) == 3
    assert clauses[0].clause_index == 1
    assert clauses[1].clause_index == 2
    assert clauses[2].clause_index == 3
    assert "informal terms agreement" in clauses[0].text
    assert "cancellation rules" in clauses[1].text
    assert "limits company liability" in clauses[2].text


def test_stable_clause_ids():
    """Verify clause IDs are stable, unique, and formatted with zero-padded indices."""
    text = (
        "1. Clause One\nText one.\n\n"
        "2. Clause Two\nText two.\n\n"
        "3. Clause Three\nText three."
    )
    doc = extract_from_text(text, document_id="doc_stable123")
    clauses = segment_document(doc)

    assert clauses[0].clause_id == "doc_stable123_clause_001"
    assert clauses[1].clause_id == "doc_stable123_clause_002"
    assert clauses[2].clause_id == "doc_stable123_clause_003"


def test_source_location_preservation_pdf():
    """Verify that source location correctly captures multi-page PDF page numbers and character offsets."""
    p1 = "1. First Section\nThis text is located on page one of the agreement."
    p2 = "2. Second Section\nThis text is located on page two of the agreement."
    pdf_bytes = create_pdf_with_text([p1, p2])

    doc = extract_from_pdf(pdf_bytes, filename="multipage.pdf")
    clauses = segment_document(doc)

    assert len(clauses) == 2
    assert clauses[0].source_location.page_number == 1
    assert clauses[0].source_location.start_char is not None
    assert clauses[0].source_location.end_char is not None
    assert clauses[1].source_location.page_number == 2
    assert clauses[1].source_location.start_char > clauses[0].source_location.end_char
    assert "Page 1" in clauses[0].source_location.to_display_string()
    assert "Page 2" in clauses[1].source_location.to_display_string()


def test_sample_tos_document_segmentation():
    """Verify segmentation on the official project sample_tos.txt."""
    sample_path = Path("data/sample_documents/sample_tos.txt")
    assert sample_path.exists()

    raw_text = sample_path.read_text(encoding="utf-8")
    doc = extract_from_text(raw_text, title="sample_tos.txt")
    clauses = segment_document(doc)

    # sample_tos.txt has 8 numbered sections
    assert len(clauses) == 8

    expected_titles = [
        "1. Acceptance of Terms",
        "2. User Accounts and Termination",
        "3. Payment, Billing, and Auto-Renewal",
        "4. Unilateral Modifications",
        "5. User Content and License Grant",
        "6. Limitation of Liability",
        "7. Dispute Resolution and Class Action Waiver",
        "8. Data Retention Following Account Closure",
    ]

    for i, expected_title in enumerate(expected_titles):
        assert expected_title in clauses[i].section_title
        assert len(clauses[i].text) > 20
