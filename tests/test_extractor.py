import pytest
import fitz
from backend.document.extractor import (
    extract_from_pdf,
    extract_from_text,
    InvalidDocumentError,
    UnsupportedDocumentError,
)


def create_pdf_with_text(pages_text: list[str]) -> bytes:
    """Helper to synthesize a machine-readable PDF in memory."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def create_image_only_pdf() -> bytes:
    """Helper to synthesize a PDF that contains only a pixmap/image and no text."""
    doc = fitz.open()
    page = doc.new_page()
    # Create a 100x100 RGB image pixmap
    pix = fitz.Pixmap(fitz.csRGB, (0, 0, 100, 100), 0)
    page.insert_image(page.rect, pixmap=pix)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_extract_valid_single_page_pdf():
    """Verify that text and metadata are extracted from a single-page machine-readable PDF."""
    pdf_bytes = create_pdf_with_text(["1. General Terms\nThis agreement governs use of the service."])
    doc = extract_from_pdf(pdf_bytes, filename="test.pdf")

    assert doc.document_id.startswith("doc_")
    assert doc.source_type == "pdf"
    assert doc.filename == "test.pdf"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert "This agreement governs" in doc.raw_text
    assert "This agreement governs" in doc.cleaned_text


def test_extract_valid_multi_page_pdf():
    """Verify multi-page PDF extracts with sequential 1-based page numbering and correct offsets."""
    pages = [
        "Section 1. Privacy\nWe collect your email and usage telemetry.",
        "Section 2. Payment\nAll fees are non-refundable and renew automatically.",
    ]
    pdf_bytes = create_pdf_with_text(pages)
    doc = extract_from_pdf(pdf_bytes, filename="multi.pdf")

    assert len(doc.pages) == 2
    assert doc.pages[0].page_number == 1
    assert doc.pages[1].page_number == 2
    assert doc.pages[0].char_start < doc.pages[1].char_start
    assert "telemetry" in doc.raw_text
    assert "non-refundable" in doc.raw_text


def test_extract_empty_pdf_raises_error():
    """Verify that empty byte array or corrupted PDF raises InvalidDocumentError."""
    with pytest.raises(InvalidDocumentError):
        extract_from_pdf(b"")

    with pytest.raises(InvalidDocumentError):
        extract_from_pdf(b"Not a valid PDF header or content")


def test_extract_image_only_pdf_raises_unsupported():
    """Verify that a scanned/image-only PDF raises UnsupportedDocumentError."""
    image_pdf_bytes = create_image_only_pdf()
    with pytest.raises(UnsupportedDocumentError) as exc_info:
        extract_from_pdf(image_pdf_bytes)

    assert "Scanned or image-only PDFs are not supported" in str(exc_info.value)


def test_extract_plain_text():
    """Verify extraction from raw string input."""
    raw = "1. Introduction\nWelcome to our platform.\n\n2. Accounts\nYou are responsible for passwords."
    doc = extract_from_text(raw, title="raw_terms.txt")

    assert doc.source_type == "text"
    assert doc.filename == "raw_terms.txt"
    assert len(doc.pages) == 1
    assert doc.pages[0].page_number == 1
    assert doc.raw_text == raw
    assert "responsible for passwords" in doc.cleaned_text


def test_extract_empty_plain_text_raises_error():
    """Verify that empty or whitespace-only text raises InvalidDocumentError."""
    with pytest.raises(InvalidDocumentError):
        extract_from_text("")

    with pytest.raises(InvalidDocumentError):
        extract_from_text("   \n\t   ")
