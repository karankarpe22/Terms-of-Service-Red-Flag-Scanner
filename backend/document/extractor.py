"""Document text extraction module using PyMuPDF for machine-readable PDFs and raw text.

Preserves page numbers, character offsets, and exact source text.
Rejects empty or image-only scanned PDFs with clear application errors.
OCR is explicitly not performed in this phase.
"""
import uuid
import fitz  # PyMuPDF
from typing import List, Optional
from backend.document.models import Document, PageInfo
from backend.document.cleaner import clean_text, remove_repeated_headers_footers


class DocumentExtractionError(Exception):
    """Base exception for document extraction failures."""
    pass


class InvalidDocumentError(DocumentExtractionError):
    """Raised when the document file is corrupted, empty, or invalid."""
    pass


class UnsupportedDocumentError(DocumentExtractionError):
    """Raised when the document contains no extractable machine-readable text (e.g. scanned/image-only)."""
    pass


def extract_from_pdf(
    pdf_bytes: bytes,
    filename: Optional[str] = None,
    document_id: Optional[str] = None,
) -> Document:
    """Extract page-by-page text from machine-readable PDF bytes.

    Args:
        pdf_bytes: Raw bytes of the uploaded PDF.
        filename: Optional original filename.
        document_id: Optional existing or custom document ID.

    Returns:
        Document domain object populated with raw text, pages, and cleaned text.

    Raises:
        InvalidDocumentError: If PDF is empty (0 bytes) or corrupted.
        UnsupportedDocumentError: If PDF is image-only/scanned with no machine-readable text.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise InvalidDocumentError("The uploaded file is empty (0 bytes).")

    doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"

    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise InvalidDocumentError(f"Failed to open PDF: {str(exc)}") from exc

    try:
        if pdf_doc.page_count == 0:
            raise InvalidDocumentError("The uploaded PDF contains 0 pages.")

        pages_info: List[PageInfo] = []
        raw_pages_text: List[str] = []
        total_extracted_chars = 0
        current_offset = 0

        for page_idx in range(pdf_doc.page_count):
            page = pdf_doc[page_idx]
            page_text = page.get_text()
            raw_pages_text.append(page_text)
            total_extracted_chars += len(page_text.strip())

            char_start = current_offset
            char_end = current_offset + len(page_text)
            current_offset = char_end + 1  # accounting for join separator

            pages_info.append(
                PageInfo(
                    page_number=page_idx + 1,
                    text=page_text,
                    char_start=char_start,
                    char_end=char_end,
                )
            )

        # Check if the document has no extractable text (image-only / scanned)
        if total_extracted_chars == 0:
            raise UnsupportedDocumentError(
                "The uploaded PDF contains no extractable machine-readable text. "
                "Scanned or image-only PDFs are not supported in this version. "
                "Please upload a machine-readable PDF or paste the text directly."
            )

        # Build full raw text
        full_raw_text = "\n".join(raw_pages_text)

        # Remove repeated headers/footers across pages
        cleaned_page_texts = remove_repeated_headers_footers(raw_pages_text)
        cleaned_doc_text = clean_text("\n\n".join(cleaned_page_texts))

        return Document(
            document_id=doc_id,
            filename=filename or "uploaded_document.pdf",
            source_type="pdf",
            raw_text=full_raw_text,
            cleaned_text=cleaned_doc_text,
            pages=pages_info,
            metadata={
                "page_count": pdf_doc.page_count,
                "total_raw_chars": len(full_raw_text),
                "total_cleaned_chars": len(cleaned_doc_text),
            },
        )
    finally:
        pdf_doc.close()


def extract_from_text(
    raw_text: str,
    title: Optional[str] = None,
    document_id: Optional[str] = None,
) -> Document:
    """Extract and wrap raw plain text into a Document object.

    Args:
        raw_text: Raw Terms of Service text string.
        title: Optional title.
        document_id: Optional custom document ID.

    Returns:
        Document domain object.

    Raises:
        InvalidDocumentError: If text is empty or contains only whitespace.
    """
    if not raw_text or not raw_text.strip():
        raise InvalidDocumentError("The submitted text is empty or contains only whitespace.")

    doc_id = document_id or f"doc_{uuid.uuid4().hex[:12]}"
    cleaned_doc_text = clean_text(raw_text)

    page_info = PageInfo(
        page_number=1,
        text=raw_text,
        char_start=0,
        char_end=len(raw_text),
    )

    return Document(
        document_id=doc_id,
        filename=title or "pasted_text.txt",
        source_type="text",
        raw_text=raw_text,
        cleaned_text=cleaned_doc_text,
        pages=[page_info],
        metadata={
            "page_count": 1,
            "total_raw_chars": len(raw_text),
            "total_cleaned_chars": len(cleaned_doc_text),
        },
    )
