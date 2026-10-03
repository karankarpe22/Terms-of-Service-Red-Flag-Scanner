"""Clause boundary segmentation module for Terms of Service documents.

Performs deterministic segmentation using:
- Major numbered sections (e.g. '1.', '1. Acceptance of Terms', 'SECTION 1', 'Section 2')
- Subsection numbering (e.g. '1.1', '1.2', '1.1.1')
- Lettered sections (e.g. 'A. Introduction', 'B. Payment')
- Parenthesized subclauses (e.g. '(a)', '(b)', '(1)')
- Distinct uppercase/title headings
- Paragraph boundaries (double newlines)
- Sentence boundaries for excessively long unnumbered text blocks

Non-negotiable requirements:
- Each Clause must contain: clause_id, clause_index, section_title, text, source_location, document_id.
- SourceLocation must preserve page number (when available) and character offset information.
"""
import re
from typing import List, Tuple, Optional
from backend.document.models import Document, Clause, SourceLocation, PageInfo


# Regex patterns for structural heading markers
PATTERN_SUBSECTION = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)*)\.?\s*[:\-\.]?\s*(.*)$"
)
PATTERN_MAJOR_NUMBERED = re.compile(
    r"^\s*(?:(?:SECTION|Section|ARTICLE|Article)\s+)?(\d+)\.?\s*[:\-\.]?\s*(.*)$"
)
PATTERN_LETTERED = re.compile(
    r"^\s*([A-Z])\.\s+([A-Z0-9].*)$"
)
PATTERN_PAREN_SUBSECTION = re.compile(
    r"^\s*\(([a-z0-9]+)\)\s*(.*)$"
)
PATTERN_UPPERCASE_HEADING = re.compile(
    r"^[A-Z0-9\s,\-\'&/]{4,60}$"
)
PATTERN_DOC_TITLE = re.compile(
    r"(?i)\b(terms\s+of\s+service|terms\s+of\s+use|user\s+agreement|terms\s+and\s+conditions)\b"
)


def _find_source_location(
    clause_text: str,
    raw_text: str,
    pages: List[PageInfo],
    search_start_hint: int = 0,
    source_type: str = "text",
) -> Tuple[SourceLocation, int]:
    """Find the exact character range and page number of a clause within raw text."""
    clean_sample = clause_text.strip()
    if not clean_sample:
        return SourceLocation(source_type=source_type), search_start_hint

    probe = clean_sample[:min(40, len(clean_sample))].strip()
    start_char = raw_text.find(probe, search_start_hint)

    if start_char == -1:
        start_char = raw_text.find(probe)

    if start_char != -1:
        end_char = start_char + len(clean_sample)
        next_hint = end_char

        start_page = 1
        end_page = 1
        if pages:
            for p in pages:
                if p.char_start <= start_char <= p.char_end:
                    start_page = p.page_number
                if p.char_start <= end_char <= p.char_end:
                    end_page = p.page_number
            if not end_page:
                end_page = start_page

        return (
            SourceLocation(
                page_number=start_page if source_type == "pdf" else None,
                end_page_number=end_page if source_type == "pdf" else None,
                start_char=start_char,
                end_char=end_char,
                source_type=source_type,
            ),
            next_hint,
        )

    return (
        SourceLocation(
            page_number=1 if source_type == "pdf" else None,
            start_char=search_start_hint,
            end_char=search_start_hint + len(clean_sample),
            source_type=source_type,
        ),
        search_start_hint + len(clean_sample),
    )


def _split_long_paragraph(text: str, max_words: int = 250) -> List[str]:
    """Split an excessively long paragraph along sentence boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_len + s_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = s_words
        else:
            current_chunk.append(sentence)
            current_len += s_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks or [text]


def segment_document(document: Document) -> List[Clause]:
    """Segment a Document into discrete, structured Clause objects.

    Handles multi-style headings, subsections, and unstructured paragraphs.
    Ensures every clause contains stable IDs, section headings, and source locations.
    """
    raw_text = document.raw_text
    cleaned_text = document.cleaned_text
    pages = document.pages
    source_type = document.source_type
    doc_id = document.document_id

    if not cleaned_text or not cleaned_text.strip():
        return []

    lines = cleaned_text.splitlines()
    raw_blocks: List[Tuple[str, str]] = []

    current_major_heading: Optional[str] = None
    current_sub_heading: Optional[str] = None
    current_paren_sub: Optional[str] = None
    current_body_lines: List[str] = []

    def flush_current_clause():
        nonlocal current_body_lines, current_paren_sub
        if not current_body_lines:
            return

        body_text = " ".join([l.strip() for l in current_body_lines if l.strip()]).strip()
        if not body_text:
            current_body_lines = []
            return

        # Determine active section title
        title_parts = []
        if current_major_heading:
            title_parts.append(current_major_heading)
        if current_sub_heading:
            title_parts.append(current_sub_heading)
        if current_paren_sub:
            title_parts.append(f"({current_paren_sub})")

        if title_parts:
            title = " - ".join(title_parts)
        else:
            inline_match = re.match(r"^([A-Z][A-Za-z0-9\s/&,-]{2,45})[:\-]\s+(.*)$", body_text, re.DOTALL)
            if inline_match and len(inline_match.group(1).split()) <= 6:
                title = inline_match.group(1).strip()
            else:
                title = "General"

        sub_chunks = _split_long_paragraph(body_text, max_words=250)
        for chunk in sub_chunks:
            if chunk.strip():
                raw_blocks.append((title, chunk.strip()))

        current_body_lines = []
        current_paren_sub = None

    i = 0
    total_lines = len(lines)

    # Check for document header/title on initial lines
    while i < total_lines:
        line_s = lines[i].strip()
        if not line_s:
            i += 1
            continue
        # If the first non-empty line looks like a document title (e.g. "Sample - Terms of Service")
        # and not a numbered section
        if not current_major_heading and not PATTERN_MAJOR_NUMBERED.match(line_s):
            if PATTERN_DOC_TITLE.search(line_s) and len(line_s.split()) <= 15:
                document.metadata["document_title"] = line_s
                i += 1
                continue
        break

    while i < total_lines:
        line = lines[i].strip()
        if not line:
            if current_body_lines:
                flush_current_clause()
            i += 1
            continue

        # 1. Check for Parenthesized Subsection (e.g. "(a) You may not...", "(1) First...")
        paren_match = PATTERN_PAREN_SUBSECTION.match(line)
        if paren_match:
            flush_current_clause()
            paren_id, paren_rest = paren_match.groups()
            current_paren_sub = paren_id
            if paren_rest.strip():
                current_body_lines.append(paren_rest.strip())
            i += 1
            continue

        # 2. Check for Subsection Pattern (e.g. "1.1 Permitted Uses", "1.2.3 Requirements")
        sub_match = PATTERN_SUBSECTION.match(line)
        if sub_match:
            flush_current_clause()
            sub_num, sub_rest = sub_match.groups()
            sub_rest = sub_rest.strip()
            if sub_rest and len(sub_rest.split()) > 10:
                current_sub_heading = sub_num
                current_body_lines.append(sub_rest)
            else:
                current_sub_heading = f"{sub_num} {sub_rest}".strip()
            current_paren_sub = None
            i += 1
            continue

        # 3. Check for Major Numbered Section (e.g. "1. Acceptance of Terms", "SECTION 2", "Section 3. Payment")
        major_match = PATTERN_MAJOR_NUMBERED.match(line)
        if major_match:
            num, title_rest = major_match.groups()
            title_rest = title_rest.strip()
            if len(title_rest.split()) <= 12:
                flush_current_clause()
                current_major_heading = f"{num}. {title_rest}".strip().rstrip(".:- ")
                current_sub_heading = None
                current_paren_sub = None
                i += 1
                continue
            elif not current_major_heading and not current_body_lines:
                parts = re.split(r"(?<=[.:])\s+", title_rest, maxsplit=1)
                flush_current_clause()
                current_major_heading = f"{num}. {parts[0]}".strip().rstrip(".:- ")
                current_sub_heading = None
                current_paren_sub = None
                if len(parts) > 1:
                    current_body_lines.append(parts[1])
                i += 1
                continue

        # 4. Check for Lettered Section (e.g. "A. Introduction", "B. Privacy")
        letter_match = PATTERN_LETTERED.match(line)
        if letter_match:
            letter, letter_rest = letter_match.groups()
            if len(letter_rest.split()) <= 10:
                flush_current_clause()
                current_major_heading = f"{letter}. {letter_rest}".strip().rstrip(".:- ")
                current_sub_heading = None
                current_paren_sub = None
                i += 1
                continue

        # 5. Check for Standalone Uppercase Heading (e.g. "LIMITATION OF LIABILITY")
        if PATTERN_UPPERCASE_HEADING.match(line) and not line.endswith("."):
            prev_blank = (i == 0) or (not lines[i - 1].strip())
            next_blank = (i + 1 == total_lines) or (not lines[i + 1].strip())
            if prev_blank and next_blank:
                flush_current_clause()
                current_major_heading = line.strip()
                current_sub_heading = None
                current_paren_sub = None
                i += 1
                continue

        # Standard body line
        current_body_lines.append(line)
        i += 1

    flush_current_clause()

    if not raw_blocks:
        paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]
        for p in paragraphs:
            inline_match = re.match(r"^([A-Z][A-Za-z0-9\s/&,-]{2,45})[:\-]\s+(.*)$", p, re.DOTALL)
            if inline_match and len(inline_match.group(1).split()) <= 6:
                raw_blocks.append((inline_match.group(1).strip(), p))
            else:
                raw_blocks.append(("General", p))

    clauses: List[Clause] = []
    search_hint = 0

    for idx, (sec_title, clause_text) in enumerate(raw_blocks, start=1):
        clause_id = f"{doc_id}_clause_{idx:03d}"

        source_loc, search_hint = _find_source_location(
            clause_text=clause_text,
            raw_text=raw_text,
            pages=pages,
            search_start_hint=search_hint,
            source_type=source_type,
        )

        clauses.append(
            Clause(
                clause_id=clause_id,
                document_id=doc_id,
                clause_index=idx,
                section_title=sec_title,
                text=clause_text,
                raw_text=clause_text,
                source_location=source_loc,
            )
        )

    document.clauses = clauses
    return clauses
