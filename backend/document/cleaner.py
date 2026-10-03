"""Text cleaning and normalization module for Terms of Service documents.

Non-negotiable rules:
- Cleaning must NOT rewrite sentences, summarize, paraphrase, or remove contractual meaning.
- Keep the original extracted text available for evidence.
- Merely normalizes encoding, excessive whitespace, line breaks, and page header/footer noise.
"""
import re
from typing import List


def normalize_encoding_artifacts(text: str) -> str:
    """Normalize unicode quotes, dashes, and hidden formatting characters."""
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip zero-width and invisible format characters (Word Joiner, ZWSP, ZWNJ, ZWJ, BOM, soft hyphen, etc.)
    text = re.sub(r"[\u200b-\u200d\u2060\ufeff\u00ad\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)

    # Standardize non-breaking spaces and special whitespace variants to standard ASCII space
    text = re.sub(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]", " ", text)

    # Standardize curly quotes, smart quotes, and backticks to straight quotes
    text = re.sub(r"[\u2018\u2019\u201a\u201b`´]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e\u201f«»]", '"', text)

    # Standardize en/em dashes to standard hyphen/dash
    text = re.sub(r"\s*[\u2013\u2014]\s*", " - ", text)

    return text


def reconnect_hyphenated_words(text: str) -> str:
    """Reconnect words split across line breaks by hyphens (e.g. 'termi-\\nnation' -> 'termination')."""
    if not text:
        return ""
    # Matches a word part ending with hyphen, newline, and continuation in lowercase
    return re.sub(r"([a-zA-Z]{2,})-\n([a-z]{2,})", r"\1\2", text)


def remove_repeated_headers_footers(pages_text: List[str]) -> List[str]:
    """Detect and remove repeated running headers and footers across multiple pages.

    Identifies lines at the very top or bottom of pages that repeat identically or match
    common page number patterns (e.g., 'Page 2 of 10' or 'Terms of Service | Confidential').
    """
    if len(pages_text) < 2:
        return pages_text

    # Extract top and bottom candidate lines across pages
    top_candidates = []
    bottom_candidates = []
    for page in pages_text:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if lines:
            top_candidates.append(lines[0])
            bottom_candidates.append(lines[-1])

    header_to_remove = set()
    footer_to_remove = set()

    # Generic page number regex: "Page 1 of 5", "Page 2", "- 3 -"
    page_num_regex = re.compile(r"^(?:page\s+\d+(?:\s+of\s+\d+)?|-?\s*\d+\s*-?)$", re.IGNORECASE)

    # Check top lines for identical repeats or page number patterns
    for candidate in top_candidates:
        if page_num_regex.match(candidate):
            header_to_remove.add(candidate)
        elif top_candidates.count(candidate) >= (len(pages_text) // 2 + 1):
            header_to_remove.add(candidate)

    # Check bottom lines
    for candidate in bottom_candidates:
        if page_num_regex.match(candidate):
            footer_to_remove.add(candidate)
        elif bottom_candidates.count(candidate) >= (len(pages_text) // 2 + 1):
            footer_to_remove.add(candidate)

    cleaned_pages = []
    for page in pages_text:
        lines = page.splitlines()
        filtered_lines = []
        for ln in lines:
            s_ln = ln.strip()
            if s_ln in header_to_remove or s_ln in footer_to_remove or page_num_regex.match(s_ln):
                continue
            filtered_lines.append(ln)
        cleaned_pages.append("\n".join(filtered_lines))

    return cleaned_pages


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace while preserving paragraph and section breaks."""
    if not text:
        return ""

    # Replace horizontal tabs with spaces
    text = text.replace("\t", " ")

    # Collapse multiple horizontal spaces on each line
    lines = [re.sub(r"[ ]{2,}", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Normalize excessive vertical line breaks (max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text: str) -> str:
    """Clean and normalize extracted text without modifying substantive contractual meaning."""
    if not text or not text.strip():
        return ""

    text = normalize_encoding_artifacts(text)
    text = reconnect_hyphenated_words(text)
    text = normalize_whitespace(text)

    return text
