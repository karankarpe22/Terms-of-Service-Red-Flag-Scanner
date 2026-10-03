import pytest
from backend.document.cleaner import (
    clean_text,
    normalize_encoding_artifacts,
    reconnect_hyphenated_words,
    remove_repeated_headers_footers,
    normalize_whitespace,
)


def test_whitespace_normalization():
    """Verify tabs, excessive spaces, and trailing whitespace are collapsed."""
    raw = "  Section 1.   Terms   \t  of   Service.  \n\n\n\nNext line here.   "
    cleaned = clean_text(raw)
    assert "   " not in cleaned
    assert "\t" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Section 1. Terms of Service.\n\nNext line here." == cleaned


def test_linebreak_hyphen_reconnection():
    """Verify words split across line breaks are cleanly joined."""
    raw = "You may request termi-\nnation of your subscrip-\ntion at any time."
    cleaned = reconnect_hyphenated_words(raw)
    assert "termination" in cleaned
    assert "subscription" in cleaned


def test_encoding_artifacts_normalized():
    """Verify smart quotes, dashes, and zero-width spaces are standardized."""
    raw = "We\u200b don\u2019t accept \u2018cash\u2019 \u2014 only cards."
    cleaned = normalize_encoding_artifacts(raw)
    assert "\u200b" not in cleaned
    assert "don't accept 'cash' - only cards." in cleaned


def test_substantive_words_preserved():
    """Verify that every substantive legal word is preserved without rephrasing or omission."""
    original = (
        "TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL THE "
        "COMPANY BE LIABLE FOR ANY CONSEQUENTIAL, INDIRECT, OR INCIDENTAL DAMAGES."
    )
    cleaned = clean_text(original)
    original_words = original.split()
    cleaned_words = cleaned.split()

    assert original_words == cleaned_words


def test_repeated_headers_footers_removed():
    """Verify that repeated running headers and footers across pages are stripped."""
    pages = [
        "Confidential Terms of Service\n1. Introduction\nWelcome.\nPage 1 of 3",
        "Confidential Terms of Service\n2. Accounts\nMaintain security.\nPage 2 of 3",
        "Confidential Terms of Service\n3. Payment\nFees are due.\nPage 3 of 3",
    ]
    cleaned_pages = remove_repeated_headers_footers(pages)

    for p in cleaned_pages:
        assert "Confidential Terms of Service" not in p
        assert "Page 1 of 3" not in p
        assert "Page 2 of 3" not in p
        assert "Page 3 of 3" not in p

    assert "1. Introduction" in cleaned_pages[0]
    assert "2. Accounts" in cleaned_pages[1]
    assert "3. Payment" in cleaned_pages[2]
