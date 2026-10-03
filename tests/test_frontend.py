"""Automated tests for Phase 7 Frontend Web Application.

Verifies:
1. Frontend static assets exist and are served via FastAPI TestClient.
2. No API keys, passwords, or secrets are exposed in client-side files.
3. HTML contains all required semantic containers, accessibility tags, and disclaimers.
4. CSS contains responsive media queries, focus states, and attention color tokens.
5. JavaScript contains state management, API base URL fallback, and graceful degradation handlers.
"""
from pathlib import Path
import re
import pytest


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def test_frontend_files_exist():
    """Verify core frontend files exist in the frontend/ directory."""
    assert (FRONTEND_DIR / "index.html").exists(), "index.html missing"
    assert (FRONTEND_DIR / "styles.css").exists(), "styles.css missing"
    assert (FRONTEND_DIR / "app.js").exists(), "app.js missing"


def test_frontend_served_via_api(client):
    """Verify FastAPI serves index.html and static assets at /app/."""
    # Root metadata includes web_interface link
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert root_res.json().get("web_interface") == "/app/"

    # Static HTML
    html_res = client.get("/app/")
    assert html_res.status_code == 200
    assert "ToS Red-Flag Scanner" in html_res.text
    assert "Not Legal Advice" in html_res.text

    # Static CSS
    css_res = client.get("/app/styles.css")
    assert css_res.status_code == 200
    assert "--color-primary" in css_res.text

    # Static JS
    js_res = client.get("/app/app.js")
    assert js_res.status_code == 200
    assert "executeAnalysisPipeline" in js_res.text


def test_no_secrets_exposed_in_frontend():
    """Critical security check: ensure no API keys or server secrets are present in frontend assets."""
    forbidden_patterns = [
        re.compile(r"AIza[0-9A-Za-z-_]{35}"),  # Google API key pattern
        re.compile(r"(?:api_key|apikey|secret)\s*[:=]\s*['\"][A-Za-z0-9-_]{20,}['\"]", re.I),
        re.compile(r"bearer\s+[A-Za-z0-9-_\.]{20,}", re.I),
    ]

    for fname in ["index.html", "styles.css", "app.js"]:
        content = (FRONTEND_DIR / fname).read_text(encoding="utf-8")
        for pat in forbidden_patterns:
            assert not pat.search(content), f"Potential secret exposed in frontend/{fname}"


def test_html_structure_completeness():
    """Verify that index.html contains all necessary UI elements, sections, and accessibility labels."""
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

    # Header & Disclaimers
    assert "ToS Red-Flag Scanner" in html
    assert "Informational Analysis &bull; Not Legal Advice" in html

    # Input modes: PDF & Text
    assert 'id="tabPdf"' in html
    assert 'id="tabText"' in html
    assert 'id="dropZone"' in html
    assert 'id="pdfFileInput"' in html
    assert 'id="tosTextInput"' in html
    assert 'id="btnAnalyze"' in html

    # Concerns selection
    for concern in ["Privacy & Data", "Payment", "Account", "Dispute", "Liability", "User Content", "Changes"]:
        assert concern in html

    # Live progress section
    assert 'id="progressSection"' in html
    assert 'id="step1"' in html
    assert 'id="step4"' in html

    # Dashboard Metrics
    assert 'id="countHigh"' in html
    assert 'id="countMedium"' in html
    assert 'id="countLow"' in html
    assert 'id="countInfo"' in html
    assert 'id="countRels"' in html

    # Filtering toolbar
    assert 'id="clauseSearchInput"' in html
    assert 'id="categoryFilter"' in html
    assert 'id="attentionFilter"' in html

    # Dedicated Clause Relationships column
    assert 'id="relationshipsContainer"' in html

    # Modals
    assert 'id="clauseModal"' in html
    assert 'id="relModal"' in html
    assert "ORIGINAL SOURCE CLAUSE" in html
    assert "DETERMINISTIC ATTENTION INDICATORS" in html
    assert "AI-GENERATED EXPLANATION" in html

    # Phase 8: Evidence-Grounded Q&A Section
    assert 'id="qaSection"' in html
    assert 'id="qaQuestionInput"' in html
    assert 'id="btnAskQuestion"' in html
    assert 'id="qaResultCard"' in html
    assert 'id="qaSourcesList"' in html


def test_css_responsive_and_accessible():
    """Verify that CSS implements responsive breakpoints, focus outlines, and color tokens."""
    css = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")

    # Accessible focus
    assert ":focus-visible" in css

    # Responsive breakpoints
    assert "@media (max-width: 1024px)" in css
    assert "@media (max-width: 768px)" in css
    assert "@media (max-width: 480px)" in css

    # Attention levels
    assert "--att-high-bg" in css
    assert "--att-med-bg" in css
    assert "--att-low-bg" in css
    assert "--att-info-bg" in css

    # Phase 8: Q&A styles
    assert ".qa-section-card" in css
    assert ".qa-source-card" in css


def test_javascript_api_integration_and_degradation():
    """Verify JS implements correct API calls, caching, and fallback handling."""
    js = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    # API endpoints called
    assert "/documents/upload" in js
    assert "/documents/text" in js
    assert "/relationships" in js
    assert "/synthesize" in js
    assert "/explain-clause" in js
    assert "/explain-relationship" in js
    assert "/ask" in js

    # Graceful degradation logic
    assert "llm_unavailable" in js
    assert "evidence_sufficient" in js
    assert "clauseExplanationCache" in js
    assert "qaCache" in js
    assert "handleAskQuestion" in js
    assert "openClauseModal" in js

