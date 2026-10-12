# AI-Based Terms of Service Red-Flag Scanner & Clause Relationship Analyzer

> **IMPORTANT DISCLAIMER**
> This application is an **informational document-analysis system**, NOT a legal-advice system. It does not provide legal advice, determine legal enforceability, or certify regulatory compliance (e.g., GDPR, CCPA). Findings are provided as **attention levels** to guide human review against original contractual text.

---

## 1. Project Overview

Terms of Service (ToS) agreements are lengthy, opaque, and complex. Crucial contractual terms are frequently dispersed across separate sections (for instance, an easy cancellation policy in Section 2 conditioned by an auto-renewal and 30-day notice requirement in Section 8).

This application processes machine-readable ToS documents (PDF or pasted text), segments and classifies clauses into supported categories, assigns attention levels, detects cross-clause relationships, and produces plain-language, evidence-grounded explanations alongside document-grounded Q&A.

### Core Differentiators
1. **Cross-Clause Relationship Analysis**: Identifies conditional dependencies, combined effects, and cross-references between distributed clauses.
2. **User Concern Prioritization**: Focuses analysis on user-selected priority areas.
3. **Evidence-Grounded Explanations**: Grounds every explanation strictly in extracted clause evidence and abstains when evidence is missing.

---

## 2. Supported Categories, Attention Levels & Relationships

### Canonical Categories (8)
- Privacy & Data
- Payment
- Account
- Dispute
- Liability
- User Content
- Changes
- General (default fallback for provisions below classification threshold)

### Attention Levels (4)
- `High Attention` ($\ge 2$ strong indicators: binding arbitration, non-refundable charges, broad licenses)
- `Medium Attention` ($1$ strong indicator: unilateral updates, prolonged data retention)
- `Low Attention` ($\ge 1$ weak indicator: standard 'as-is' warranty disclaimer, general review notice)
- `Informational` (standard boilerplate without elevated attention indicators)

*(The system evaluates deterministic contractual indicators and explicitly avoids "legal risk scores" or assertions of illegality).*

### Canonical Cross-Clause Relationship Types (7)
- `SUPPORTS`: Explicit cross-references providing procedural elaboration (never an indiscriminate generic fallback)
- `QUALIFIES`: Scope limitations or restrictive provisos ("to the maximum extent permitted by law")
- `EXCEPTS`: Explicit carve-outs or exception phrasing ("except as provided herein")
- `OVERRIDES`: Supremacy or precedence language ("in the event of conflict, this section shall control")
- `DEPENDS_ON`: Operational or conditional dependencies ("subject to the provisions of Section X")
- `TEMPORAL`: Timing sequence, notice windows, billing cycles, or post-termination retention horizons
- `POTENTIAL_TENSION`: Potential textual or conditional friction between clauses (e.g. cancel anytime vs. advance renewal notice requirement; strictly cross-clause tension, never a legal conflict or violation)

---

## 3. Technology Stack

- **Backend**: Python 3.10+ (tested on Python 3.13), FastAPI, Uvicorn, Pydantic v2
- **Document Processing**: PyMuPDF (`fitz` $\ge 1.24.0$) for page-by-page machine-readable PDF extraction (OCR is out of scope)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalized)
- **Vector Store**: FAISS (`faiss-cpu`) with strict per-document isolated indices, calibrated retrieval threshold (`0.35`), and domain synonym lexical boosting (`CONTRACT_SYNONYM_MAP`)
- **Generative AI (Runtime)**: Google Gemini API (`gemini-2.5-flash`, `gemini-3.5-flash-lite`) via official modern SDK (`google-genai`) with structured JSON schema enforcement and deterministic mock mode
- **Evidence Validator**: Post-generation verbatim quote substring validation and forbidden legal conclusion guardrails
- **Frontend**: Responsive Vanilla Web UI (HTML5, modern CSS3 Glassmorphism design tokens, Vanilla JavaScript; zero external UI frameworks)
- **Testing**: `pytest`, `httpx` (130 automated test cases)

---

## 4. Folder Structure

```text
├── .env.example                  # Environment variables template
├── .gitignore                    # Ignored files (secrets, venv, caches, handoff notes)
├── requirements.txt              # Python dependencies
├── README.md                     # Primary project documentation
├── FINAL_IMPLEMENTATION_AUDIT.md # Phase F0 comprehensive verification report
├── FINAL_DOCUMENTATION_STATUS.md # Phase F1 documentation consolidation status
├── backend/                      # Backend core application
│   ├── main.py                   # FastAPI entrypoint, CORS & static mounting
│   ├── config.py                 # Centralized environment configuration (Pydantic Settings)
│   ├── core/                     # Domain constants, canonical categories, disclaimers
│   ├── api/                      # REST routes, schemas, request/response models
│   ├── document/                 # PDF extraction (PyMuPDF), cleaning, segmentation, store
│   ├── analysis/                 # Multi-label classifier, attention scoring, relationship engine
│   ├── retrieval/                # Embeddings (MiniLM) & isolated FAISS vector store
│   └── llm/                      # Google GenAI Gemini client, prompts, evidence validator
├── frontend/                     # Interactive web interface (HTML5, CSS3, JS)
│   ├── app.js                    # UI logic, state management, modal handling, Q&A console
│   ├── index.html                # Semantic UI layout, stepper, clause explorer, filters
│   └── styles.css                # Dark theme tokens, glassmorphism, responsive cards
├── evaluation/                   # Benchmark datasets, evaluation runner & results
│   ├── data/                     # Annotated ToS test documents
│   └── results/                  # Empirical evaluation report and confusion matrices
├── tests/                        # 130 automated unit, integration, and safety tests
└── data/                         # Sample Terms of Service documents
```

---

## 5. Getting Started & Configuration

### 1. Prerequisites
- Python 3.10+ installed
- Virtual environment (recommended)

### 2. Environment Setup & Dependency Installation
```bash
# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Environment File Configuration
Copy `.env.example` to create your local `.env`:
```bash
# On Windows:
copy .env.example .env

# On macOS/Linux:
cp .env.example .env
```

> [!WARNING]
> **Secret Safety Warning:** Never commit your `.env` file to version control. Keep `GEMINI_API_KEY` private at all times. The `.env` file is included in `.gitignore` to prevent accidental exposure. Do not embed API keys in client-side code, scripts, or commit history.

### 4. LLM Mode Configuration

The application supports two operating modes via `backend/config.py`:

#### A. Mock Mode (Development & Offline Testing — Default for Testing)
To run without external API dependencies or an API key:
```env
GEMINI_MOCK_MODE=true
```
- Uses local, deterministic mock generation grounded in extracted document clauses.
- Does not require a `GEMINI_API_KEY`.
- All automated unit and regression tests pass in mock mode.

#### B. Real Gemini Mode (Production / Live Evaluation)
To use live Google Gemini models via the official `google-genai` SDK:
```env
GEMINI_MOCK_MODE=false
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash
```
- **Error Behavior**: If `GEMINI_MOCK_MODE=false` and `GEMINI_API_KEY` is missing or empty, the application returns a clear `llm_unavailable` error status. It will **never** silently fall back to mock mode in real mode.
- `GEMINI_MODEL_NAME` is fully configurable and should match a model supported by your Gemini API account (e.g., `gemini-2.5-flash`).

### 5. Running the Development Server
```bash
uvicorn backend.main:app --reload
```
The server starts locally at `http://127.0.0.1:8000`.

### 6. Accessing the Frontend & API Endpoints
- **Web Application Interface**: Open [`http://127.0.0.1:8000/app/`](http://127.0.0.1:8000/app/) in your browser.
- **Root Metadata**: `GET http://127.0.0.1:8000/`
- **Health Check Endpoint**:
  ```bash
  curl http://127.0.0.1:8000/api/health
  ```
  Expected response:
  ```json
  {
    "status": "healthy",
    "service": "tos-red-flag-scanner",
    "version": "0.1.0"
  }
  ```

### 7. Running the Test Suite
```bash
python -m pytest -v
```
All 130 test cases run offline using deterministic fixtures and mock mode without requiring an active network connection or real API key.
