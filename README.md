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

## 2. Supported Categories & Attention Levels

### Categories
- Privacy & Data
- Payment
- Account
- Dispute
- Liability
- User Content
- Changes

### Attention Levels
- `High Attention`
- `Medium Attention`
- `Low Attention`
- `Informational`

*(The system explicitly avoids using "legal risk scores".)*

---

## 3. Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, Pydantic v2
- **Document Processing**: PyMuPDF (`fitz`) for machine-readable PDFs (OCR is out of scope for MVP)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS (`faiss-cpu`) with configurable similarity thresholding
- **Generative AI**: Google Gemini API via official SDK (`google-genai`)
- **Testing**: `pytest`, `httpx`

---

## 4. Folder Structure

```text
├── .env.example              # Environment variables template
├── .gitignore                # Ignored files (secrets, venv, caches)
├── requirements.txt          # Python dependencies
├── README.md                 # Primary project documentation
├── SYSTEM_ARCHITECTURE.md    # Complete end-to-end system architecture blueprint
├── BACKEND_ARCHITECTURE.md   # Backend architecture & frontend integration guide
├── docs/                     # Product, technical & design specifications (PRD, TRD, ADRs)
├── backend/                  # Backend core application
│   ├── main.py               # FastAPI entrypoint & static mounting
│   ├── config.py             # Environment configuration (Pydantic Settings)
│   ├── core/                 # Domain constants and disclaimers
│   ├── api/                  # Routes, schemas, REST endpoints
│   ├── document/             # PDF extraction, cleaning, segmentation, store
│   ├── analysis/             # Classification, attention scoring, relationships
│   ├── retrieval/            # Embeddings (MiniLM) & FAISS vector store
│   ├── llm/                  # Google GenAI Gemini client, prompts, validator
│   └── evaluation/           # Evaluation metrics suite
├── frontend/                 # Interactive web interface (HTML5, CSS3, JS)
├── evaluation/               # Benchmark datasets, evaluation scripts & results
├── tests/                    # 130 automated unit, integration, and safety tests
└── data/                     # Sample Terms of Service documents
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
