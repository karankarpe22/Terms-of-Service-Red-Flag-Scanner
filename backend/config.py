import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    """Application configuration loaded from environment variables."""

    # Server settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Generative AI settings (Google Gemini API) - Phase 6
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME: str = os.getenv(
        "GEMINI_MODEL_NAME", os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048"))
    GEMINI_MOCK_MODE: bool = os.getenv("GEMINI_MOCK_MODE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # NLP & Embeddings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Retrieval settings
    RETRIEVAL_SIMILARITY_THRESHOLD: float = float(
        os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.35")
    )
    TOP_K: int = int(os.getenv("TOP_K", os.getenv("RETRIEVAL_TOP_K", "5")))
    RETRIEVAL_TOP_K: int = TOP_K

    # Vector store directory
    VECTOR_STORE_DIR: Path = Path(
        os.getenv("VECTOR_STORE_DIR", str(BASE_DIR / "data" / "vector_store"))
    )

    # Clause Classification Thresholds (Model scores, not legal probabilities)
    CATEGORY_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("CATEGORY_CONFIDENCE_THRESHOLD", "0.32")
    )
    SECONDARY_CATEGORY_THRESHOLD: float = float(
        os.getenv("SECONDARY_CATEGORY_THRESHOLD", "0.28")
    )

    # Attention Detection Thresholds (Deterministic indicator counts)
    ATTENTION_HIGH_THRESHOLD: int = int(os.getenv("ATTENTION_HIGH_THRESHOLD", "2"))
    ATTENTION_MEDIUM_THRESHOLD: int = int(os.getenv("ATTENTION_MEDIUM_THRESHOLD", "1"))

    # Cross-Clause Relationship Analysis Settings (Phase 5)
    RELATIONSHIP_SEMANTIC_THRESHOLD: float = float(
        os.getenv("RELATIONSHIP_SEMANTIC_THRESHOLD", "0.55")
    )
    MAX_CANDIDATE_PAIRS_PER_CLAUSE: int = int(
        os.getenv("MAX_CANDIDATE_PAIRS_PER_CLAUSE", "10")
    )
    MAX_TOTAL_CANDIDATE_PAIRS: int = int(
        os.getenv("MAX_TOTAL_CANDIDATE_PAIRS", "500")
    )
    RELATIONSHIP_MIN_CONFIDENCE: float = float(
        os.getenv("RELATIONSHIP_MIN_CONFIDENCE", "0.50")
    )


settings = Settings()
