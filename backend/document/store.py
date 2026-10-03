"""In-memory document storage registry for active documents during runtime."""
from typing import Dict, Optional, List
from backend.document.models import Document

_DOCUMENT_STORE: Dict[str, Document] = {}


def save_document(document: Document) -> None:
    """Store or update a processed document in memory."""
    _DOCUMENT_STORE[document.document_id] = document


def get_document(document_id: str) -> Optional[Document]:
    """Retrieve a stored document by its ID."""
    return _DOCUMENT_STORE.get(document_id)


def list_documents() -> List[Document]:
    """Return all stored documents."""
    return list(_DOCUMENT_STORE.values())


def clear_document_store() -> None:
    """Clear all documents from memory (primarily for test isolation)."""
    _DOCUMENT_STORE.clear()
