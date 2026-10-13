"""SentenceTransformers embedding generator module.

Loads the model once and provides reusable, batched embedding generation.
Normalizes embeddings to unit length for high-performance cosine similarity computation.
"""
import gc
import torch
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import settings

# Restrict PyTorch to 1 CPU thread and disable autograd to stay comfortably under 512MB RAM
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
torch.set_grad_enabled(False)

from pathlib import Path

LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "all-MiniLM-L6-v2"

_EMBEDDING_SERVICE = None


class EmbeddingService:
    """Singleton service for generating dense semantic embeddings."""

    def __init__(self, model_name: str = None):
        if model_name:
            self.model_name = model_name
        elif LOCAL_MODEL_DIR.exists():
            self.model_name = str(LOCAL_MODEL_DIR)
        else:
            self.model_name = settings.EMBEDDING_MODEL

        self._model = SentenceTransformer(self.model_name, device="cpu")
        self._model.eval()
        gc.collect()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Generate normalized vector embeddings for a text or list of texts.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) or (embedding_dim,)
        """
        is_single = isinstance(texts, str)
        input_list = [texts] if is_single else texts

        if not input_list:
            return np.empty((0, 384), dtype=np.float32)

        with torch.no_grad():
            embeddings = self._model.encode(
                input_list,
                batch_size=min(batch_size, 16),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        gc.collect()

        if is_single:
            return embeddings[0]
        return embeddings

    @staticmethod
    def compute_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between normalized embeddings via dot product."""
        # Handles 1D or 2D inputs
        return np.dot(emb_a, emb_b.T)


def get_embedding_service() -> EmbeddingService:
    """Return or initialize the singleton EmbeddingService."""
    global _EMBEDDING_SERVICE
    if _EMBEDDING_SERVICE is None:
        _EMBEDDING_SERVICE = EmbeddingService()
    return _EMBEDDING_SERVICE


def generate_embeddings(texts: Union[str, List[str]]) -> np.ndarray:
    """Generate dense vector embeddings using the singleton EmbeddingService."""
    return get_embedding_service().encode(texts)


# Convenient alias
get_embedder = get_embedding_service
