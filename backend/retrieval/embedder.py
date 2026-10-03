"""SentenceTransformers embedding generator module.

Loads the model once and provides reusable, batched embedding generation.
Normalizes embeddings to unit length for high-performance cosine similarity computation.
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.config import settings

_EMBEDDING_SERVICE = None


class EmbeddingService:
    """Singleton service for generating dense semantic embeddings."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = SentenceTransformer(self.model_name)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Generate normalized vector embeddings for a text or list of texts.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) or (embedding_dim,)
        """
        is_single = isinstance(texts, str)
        input_list = [texts] if is_single else texts

        if not input_list:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self._model.encode(
            input_list,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

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
