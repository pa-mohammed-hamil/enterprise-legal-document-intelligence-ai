# backend/models/huggingface/embedding_model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class EmbeddingResult:
    """
    Result returned by the embedding model.
    """

    embeddings: List[List[float]]
    model_name: str
    model_version: Optional[str]
    dimension: int
    normalized: bool
    metadata: Dict[str, Any]


class EmbeddingModel:
    """
    Production abstraction around a sentence-transformer / Hugging Face
    embedding model.

    Responsibilities:
        - encode documents and queries
        - batching
        - vector normalization
        - dimension validation
        - model metadata
        - CPU/GPU device selection

    The class deliberately keeps model inference separate from:
        - chunking
        - vector database storage
        - retrieval
        - reranking
        - caching
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        *,
        model_version: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        max_seq_length: int = 512,
        query_prefix: str = "",
        document_prefix: str = "",
        model: Optional[Any] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if max_seq_length < 1:
            raise ValueError("max_seq_length must be >= 1")

        self.model_name = model_name
        self.model_version = model_version
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

        self.device = device or self._detect_device()

        self.model = model or self._load_model()

        self._configure_model()

        self.dimension = self._get_dimension()

    def encode_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for document/chunk text.
        """

        cleaned = self._validate_texts(texts)

        prepared = [
            f"{self.document_prefix}{text}"
            for text in cleaned
        ]

        vectors = self._encode(
            prepared,
            batch_size=batch_size,
        )

        return self._build_result(vectors)

    def encode_query(
        self,
        query: str,
    ) -> EmbeddingResult:
        """
        Generate an embedding for a search query.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")

        prepared = f"{self.query_prefix}{query.strip()}"

        vectors = self._encode(
            [prepared],
            batch_size=1,
        )

        return self._build_result(vectors)

    def encode_queries(
        self,
        queries: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple queries.
        """

        cleaned = self._validate_texts(queries)

        prepared = [
            f"{self.query_prefix}{query}"
            for query in cleaned
        ]

        vectors = self._encode(
            prepared,
            batch_size=batch_size,
        )

        return self._build_result(vectors)

    def similarity(
        self,
        first: Sequence[float],
        second: Sequence[float],
    ) -> float:
        """
        Calculate cosine similarity.

        If embeddings are normalized, this becomes a dot product.
        """

        a = np.asarray(first, dtype=np.float32)
        b = np.asarray(second, dtype=np.float32)

        if a.ndim != 1 or b.ndim != 1:
            raise ValueError(
                "Embeddings must be one-dimensional vectors"
            )

        if a.shape != b.shape:
            raise ValueError(
                "Embedding dimensions do not match"
            )

        denominator = (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(a, b) / denominator
        )

    def validate_dimension(
        self,
        vector: Sequence[float],
    ) -> None:
        """
        Validate vector dimensionality before inserting into Qdrant.
        """

        if len(vector) != self.dimension:
            raise ValueError(
                f"Invalid embedding dimension: "
                f"expected={self.dimension}, "
                f"received={len(vector)}"
            )

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "dimension": self.dimension,
            "normalize_embeddings": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
        }

    def _encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Run model inference.
        """

        if not texts:
            return []

        effective_batch_size = (
            batch_size or self.batch_size
        )

        if effective_batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1"
            )

        all_vectors: List[List[float]] = []

        for start in range(
            0,
            len(texts),
            effective_batch_size,
        ):
            batch = list(
                texts[
                    start:start + effective_batch_size
                ]
            )

            vectors = self._encode_batch(batch)

            all_vectors.extend(
                self._convert_vectors(vectors)
            )

        for vector in all_vectors:
            self.validate_dimension(vector)

        return all_vectors

    def _encode_batch(
        self,
        texts: Sequence[str],
    ) -> Any:
        """
        Supports SentenceTransformer-style models.

        The injected model can therefore be mocked easily in tests.
        """

        if not hasattr(self.model, "encode"):
            raise TypeError(
                "Embedding model must expose an encode() method"
            )

        return self.model.encode(
            list(texts),
            batch_size=len(texts),
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _convert_vectors(
        self,
        vectors: Any,
    ) -> List[List[float]]:
        array = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise ValueError(
                "Embedding model returned invalid shape"
            )

        return array.tolist()

    def _build_result(
        self,
        vectors: List[List[float]],
    ) -> EmbeddingResult:
        return EmbeddingResult(
            embeddings=vectors,
            model_name=self.model_name,
            model_version=self.model_version,
            dimension=self.dimension,
            normalized=self.normalize_embeddings,
            metadata={
                "device": self.device,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
            },
        )

    def _load_model(self) -> Any:
        """
        Lazy import so the rest of the application can be imported
        without requiring sentence-transformers during unit tests.
        """

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required to load "
                "the embedding model"
            ) from exc

        model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        return model

    def _configure_model(self) -> None:
        """
        Configure tokenizer/model limits when supported.
        """

        if hasattr(
            self.model,
            "max_seq_length",
        ):
            self.model.max_seq_length = (
                self.max_seq_length
            )

    def _get_dimension(self) -> int:
        """
        Determine vector dimension from the loaded model.
        """

        if hasattr(
            self.model,
            "get_sentence_embedding_dimension",
        ):
            dimension = (
                self.model
                .get_sentence_embedding_dimension()
            )

            if dimension is None:
                raise RuntimeError(
                    "Unable to determine embedding dimension"
                )

            return int(dimension)

        if hasattr(self.model, "dimension"):
            return int(self.model.dimension)

        # Last-resort inference.
        probe = self._encode_batch(["dimension probe"])

        vectors = self._convert_vectors(probe)

        if not vectors:
            raise RuntimeError(
                "Embedding model returned no vector"
            )

        return len(vectors[0])

    def _validate_texts(
        self,
        texts: Sequence[str],
    ) -> List[str]:
        if not isinstance(texts, Sequence):
            raise TypeError(
                "texts must be a sequence of strings"
            )

        cleaned: List[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"text at index {index} must be a string"
                )

            value = text.strip()

            if not value:
                raise ValueError(
                    f"text at index {index} is empty"
                )

            cleaned.append(value)

        return cleaned

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            if hasattr(
                torch.backends,
                "mps",
            ) and torch.backends.mps.is_available():
                return "mps"

        except ImportError:
            pass

        return "cpu"