# backend/rag/embeddings/hf_embeddings.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
import math
import threading
import time


@dataclass(slots=True)
class HuggingFaceEmbeddingConfig:
    """
    Hugging Face / SentenceTransformer embedding adapter config.

    This adapter intentionally stays lightweight. Model lifecycle,
    caching, batching orchestration and retrieval concerns belong to
    their dedicated layers.
    """

    model_name: str = "BAAI/bge-base-en-v1.5"
    model_version: str = "v1"

    device: str = "auto"

    batch_size: int = 32
    max_seq_length: int = 512

    normalize_embeddings: bool = True

    query_prefix: str = (
        "Represent this sentence for searching "
        "relevant passages: "
    )

    document_prefix: str = ""

    trust_remote_code: bool = False

    show_progress_bar: bool = False

    convert_to_numpy: bool = True

    expected_dimension: Optional[int] = 768

    lazy_load: bool = True


@dataclass(slots=True)
class HFEmbeddingResult:
    embeddings: list[list[float]]

    model_name: str
    model_version: str

    dimension: int

    normalized: bool

    input_count: int

    latency_ms: float = 0.0

    device: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "embeddings": self.embeddings,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "normalized": self.normalized,
            "input_count": self.input_count,
            "latency_ms": self.latency_ms,
            "device": self.device,
            "metadata": self.metadata,
        }


class HuggingFaceEmbeddingError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "HF_EMBEDDING_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class HuggingFaceEmbeddings:
    """
    Hugging Face embedding provider adapter.

    Responsibilities:
        - load SentenceTransformer model
        - document embedding
        - query embedding
        - batching
        - BGE query/document instructions
        - vector normalization
        - dimension validation
        - model metadata

    Not responsible for:
        - Redis caching
        - Qdrant
        - BM25
        - retrieval
        - reranking
        - permissions
        - chunking

    Those responsibilities belong to the surrounding RAG layers.

    Typical architecture:

        EmbeddingService
              |
              v
        HuggingFaceEmbeddings
              |
              v
        SentenceTransformer
              |
              v
        BAAI/bge-base-en-v1.5
    """

    def __init__(
        self,
        config: Optional[
            HuggingFaceEmbeddingConfig
        ] = None,
        *,
        model: Any = None,
    ) -> None:
        self.config = (
            config
            or HuggingFaceEmbeddingConfig()
        )

        self._model = model
        self._lock = threading.RLock()

        self._resolved_device: Optional[
            str
        ] = None

        self._dimension: Optional[int] = (
            self.config.expected_dimension
        )

        self._validate_config()

        if (
            self._model is None
            and not self.config.lazy_load
        ):
            self._load_model()

        elif self._model is not None:
            self._configure_model(
                self._model
            )

    # ==========================================================
    # Compatibility properties
    # ==========================================================

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def model_version(self) -> str:
        return self.config.model_version

    @property
    def dimension(self) -> Optional[int]:
        if self._dimension is not None:
            return self._dimension

        model = self._ensure_model()

        try:
            dimension = (
                model
                .get_sentence_embedding_dimension()
            )

            if dimension is not None:
                self._dimension = int(
                    dimension
                )

        except Exception:
            pass

        return self._dimension

    @property
    def device(self) -> Optional[str]:
        if self._resolved_device:
            return self._resolved_device

        if self._model is not None:
            model_device = getattr(
                self._model,
                "device",
                None,
            )

            if model_device is not None:
                return str(model_device)

        return None

    @property
    def model(self) -> Any:
        return self._ensure_model()

    # ==========================================================
    # Document APIs
    # ==========================================================

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> HFEmbeddingResult:
        """
        Main document embedding API used by EmbeddingService.
        """

        return self._encode(
            texts,
            is_query=False,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        LangChain-style compatibility API.
        """

        return self.encode_documents(
            texts
        ).embeddings

    def encode(
        self,
        texts: Sequence[str] | str,
    ) -> list[list[float]] | list[float]:
        """
        Generic compatibility API.

        A string returns one vector.
        A sequence returns multiple vectors.
        """

        if isinstance(texts, str):
            result = self.encode_documents(
                [texts]
            )

            if not result.embeddings:
                return []

            return result.embeddings[0]

        return self.encode_documents(
            texts
        ).embeddings

    # ==========================================================
    # Query APIs
    # ==========================================================

    def encode_query(
        self,
        text: str,
    ) -> HFEmbeddingResult:

        if text is None:
            text = ""

        return self._encode(
            [text],
            is_query=True,
        )

    def encode_queries(
        self,
        texts: Sequence[str],
    ) -> HFEmbeddingResult:

        return self._encode(
            texts,
            is_query=True,
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        LangChain-style compatibility API.
        """

        result = self.encode_query(
            text
        )

        if not result.embeddings:
            return []

        return result.embeddings[0]

    # ==========================================================
    # Core inference
    # ==========================================================

    def _encode(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
    ) -> HFEmbeddingResult:

        started = time.perf_counter()

        values = [
            self._prepare_text(
                text,
                is_query=is_query,
            )
            for text in texts
        ]

        if not values:
            return HFEmbeddingResult(
                embeddings=[],
                model_name=self.model_name,
                model_version=(
                    self.model_version
                ),
                dimension=(
                    self.dimension or 0
                ),
                normalized=(
                    self.config
                    .normalize_embeddings
                ),
                input_count=0,
                latency_ms=0.0,
                device=self.device,
            )

        model = self._ensure_model()

        try:
            with self._lock:
                raw = model.encode(
                    values,
                    batch_size=(
                        self.config.batch_size
                    ),
                    show_progress_bar=(
                        self.config
                        .show_progress_bar
                    ),
                    convert_to_numpy=(
                        self.config
                        .convert_to_numpy
                    ),
                    normalize_embeddings=(
                        self.config
                        .normalize_embeddings
                    ),
                )

        except TypeError:
            # Compatibility fallback for injected/mock/custom
            # SentenceTransformer-like models.
            try:
                raw = model.encode(
                    values
                )

            except Exception as exc:
                raise (
                    HuggingFaceEmbeddingError(
                        "Hugging Face embedding "
                        "inference failed.",
                        code=(
                            "INFERENCE_FAILED"
                        ),
                        cause=exc,
                    )
                ) from exc

        except Exception as exc:
            raise HuggingFaceEmbeddingError(
                "Hugging Face embedding "
                "inference failed.",
                code="INFERENCE_FAILED",
                cause=exc,
            ) from exc

        embeddings = (
            self._convert_embeddings(
                raw
            )
        )

        if (
            len(embeddings)
            != len(values)
        ):
            raise HuggingFaceEmbeddingError(
                "Embedding model returned "
                "an unexpected number of "
                "vectors.",
                code=(
                    "BATCH_SIZE_MISMATCH"
                ),
            )

        for vector in embeddings:
            self._validate_vector(
                vector
            )

        if (
            self.config
            .normalize_embeddings
        ):
            # SentenceTransformer normally already does this.
            # Re-normalizing is harmless and protects injected
            # model implementations.
            embeddings = [
                self._normalize_vector(
                    vector
                )
                for vector in embeddings
            ]

        dimension = (
            len(embeddings[0])
            if embeddings
            else (
                self.dimension or 0
            )
        )

        self._validate_dimension(
            dimension
        )

        self._dimension = dimension

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return HFEmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            model_version=(
                self.model_version
            ),
            dimension=dimension,
            normalized=(
                self.config
                .normalize_embeddings
            ),
            input_count=len(values),
            latency_ms=round(
                latency_ms,
                3,
            ),
            device=self.device,
            metadata={
                "input_type": (
                    "query"
                    if is_query
                    else "document"
                ),
                "batch_size":
                    self.config.batch_size,
                "max_seq_length":
                    self.config
                    .max_seq_length,
            },
        )

    # ==========================================================
    # Model lifecycle
    # ==========================================================

    def _ensure_model(
        self,
    ) -> Any:

        if self._model is None:
            self._load_model()

        if self._model is None:
            raise HuggingFaceEmbeddingError(
                "Embedding model could not "
                "be initialized.",
                code="MODEL_NOT_LOADED",
            )

        return self._model

    def _load_model(
        self,
    ) -> None:

        with self._lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import (
                    SentenceTransformer,
                )

            except ImportError as exc:
                raise (
                    HuggingFaceEmbeddingError(
                        "sentence-transformers "
                        "is required. Install it "
                        "with: pip install "
                        "sentence-transformers",
                        code=(
                            "DEPENDENCY_MISSING"
                        ),
                        cause=exc,
                    )
                ) from exc

            device = (
                self._resolve_device()
            )

            kwargs: dict[str, Any] = {
                "device": device,
                "trust_remote_code": (
                    self.config
                    .trust_remote_code
                ),
            }

            try:
                self._model = (
                    SentenceTransformer(
                        self.config
                        .model_name,
                        **kwargs,
                    )
                )

            except TypeError:
                # Older sentence-transformers releases may not
                # accept every constructor argument.
                kwargs.pop(
                    "trust_remote_code",
                    None,
                )

                try:
                    self._model = (
                        SentenceTransformer(
                            self.config
                            .model_name,
                            **kwargs,
                        )
                    )

                except Exception as exc:
                    raise (
                        HuggingFaceEmbeddingError(
                            "Unable to load "
                            f"embedding model "
                            f"{self.config.model_name}.",
                            code=(
                                "MODEL_LOAD_FAILED"
                            ),
                            cause=exc,
                        )
                    ) from exc

            except Exception as exc:
                raise HuggingFaceEmbeddingError(
                    "Unable to load embedding "
                    f"model "
                    f"{self.config.model_name}.",
                    code=(
                        "MODEL_LOAD_FAILED"
                    ),
                    cause=exc,
                ) from exc

            self._resolved_device = device

            self._configure_model(
                self._model
            )

    def _configure_model(
        self,
        model: Any,
    ) -> None:

        if hasattr(
            model,
            "max_seq_length",
        ):
            try:
                model.max_seq_length = (
                    self.config
                    .max_seq_length
                )
            except Exception:
                pass

        try:
            dimension = (
                model
                .get_sentence_embedding_dimension()
            )

            if dimension is not None:
                dimension = int(
                    dimension
                )

                self._validate_dimension(
                    dimension
                )

                self._dimension = dimension

        except (
            HuggingFaceEmbeddingError
        ):
            raise

        except Exception:
            pass

        model_device = getattr(
            model,
            "device",
            None,
        )

        if model_device is not None:
            self._resolved_device = str(
                model_device
            )

    def _resolve_device(
        self,
    ) -> str:

        configured = (
            self.config.device.strip()
            .lower()
        )

        if configured != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            mps = getattr(
                torch.backends,
                "mps",
                None,
            )

            if (
                mps is not None
                and mps.is_available()
            ):
                return "mps"

        except ImportError:
            pass

        return "cpu"

    # ==========================================================
    # Text preparation
    # ==========================================================

    def _prepare_text(
        self,
        text: Any,
        *,
        is_query: bool,
    ) -> str:

        if text is None:
            value = ""
        elif isinstance(
            text,
            str,
        ):
            value = text
        else:
            value = str(text)

        value = value.strip()

        if is_query:
            prefix = (
                self.config.query_prefix
            )
        else:
            prefix = (
                self.config
                .document_prefix
            )

        if prefix:
            value = (
                prefix + value
            )

        return value

    # ==========================================================
    # Result conversion
    # ==========================================================

    @staticmethod
    def _convert_embeddings(
        value: Any,
    ) -> list[list[float]]:

        if value is None:
            return []

        if hasattr(
            value,
            "detach",
        ):
            value = (
                value.detach()
            )

        if hasattr(
            value,
            "cpu",
        ):
            value = value.cpu()

        if hasattr(
            value,
            "tolist",
        ):
            value = value.tolist()

        if not isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            raise HuggingFaceEmbeddingError(
                "Unsupported embedding "
                "output type.",
                code=(
                    "INVALID_MODEL_OUTPUT"
                ),
            )

        if not value:
            return []

        # Single vector returned by a custom model.
        if isinstance(
            value[0],
            (
                int,
                float,
            ),
        ):
            return [
                [
                    float(item)
                    for item in value
                ]
            ]

        embeddings: list[
            list[float]
        ] = []

        for vector in value:
            if hasattr(
                vector,
                "tolist",
            ):
                vector = (
                    vector.tolist()
                )

            embeddings.append(
                [
                    float(item)
                    for item in vector
                ]
            )

        return embeddings

    # ==========================================================
    # Vector validation
    # ==========================================================

    @staticmethod
    def _validate_vector(
        vector: Sequence[float],
    ) -> None:

        if not vector:
            raise HuggingFaceEmbeddingError(
                "Embedding vector is empty.",
                code="EMPTY_VECTOR",
            )

        for value in vector:
            try:
                number = float(value)

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise (
                    HuggingFaceEmbeddingError(
                        "Embedding contains "
                        "non-numeric values.",
                        code=(
                            "INVALID_VECTOR"
                        ),
                        cause=exc,
                    )
                ) from exc

            if not math.isfinite(
                number
            ):
                raise HuggingFaceEmbeddingError(
                    "Embedding contains NaN "
                    "or infinite values.",
                    code=(
                        "INVALID_VECTOR"
                    ),
                )

    def _validate_dimension(
        self,
        dimension: int,
    ) -> None:

        if dimension <= 0:
            raise HuggingFaceEmbeddingError(
                "Invalid embedding dimension.",
                code=(
                    "INVALID_DIMENSION"
                ),
            )

        expected = (
            self.config
            .expected_dimension
        )

        if (
            expected is not None
            and dimension != expected
        ):
            raise HuggingFaceEmbeddingError(
                "Embedding dimension "
                f"{dimension} does not match "
                f"configured dimension "
                f"{expected}.",
                code=(
                    "DIMENSION_MISMATCH"
                ),
            )

    @staticmethod
    def _normalize_vector(
        vector: Sequence[float],
    ) -> list[float]:

        norm = math.sqrt(
            sum(
                float(value)
                * float(value)
                for value in vector
            )
        )

        if norm == 0:
            return [
                float(value)
                for value in vector
            ]

        return [
            float(value) / norm
            for value in vector
        ]

    # ==========================================================
    # Similarity
    # ==========================================================

    @staticmethod
    def cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:

        if len(left) != len(right):
            raise HuggingFaceEmbeddingError(
                "Embedding dimensions do not "
                "match.",
                code=(
                    "SIMILARITY_DIMENSION_"
                    "MISMATCH"
                ),
            )

        if not left:
            return 0.0

        left_norm = math.sqrt(
            sum(
                float(value) ** 2
                for value in left
            )
        )

        right_norm = math.sqrt(
            sum(
                float(value) ** 2
                for value in right
            )
        )

        if (
            left_norm == 0
            or right_norm == 0
        ):
            return 0.0

        dot = sum(
            float(a) * float(b)
            for a, b in zip(
                left,
                right,
            )
        )

        return max(
            -1.0,
            min(
                1.0,
                dot
                / (
                    left_norm
                    * right_norm
                ),
            ),
        )

    def similarity(
        self,
        query: str,
        document: str,
    ) -> float:

        query_vector = (
            self.embed_query(
                query
            )
        )

        document_result = (
            self.encode_documents(
                [document]
            )
        )

        if (
            not document_result
            .embeddings
        ):
            return 0.0

        return self.cosine_similarity(
            query_vector,
            document_result
            .embeddings[0],
        )

    # ==========================================================
    # Health / metadata
    # ==========================================================

    def get_model_info(
        self,
    ) -> dict[str, Any]:

        return {
            "provider": "huggingface",
            "backend":
                "sentence-transformers",
            "model_name":
                self.model_name,
            "model_version":
                self.model_version,
            "dimension":
                self.dimension,
            "device":
                self.device,
            "max_seq_length":
                self.config
                .max_seq_length,
            "batch_size":
                self.config.batch_size,
            "normalize_embeddings":
                self.config
                .normalize_embeddings,
            "query_prefix":
                self.config.query_prefix,
            "document_prefix":
                self.config
                .document_prefix,
            "loaded":
                self._model is not None,
        }

    def health_check(
        self,
    ) -> dict[str, Any]:

        started = time.perf_counter()

        try:
            result = self.encode_query(
                "embedding health check"
            )

            healthy = bool(
                result.embeddings
                and result.dimension > 0
            )

            return {
                "healthy": healthy,
                "model_name":
                    self.model_name,
                "model_version":
                    self.model_version,
                "dimension":
                    result.dimension,
                "device":
                    self.device,
                "latency_ms": round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000,
                    3,
                ),
            }

        except Exception as exc:
            return {
                "healthy": False,
                "model_name":
                    self.model_name,
                "model_version":
                    self.model_version,
                "device":
                    self.device,
                "error": str(exc),
                "latency_ms": round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000,
                    3,
                ),
            }

    # ==========================================================
    # Config validation
    # ==========================================================

    def _validate_config(
        self,
    ) -> None:

        if not self.config.model_name:
            raise ValueError(
                "model_name is required."
            )

        if self.config.batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        if (
            self.config
            .max_seq_length
            <= 0
        ):
            raise ValueError(
                "max_seq_length must "
                "be > 0."
            )

        if (
            self.config
            .expected_dimension
            is not None
            and self.config
            .expected_dimension
            <= 0
        ):
            raise ValueError(
                "expected_dimension must "
                "be > 0."
            )


# Backward-compatible alias if other modules use this naming.
HFEmbeddings = HuggingFaceEmbeddings


def create_hf_embeddings(
    *,
    model_name: str = (
        "BAAI/bge-base-en-v1.5"
    ),
    model_version: str = "v1",
    device: str = "auto",
    batch_size: int = 32,
    dimension: Optional[int] = 768,
    normalize_embeddings: bool = True,
) -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        HuggingFaceEmbeddingConfig(
            model_name=model_name,
            model_version=model_version,
            device=device,
            batch_size=batch_size,
            expected_dimension=dimension,
            normalize_embeddings=(
                normalize_embeddings
            ),
        )
    )