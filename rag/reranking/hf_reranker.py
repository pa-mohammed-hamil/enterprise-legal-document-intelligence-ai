"""
Hugging Face Cross-Encoder Reranker
===================================

Production-oriented Hugging Face reranker for the Enterprise Legal
Document Intelligence AI platform.

Features
--------
- Cross-encoder sequence-classification reranking
- Lazy model loading
- CPU / CUDA / MPS device selection
- Batched inference
- Configurable score normalization
- Long-document truncation
- Metadata preservation
- Stable ranking
- Async interface
- Graceful fallback when the model is unavailable
- Singleton-friendly lifecycle
- Dependency injection support
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional ML dependencies
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    _TRANSFORMERS_AVAILABLE = True

except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

    _TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ScoreNormalization(str, Enum):
    """Supported reranker score normalization methods."""

    NONE = "none"
    SIGMOID = "sigmoid"
    SOFTMAX = "softmax"


@dataclass(slots=True)
class HFRerankerConfig:
    """
    Hugging Face reranker configuration.

    The default model can be overridden through:

        RERANKER_MODEL_NAME
    """

    model_name: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_NAME",
            "BAAI/bge-reranker-base",
        )
    )

    tokenizer_name: str | None = None

    device: str | None = None

    batch_size: int = 16

    max_length: int = 512

    normalization: ScoreNormalization = ScoreNormalization.SIGMOID

    trust_remote_code: bool = False

    local_files_only: bool = False

    use_fp16: bool = True

    fallback_to_original_order: bool = True

    min_score: float | None = None

    max_results: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        if self.max_results is not None and self.max_results <= 0:
            raise ValueError("max_results must be positive")

        if isinstance(self.normalization, str):
            self.normalization = ScoreNormalization(
                self.normalization.lower()
            )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RerankerDocument:
    """
    Normalized document consumed by the reranker.
    """

    id: str
    text: str

    metadata: dict[str, Any] = field(default_factory=dict)

    retrieval_score: float | None = None


@dataclass(slots=True)
class RerankedDocument:
    """
    Document returned after cross-encoder reranking.
    """

    id: str
    text: str

    score: float

    rank: int

    metadata: dict[str, Any] = field(default_factory=dict)

    retrieval_score: float | None = None

    raw_score: float | None = None


# ---------------------------------------------------------------------------
# Main implementation
# ---------------------------------------------------------------------------


class HFReranker:
    """
    Hugging Face cross-encoder reranker.

    Example
    -------
    reranker = HFReranker()

    results = await reranker.rerank(
        query="What is the termination notice period?",
        documents=[
            {
                "id": "chunk-1",
                "text": "Either party may terminate with 30 days notice.",
            },
            {
                "id": "chunk-2",
                "text": "Invoices are payable within 45 days.",
            },
        ],
    )
    """

    def __init__(
        self,
        config: HFRerankerConfig | None = None,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config or HFRerankerConfig()

        self._model = model
        self._tokenizer = tokenizer

        self._device: str | None = None

        self._loaded = model is not None and tokenizer is not None

        self._load_lock = threading.Lock()

        if self._loaded:
            self._device = self._resolve_device()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = self._resolve_device()

        return self._device

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Lazily load the tokenizer and sequence-classification model.
        """

        if self._loaded:
            return

        if not _TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "HFReranker requires 'torch' and 'transformers'. "
                "Install them before enabling Hugging Face reranking."
            )

        with self._load_lock:
            if self._loaded:
                return

            logger.info(
                "Loading Hugging Face reranker: %s",
                self.config.model_name,
            )

            tokenizer_name = (
                self.config.tokenizer_name
                or self.config.model_name
            )

            self._tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
            )

            self._model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    trust_remote_code=self.config.trust_remote_code,
                    local_files_only=self.config.local_files_only,
                )
            )

            self._device = self._resolve_device()

            self._model.to(self._device)
            self._model.eval()

            if (
                self.config.use_fp16
                and self._device.startswith("cuda")
            ):
                self._model.half()

            self._loaded = True

            logger.info(
                "Hugging Face reranker loaded on %s",
                self._device,
            )

    def unload(self) -> None:
        """
        Release model resources.
        """

        self._model = None
        self._tokenizer = None
        self._loaded = False

        if (
            _TRANSFORMERS_AVAILABLE
            and torch is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

        logger.info("Hugging Face reranker unloaded")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: Sequence[
            RerankerDocument | Mapping[str, Any] | str
        ],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RerankedDocument]:
        """
        Asynchronously rerank documents against a query.

        Heavy model inference is moved to a worker thread so that the
        FastAPI event loop is not blocked.
        """

        return await asyncio.to_thread(
            self.rerank_sync,
            query,
            documents,
            top_k=top_k,
            min_score=min_score,
        )

    def rerank_sync(
        self,
        query: str,
        documents: Sequence[
            RerankerDocument | Mapping[str, Any] | str
        ],
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RerankedDocument]:
        """
        Synchronous reranking implementation.
        """

        query = self._validate_query(query)

        normalized_documents = self._normalize_documents(documents)

        if not normalized_documents:
            return []

        top_k = self._resolve_top_k(
            top_k,
            len(normalized_documents),
        )

        threshold = (
            min_score
            if min_score is not None
            else self.config.min_score
        )

        try:
            self.load()

            raw_scores = self._predict(
                query,
                normalized_documents,
            )

            scores = self._normalize_scores(raw_scores)

            results = self._build_results(
                normalized_documents,
                raw_scores,
                scores,
            )

        except Exception:
            logger.exception(
                "Hugging Face reranking failed for model %s",
                self.config.model_name,
            )

            if not self.config.fallback_to_original_order:
                raise

            results = self._fallback_results(
                normalized_documents
            )

        if threshold is not None:
            results = [
                result
                for result in results
                if result.score >= threshold
            ]

        results = results[:top_k]

        # Recalculate ranks after filtering/truncation.
        for index, result in enumerate(results, start=1):
            result.rank = index

        return results

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _predict(
        self,
        query: str,
        documents: Sequence[RerankerDocument],
    ) -> list[float]:
        """
        Run batched cross-encoder inference.
        """

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Reranker model is not loaded")

        if torch is None:
            raise RuntimeError("PyTorch is unavailable")

        scores: list[float] = []

        batch_size = self.config.batch_size

        for start in range(
            0,
            len(documents),
            batch_size,
        ):
            batch = documents[start : start + batch_size]

            queries = [query] * len(batch)
            passages = [document.text for document in batch]

            encoded = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            with torch.inference_mode():
                outputs = self._model(**encoded)

            logits = outputs.logits

            batch_scores = self._extract_scores(logits)

            scores.extend(batch_scores)

        if len(scores) != len(documents):
            raise RuntimeError(
                "Reranker returned an unexpected number of scores"
            )

        return scores

    @staticmethod
    def _extract_scores(logits: Any) -> list[float]:
        """
        Extract relevance scores from common sequence-classification
        output shapes.

        Supported:
            [batch]
            [batch, 1]
            [batch, 2+]

        For multi-class models the final class is treated as relevance.
        """

        if torch is None:
            raise RuntimeError("PyTorch is unavailable")

        if logits.ndim == 1:
            tensor = logits

        elif logits.ndim == 2:
            if logits.shape[1] == 1:
                tensor = logits[:, 0]

            else:
                # Common binary classification convention:
                # final class = relevant.
                tensor = logits[:, -1]

        else:
            raise ValueError(
                f"Unsupported reranker logits shape: "
                f"{tuple(logits.shape)}"
            )

        return [
            float(value)
            for value in tensor.detach().float().cpu().tolist()
        ]

    # ------------------------------------------------------------------
    # Score processing
    # ------------------------------------------------------------------

    def _normalize_scores(
        self,
        raw_scores: Sequence[float],
    ) -> list[float]:

        method = self.config.normalization

        if method is ScoreNormalization.NONE:
            return list(raw_scores)

        if method is ScoreNormalization.SIGMOID:
            return [
                self._sigmoid(score)
                for score in raw_scores
            ]

        if method is ScoreNormalization.SOFTMAX:
            return self._softmax(raw_scores)

        raise ValueError(
            f"Unsupported normalization: {method}"
        )

    @staticmethod
    def _sigmoid(value: float) -> float:
        """
        Numerically stable sigmoid.
        """

        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)

        z = math.exp(value)
        return z / (1.0 + z)

    @staticmethod
    def _softmax(
        values: Sequence[float],
    ) -> list[float]:

        if not values:
            return []

        maximum = max(values)

        exponentials = [
            math.exp(value - maximum)
            for value in values
        ]

        denominator = sum(exponentials)

        if denominator == 0:
            return [0.0 for _ in values]

        return [
            value / denominator
            for value in exponentials
        ]

    # ------------------------------------------------------------------
    # Result creation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_results(
        documents: Sequence[RerankerDocument],
        raw_scores: Sequence[float],
        scores: Sequence[float],
    ) -> list[RerankedDocument]:

        indexed = list(
            zip(
                range(len(documents)),
                documents,
                raw_scores,
                scores,
            )
        )

        # Stable sorting:
        # score DESC followed by original retrieval position.
        indexed.sort(
            key=lambda item: (
                -item[3],
                item[0],
            )
        )

        results: list[RerankedDocument] = []

        for rank, (
            _,
            document,
            raw_score,
            score,
        ) in enumerate(indexed, start=1):

            results.append(
                RerankedDocument(
                    id=document.id,
                    text=document.text,
                    score=float(score),
                    raw_score=float(raw_score),
                    rank=rank,
                    metadata=dict(document.metadata),
                    retrieval_score=document.retrieval_score,
                )
            )

        return results

    @staticmethod
    def _fallback_results(
        documents: Sequence[RerankerDocument],
    ) -> list[RerankedDocument]:
        """
        Preserve retrieval order if model inference fails.

        Existing retrieval scores are retained when available.
        """

        results: list[RerankedDocument] = []

        total = len(documents)

        for index, document in enumerate(
            documents,
            start=1,
        ):
            if document.retrieval_score is not None:
                score = float(document.retrieval_score)
            else:
                # Deterministic decreasing fallback score.
                score = (
                    (total - index + 1) / total
                    if total
                    else 0.0
                )

            results.append(
                RerankedDocument(
                    id=document.id,
                    text=document.text,
                    score=score,
                    raw_score=None,
                    rank=index,
                    metadata=dict(document.metadata),
                    retrieval_score=document.retrieval_score,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Document normalization
    # ------------------------------------------------------------------

    def _normalize_documents(
        self,
        documents: Sequence[
            RerankerDocument | Mapping[str, Any] | str
        ],
    ) -> list[RerankerDocument]:

        normalized: list[RerankerDocument] = []

        for index, item in enumerate(documents):
            if isinstance(item, RerankerDocument):
                document = item

            elif isinstance(item, str):
                document = RerankerDocument(
                    id=str(index),
                    text=item,
                )

            elif isinstance(item, Mapping):
                document = self._mapping_to_document(
                    item,
                    index,
                )

            else:
                raise TypeError(
                    "Each reranker document must be a string, "
                    "mapping, or RerankerDocument"
                )

            if not document.text.strip():
                logger.debug(
                    "Skipping empty reranker document: %s",
                    document.id,
                )
                continue

            normalized.append(document)

        return normalized

    @staticmethod
    def _mapping_to_document(
        item: Mapping[str, Any],
        index: int,
    ) -> RerankerDocument:

        document_id = (
            item.get("id")
            or item.get("chunk_id")
            or item.get("document_id")
            or str(index)
        )

        text = (
            item.get("text")
            or item.get("content")
            or item.get("page_content")
            or ""
        )

        metadata = item.get("metadata") or {}

        if not isinstance(metadata, Mapping):
            metadata = {}

        retrieval_score = (
            item.get("retrieval_score")
            if item.get("retrieval_score") is not None
            else item.get("score")
        )

        try:
            retrieval_score = (
                float(retrieval_score)
                if retrieval_score is not None
                else None
            )
        except (TypeError, ValueError):
            retrieval_score = None

        return RerankerDocument(
            id=str(document_id),
            text=str(text),
            metadata=dict(metadata),
            retrieval_score=retrieval_score,
        )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:

        if self.config.device:
            return self.config.device

        if not _TRANSFORMERS_AVAILABLE or torch is None:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"

        mps = getattr(torch.backends, "mps", None)

        if (
            mps is not None
            and mps.is_available()
        ):
            return "mps"

        return "cpu"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> str:

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        if "\x00" in query:
            raise ValueError(
                "query cannot contain null characters"
            )

        return query

    def _resolve_top_k(
        self,
        top_k: int | None,
        document_count: int,
    ) -> int:

        if top_k is None:
            top_k = self.config.max_results

        if top_k is None:
            return document_count

        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        return min(top_k, document_count)

    # ------------------------------------------------------------------
    # Health information
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """
        Return lightweight runtime information for model-health endpoints.
        """

        return {
            "model": self.model_name,
            "loaded": self.loaded,
            "device": self.device,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "normalization": self.config.normalization.value,
            "transformers_available": _TRANSFORMERS_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# Default instance
# ---------------------------------------------------------------------------


_default_reranker: HFReranker | None = None
_default_lock = threading.Lock()


def get_hf_reranker(
    config: HFRerankerConfig | None = None,
) -> HFReranker:
    """
    Return the process-level default reranker.

    Supplying configuration affects only the first initialization.
    """

    global _default_reranker

    if _default_reranker is not None:
        return _default_reranker

    with _default_lock:
        if _default_reranker is None:
            _default_reranker = HFReranker(config)

    return _default_reranker


async def rerank_documents(
    query: str,
    documents: Sequence[
        RerankerDocument | Mapping[str, Any] | str
    ],
    *,
    top_k: int | None = None,
) -> list[RerankedDocument]:
    """
    Convenience API using the process-level reranker.
    """

    reranker = get_hf_reranker()

    return await reranker.rerank(
        query,
        documents,
        top_k=top_k,
    )


__all__ = [
    "HFReranker",
    "HFRerankerConfig",
    "RerankedDocument",
    "RerankerDocument",
    "ScoreNormalization",
    "get_hf_reranker",
    "rerank_documents",
]