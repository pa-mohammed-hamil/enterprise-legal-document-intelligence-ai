"""
Legal document reranker.

Path:
    backend/models/huggingface/reranker_model.py

Responsibilities:
    - Rerank retrieved legal-document candidates.
    - Score query/document relevance.
    - Support cross-encoder / Hugging Face reranker models.
    - Provide deterministic result ordering.
    - Support batch scoring.
    - Track model/version metadata.

Non-responsibilities:
    - Initial retrieval.
    - Embedding generation.
    - Legal interpretation.
    - Legal risk decisions.
    - Citation validation.
    - Final answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import logging
import math


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RerankCandidate:
    """A candidate document/chunk returned by retrieval."""

    id: str
    text: str

    retrieval_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RerankResult:
    """A candidate after reranking."""

    id: str
    text: str

    rerank_score: float
    rank: int

    retrieval_score: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RerankingResponse:
    """Complete reranking response."""

    query: str
    results: list[RerankResult]

    model_name: str
    model_version: str

    candidate_count: int
    returned_count: int

    metadata: dict[str, Any] = field(default_factory=dict)


class RerankerModel:
    """
    Cross-encoder based reranker.

    Typical pipeline:

        Query
          │
          ▼
        BM25
          │
          ├─────────────┐
          ▼             ▼
       Vector        Metadata
       Search         Filter
          │             │
          └──────┬──────┘
                 ▼
          Hybrid Candidates
                 │
                 ▼
          RerankerModel
                 │
                 ▼
          Top-K Relevant Chunks
                 │
                 ▼
          Legal RAG Generation

    The reranker improves ordering; it does not determine
    whether a document is legally valid or legally risky.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        model_version: str = "v1",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
        min_score: Optional[float] = None,
        model: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version

        self.device = (
            device
            or self._detect_device()
        )

        self.max_length = max(
            32,
            int(max_length),
        )

        self.batch_size = max(
            1,
            int(batch_size),
        )

        self.min_score = min_score

        self._model = model
        self._tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[
            RerankCandidate
        ],
        *,
        top_k: Optional[int] = None,
    ) -> RerankingResponse:
        """
        Rerank candidates against a query.
        """

        self._validate_query(query)

        candidate_list = list(candidates)

        if not candidate_list:
            return self._build_response(
                query=query,
                results=[],
                candidate_count=0,
            )

        self._validate_candidates(
            candidate_list
        )

        pairs = [
            (query, candidate.text)
            for candidate in candidate_list
        ]

        scores = self.score_pairs(pairs)

        results: list[RerankResult] = []

        for candidate, score in zip(
            candidate_list,
            scores,
        ):
            if (
                self.min_score is not None
                and score < self.min_score
            ):
                continue

            results.append(
                RerankResult(
                    id=candidate.id,
                    text=candidate.text,
                    rerank_score=score,
                    rank=0,
                    retrieval_score=(
                        candidate.retrieval_score
                    ),
                    metadata=dict(
                        candidate.metadata
                    ),
                )
            )

        results.sort(
            key=lambda result: (
                -result.rerank_score,
                -result.retrieval_score,
                result.id,
            )
        )

        if top_k is not None:
            top_k = max(0, int(top_k))
            results = results[:top_k]

        for rank, result in enumerate(
            results,
            start=1,
        ):
            result.rank = rank

        return self._build_response(
            query=query,
            results=results,
            candidate_count=len(candidate_list),
        )

    def score_pairs(
        self,
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> list[float]:
        """
        Score query/document pairs.

        Uses the loaded cross-encoder when available.
        """

        if not pairs:
            return []

        self._validate_pairs(pairs)

        model = self._get_model()

        # CrossEncoder-style interface.
        if hasattr(model, "predict"):
            return self._predict_cross_encoder(
                model,
                pairs,
            )

        # Generic Hugging Face model.
        return self._predict_huggingface(
            pairs
        )

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """Convenience API for scoring documents."""

        return self.score_pairs(
            [
                (query, document)
                for document in documents
            ]
        )

    def rerank_dicts(
        self,
        query: str,
        candidates: Sequence[
            dict[str, Any]
        ],
        *,
        top_k: Optional[int] = None,
    ) -> RerankingResponse:
        """
        Rerank dictionary-based retrieval results.

        Expected fields:
            id
            text

        Optional:
            retrieval_score
            metadata
        """

        normalized: list[RerankCandidate] = []

        for index, candidate in enumerate(
            candidates
        ):
            normalized.append(
                RerankCandidate(
                    id=str(
                        candidate.get(
                            "id",
                            index,
                        )
                    ),
                    text=str(
                        candidate.get(
                            "text",
                            "",
                        )
                    ),
                    retrieval_score=float(
                        candidate.get(
                            "retrieval_score",
                            candidate.get(
                                "score",
                                0.0,
                            ),
                        )
                    ),
                    metadata=dict(
                        candidate.get(
                            "metadata",
                            {},
                        )
                    ),
                )
            )

        return self.rerank(
            query,
            normalized,
            top_k=top_k,
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return model configuration."""

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "min_score": self.min_score,
            "model_class": (
                self._model.__class__.__name__
                if self._model is not None
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _get_model(self) -> Any:
        """
        Lazily load a SentenceTransformers CrossEncoder.

        If a model was injected during construction, it is reused.
        """

        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import (
                CrossEncoder,
            )
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required "
                "for the default reranker implementation."
            ) from exc

        logger.info(
            "Loading reranker model=%s device=%s",
            self.model_name,
            self.device,
        )

        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
        )

        return self._model

    # ------------------------------------------------------------------
    # Cross encoder inference
    # ------------------------------------------------------------------

    def _predict_cross_encoder(
        self,
        model: Any,
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> list[float]:
        """
        Run CrossEncoder.predict().

        Batch processing prevents very large candidate sets
        from being processed in one operation.
        """

        scores: list[float] = []

        for start in range(
            0,
            len(pairs),
            self.batch_size,
        ):
            batch = pairs[
                start:start + self.batch_size
            ]

            batch_scores = model.predict(
                list(batch),
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            scores.extend(
                self._normalize_scores(
                    batch_scores
                )
            )

        return scores

    # ------------------------------------------------------------------
    # Generic Hugging Face inference
    # ------------------------------------------------------------------

    def _predict_huggingface(
        self,
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> list[float]:
        """
        Generic AutoModel-based scoring.

        Expected output:
            logits with one relevance score.

        This path is useful when a custom model is injected.
        """

        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer is required when using a "
                "generic Hugging Face reranker."
            )

        model = self._model

        if model is None:
            raise RuntimeError(
                "Reranker model is not initialized."
            )

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for generic "
                "Hugging Face reranking."
            ) from exc

        results: list[float] = []

        model.eval()

        for start in range(
            0,
            len(pairs),
            self.batch_size,
        ):
            batch = pairs[
                start:start + self.batch_size
            ]

            queries = [
                pair[0]
                for pair in batch
            ]

            documents = [
                pair[1]
                for pair in batch
            ]

            encoded = self._tokenizer(
                queries,
                documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(
                    self.device
                )
                for key, value in encoded.items()
            }

            with torch.no_grad():
                output = model(
                    **encoded
                )

            logits = output.logits

            if logits.ndim == 1:
                batch_scores = logits

            elif logits.shape[-1] == 1:
                batch_scores = (
                    logits.squeeze(-1)
                )

            else:
                # Binary classification:
                # use positive-class logit.
                batch_scores = logits[:, -1]

            results.extend(
                self._normalize_scores(
                    batch_scores.detach().cpu().tolist()
                )
            )

        return results

    # ------------------------------------------------------------------
    # Score normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        scores: Iterable[Any],
    ) -> list[float]:
        """
        Convert model scores into finite floats.

        Cross-encoder models may return logits rather than
        probabilities. We intentionally preserve ranking
        information while mapping pathological values safely.
        """

        normalized: list[float] = []

        for value in scores:
            try:
                score = float(value)
            except (
                TypeError,
                ValueError,
            ):
                score = 0.0

            if not math.isfinite(score):
                score = 0.0

            normalized.append(score)

        return normalized

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(
        query: str,
    ) -> None:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if len(query) > 100_000:
            raise ValueError(
                "query exceeds the maximum supported length."
            )

    @staticmethod
    def _validate_candidates(
        candidates: Sequence[
            RerankCandidate
        ],
    ) -> None:
        for candidate in candidates:
            if not isinstance(
                candidate,
                RerankCandidate,
            ):
                raise TypeError(
                    "All candidates must be "
                    "RerankCandidate instances."
                )

            if not candidate.id:
                raise ValueError(
                    "Candidate id cannot be empty."
                )

            if not candidate.text.strip():
                raise ValueError(
                    f"Candidate '{candidate.id}' "
                    "contains empty text."
                )

    @staticmethod
    def _validate_pairs(
        pairs: Sequence[
            tuple[str, str]
        ],
    ) -> None:
        for pair in pairs:
            if not isinstance(pair, tuple):
                raise TypeError(
                    "Each pair must be a "
                    "(query, document) tuple."
                )

            if len(pair) != 2:
                raise ValueError(
                    "Each pair must contain "
                    "exactly two values."
                )

            query, document = pair

            if not isinstance(
                query,
                str,
            ):
                raise TypeError(
                    "Pair query must be a string."
                )

            if not isinstance(
                document,
                str,
            ):
                raise TypeError(
                    "Pair document must be a string."
                )

            if not query.strip():
                raise ValueError(
                    "Pair query cannot be empty."
                )

            if not document.strip():
                raise ValueError(
                    "Pair document cannot be empty."
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_response(
        self,
        *,
        query: str,
        results: list[RerankResult],
        candidate_count: int,
    ) -> RerankingResponse:
        return RerankingResponse(
            query=query,
            results=results,
            model_name=self.model_name,
            model_version=self.model_version,
            candidate_count=candidate_count,
            returned_count=len(results),
            metadata={
                "device": self.device,
                "max_length": self.max_length,
                "batch_size": self.batch_size,
            },
        )

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"

        except ImportError:
            pass

        return "cpu"


__all__ = [
    "RerankCandidate",
    "RerankResult",
    "RerankingResponse",
    "RerankerModel",
]