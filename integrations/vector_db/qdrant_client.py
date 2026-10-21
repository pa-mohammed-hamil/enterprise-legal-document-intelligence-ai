"""
Qdrant vector database integration.

Responsibilities:
- Collection management
- Vector upsert
- Similarity search
- Metadata filtering
- Batch operations
- Document/chunk deletion
- Point retrieval
- Health checks

RAG orchestration should remain in the rag/ layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.config import settings

logger = logging.getLogger(__name__)


class QdrantError(Exception):
    """Base Qdrant integration error."""


class QdrantConfigurationError(QdrantError):
    """Raised when Qdrant configuration is invalid."""


class QdrantCollectionError(QdrantError):
    """Raised when collection operations fail."""


class QdrantSearchError(QdrantError):
    """Raised when vector search fails."""


@dataclass(frozen=True)
class VectorPoint:
    """Normalized vector point."""

    id: str | int
    vector: Sequence[float]
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class SearchResult:
    """Normalized Qdrant search result."""

    id: str | int
    score: float
    payload: dict[str, Any]


class QdrantVectorStore:
    """
    Application-level Qdrant client.

    Designed for legal-document chunk embeddings and
    permission-aware RAG retrieval.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        distance: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url or getattr(
            settings,
            "QDRANT_URL",
            "http://localhost:6333",
        )

        self.api_key = api_key or getattr(
            settings,
            "QDRANT_API_KEY",
            None,
        )

        self.collection_name = collection_name or getattr(
            settings,
            "QDRANT_COLLECTION",
            "legal_documents",
        )

        self.vector_size = vector_size or getattr(
            settings,
            "QDRANT_VECTOR_SIZE",
            1024,
        )

        self.distance = distance or getattr(
            settings,
            "QDRANT_DISTANCE",
            "COSINE",
        )

        self.timeout = timeout

        self.client = self._create_client()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(
        self,
        *,
        collection_name: str | None = None,
        vector_size: int | None = None,
        recreate: bool = False,
    ) -> None:
        """Create the configured vector collection."""
        name = collection_name or self.collection_name
        size = vector_size or self.vector_size

        try:
            if self.collection_exists(name):
                if not recreate:
                    return

                self.delete_collection(name)

            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=size,
                    distance=self._distance(),
                ),
            )

            logger.info(
                "Qdrant collection created",
                extra={
                    "collection": name,
                    "vector_size": size,
                },
            )

        except Exception as exc:
            logger.exception(
                "Failed to create Qdrant collection",
                extra={"collection": name},
            )

            raise QdrantCollectionError(
                f"Failed to create collection: {name}"
            ) from exc

    def collection_exists(
        self,
        collection_name: str | None = None,
    ) -> bool:
        """Check whether a collection exists."""
        name = collection_name or self.collection_name

        try:
            collections = self.client.get_collections()

            return any(
                collection.name == name
                for collection in collections.collections
            )

        except Exception as exc:
            raise QdrantCollectionError(
                f"Failed to inspect collection: {name}"
            ) from exc

    def delete_collection(
        self,
        collection_name: str | None = None,
    ) -> None:
        """Delete a collection."""
        name = collection_name or self.collection_name

        try:
            self.client.delete_collection(
                collection_name=name,
            )

        except Exception as exc:
            raise QdrantCollectionError(
                f"Failed to delete collection: {name}"
            ) from exc

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(
        self,
        points: Iterable[VectorPoint],
        *,
        collection_name: str | None = None,
        batch_size: int = 128,
        wait: bool = True,
    ) -> int:
        """
        Upsert vectors in batches.

        Returns the number of processed points.
        """
        name = collection_name or self.collection_name

        normalized = list(points)

        if not normalized:
            return 0

        processed = 0

        try:
            for batch in self._batches(
                normalized,
                batch_size,
            ):
                qdrant_points = [
                    models.PointStruct(
                        id=point.id,
                        vector=list(point.vector),
                        payload=dict(point.payload),
                    )
                    for point in batch
                ]

                self.client.upsert(
                    collection_name=name,
                    points=qdrant_points,
                    wait=wait,
                )

                processed += len(batch)

            logger.info(
                "Vectors upserted",
                extra={
                    "collection": name,
                    "count": processed,
                },
            )

            return processed

        except Exception as exc:
            logger.exception(
                "Qdrant upsert failed",
                extra={"collection": name},
            )

            raise QdrantError(
                "Failed to upsert vectors."
            ) from exc

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        *,
        vector: Sequence[float],
        limit: int = 10,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
        collection_name: str | None = None,
        with_payload: bool = True,
    ) -> list[SearchResult]:
        """
        Perform semantic vector search.
        """
        name = collection_name or self.collection_name

        try:
            response = self.client.query_points(
                collection_name=name,
                query=list(vector),
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
            )

            return [
                SearchResult(
                    id=result.id,
                    score=float(result.score),
                    payload=dict(result.payload or {}),
                )
                for result in response.points
            ]

        except Exception as exc:
            logger.exception(
                "Qdrant vector search failed",
                extra={"collection": name},
            )

            raise QdrantSearchError(
                "Vector search failed."
            ) from exc

    # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------

    def build_filter(
        self,
        *,
        tenant_id: str | None = None,
        organization_id: str | None = None,
        document_id: str | None = None,
        document_version_id: str | None = None,
        document_type: str | None = None,
        user_id: str | None = None,
        tags: Sequence[str] | None = None,
    ) -> models.Filter | None:
        """
        Build a permission-aware metadata filter.

        Payload field names should remain synchronized with
        the document_chunk indexing pipeline.
        """
        conditions: list[models.FieldCondition] = []

        if tenant_id:
            conditions.append(
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(
                        value=tenant_id,
                    ),
                )
            )

        if organization_id:
            conditions.append(
                models.FieldCondition(
                    key="organization_id",
                    match=models.MatchValue(
                        value=organization_id,
                    ),
                )
            )

        if document_id:
            conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(
                        value=document_id,
                    ),
                )
            )

        if document_version_id:
            conditions.append(
                models.FieldCondition(
                    key="document_version_id",
                    match=models.MatchValue(
                        value=document_version_id,
                    ),
                )
            )

        if document_type:
            conditions.append(
                models.FieldCondition(
                    key="document_type",
                    match=models.MatchValue(
                        value=document_type,
                    ),
                )
            )

        if user_id:
            conditions.append(
                models.FieldCondition(
                    key="allowed_user_ids",
                    match=models.MatchValue(
                        value=user_id,
                    ),
                )
            )

        if tags:
            conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(
                        any=list(tags),
                    ),
                )
            )

        if not conditions:
            return None

        return models.Filter(
            must=conditions,
        )

    # ------------------------------------------------------------------
    # Permission-aware search
    # ------------------------------------------------------------------

    def permission_aware_search(
        self,
        *,
        vector: Sequence[float],
        tenant_id: str,
        user_id: str | None = None,
        organization_id: str | None = None,
        document_id: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Search only documents visible to the tenant/user.
        """
        query_filter = self.build_filter(
            tenant_id=tenant_id,
            organization_id=organization_id,
            document_id=document_id,
            user_id=user_id,
        )

        return self.search(
            vector=vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
        )

    # ------------------------------------------------------------------
    # Point retrieval
    # ------------------------------------------------------------------

    def get_points(
        self,
        point_ids: Sequence[str | int],
        *,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve points by ID."""
        name = collection_name or self.collection_name

        try:
            points = self.client.retrieve(
                collection_name=name,
                ids=list(point_ids),
                with_payload=True,
                with_vectors=False,
            )

            return [
                {
                    "id": point.id,
                    "payload": dict(point.payload or {}),
                }
                for point in points
            ]

        except Exception as exc:
            raise QdrantError(
                "Failed to retrieve Qdrant points."
            ) from exc

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_points(
        self,
        point_ids: Sequence[str | int],
        *,
        collection_name: str | None = None,
        wait: bool = True,
    ) -> None:
        """Delete specific vector points."""
        name = collection_name or self.collection_name

        if not point_ids:
            return

        try:
            self.client.delete(
                collection_name=name,
                points_selector=models.PointIdsList(
                    points=list(point_ids),
                ),
                wait=wait,
            )

        except Exception as exc:
            raise QdrantError(
                "Failed to delete Qdrant points."
            ) from exc

    def delete_by_filter(
        self,
        query_filter: models.Filter,
        *,
        collection_name: str | None = None,
        wait: bool = True,
    ) -> None:
        """Delete vectors matching a metadata filter."""
        name = collection_name or self.collection_name

        try:
            self.client.delete(
                collection_name=name,
                points_selector=models.FilterSelector(
                    filter=query_filter,
                ),
                wait=wait,
            )

        except Exception as exc:
            raise QdrantError(
                "Failed to delete filtered Qdrant points."
            ) from exc

    def delete_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> None:
        """Delete all indexed chunks belonging to a document."""
        query_filter = self.build_filter(
            tenant_id=tenant_id,
            document_id=document_id,
        )

        if query_filter:
            self.delete_by_filter(query_filter)

    # ------------------------------------------------------------------
    # Collection information
    # ------------------------------------------------------------------

    def collection_info(
        self,
        collection_name: str | None = None,
    ) -> dict[str, Any]:
        """Return collection statistics."""
        name = collection_name or self.collection_name

        try:
            info = self.client.get_collection(
                collection_name=name,
            )

            return {
                "name": name,
                "status": str(info.status),
                "vectors_count": getattr(
                    info,
                    "vectors_count",
                    None,
                ),
                "points_count": getattr(
                    info,
                    "points_count",
                    None,
                ),
            }

        except Exception as exc:
            raise QdrantCollectionError(
                f"Failed to retrieve collection information: {name}"
            ) from exc

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check Qdrant connectivity."""
        try:
            self.client.get_collections()
            return True

        except Exception as exc:
            logger.exception(
                "Qdrant health check failed"
            )

            raise QdrantError(
                "Qdrant is unavailable."
            ) from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_client(self) -> QdrantClient:
        """Create the Qdrant client."""
        if not self.url:
            raise QdrantConfigurationError(
                "QDRANT_URL is not configured."
            )

        kwargs: dict[str, Any] = {
            "url": self.url,
            "timeout": self.timeout,
        }

        if self.api_key:
            kwargs["api_key"] = self.api_key

        return QdrantClient(**kwargs)

    def _distance(self) -> models.Distance:
        """Convert configured distance to Qdrant enum."""
        mapping = {
            "COSINE": models.Distance.COSINE,
            "DOT": models.Distance.DOT,
            "EUCLID": models.Distance.EUCLID,
            "MANHATTAN": models.Distance.MANHATTAN,
        }

        normalized = self.distance.upper()

        if normalized not in mapping:
            raise QdrantConfigurationError(
                f"Unsupported Qdrant distance: {self.distance}"
            )

        return mapping[normalized]

    @staticmethod
    def _batches(
        items: Sequence[VectorPoint],
        batch_size: int,
    ) -> Iterable[Sequence[VectorPoint]]:
        """Yield fixed-size batches."""
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        for index in range(
            0,
            len(items),
            batch_size,
        ):
            yield items[index:index + batch_size]


def get_qdrant_client() -> QdrantVectorStore:
    """Return the application Qdrant client."""
    return QdrantVectorStore()