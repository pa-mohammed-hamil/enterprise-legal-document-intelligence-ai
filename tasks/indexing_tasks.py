"""
Background tasks for legal-document indexing.

Responsibilities:
    - Index document chunks into the configured vector store
    - Support Qdrant and project-specific indexing services
    - Keep tenant/organization metadata attached to vectors
    - Generate deterministic point IDs for idempotent upserts
    - Support re-indexing and deletion
    - Persist indexing manifests
    - Expose synchronous and Celery execution
    - Provide safe retry behavior

Expected pipeline:

    Document
        |
        v
    DocumentChunk
        |
        v
    Embedding
        |
        v
    Metadata enrichment
        |
        v
    Vector index
        |
        v
    Retrieval / RAG

The implementation supports the project's existing:
    backend.rag.*
    backend.db.*
    backend.tasks.*
layouts while remaining usable during incremental development.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(
        PROJECT_ROOT / ".env",
        override=False,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(
    "legal-indexing-tasks"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_ARTIFACT_ROOT = Path(
    os.getenv(
        "INDEX_ARTIFACT_ROOT",
        str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "indexing"
        ),
    )
)

VECTOR_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    os.getenv(
        "VECTOR_COLLECTION",
        "legal_documents",
    ),
)

VECTOR_DISTANCE = os.getenv(
    "QDRANT_DISTANCE",
    "Cosine",
)

VECTOR_BATCH_SIZE = int(
    os.getenv(
        "VECTOR_BATCH_SIZE",
        "64",
    )
)

VECTOR_UPSERT_BATCH_SIZE = int(
    os.getenv(
        "VECTOR_UPSERT_BATCH_SIZE",
        "128",
    )
)

INDEX_TASK_MAX_RETRIES = int(
    os.getenv(
        "INDEX_TASK_MAX_RETRIES",
        "3",
    )
)

INDEX_TASK_RETRY_BACKOFF = int(
    os.getenv(
        "INDEX_TASK_RETRY_BACKOFF_SECONDS",
        "15",
    )
)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from celery import Task

    CELERY_AVAILABLE = True
except ImportError:
    Task = object  # type: ignore
    CELERY_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    FieldCondition = None  # type: ignore
    Filter = None  # type: ignore
    MatchValue = None  # type: ignore
    PointStruct = None  # type: ignore
    VectorParams = None  # type: ignore
    QDRANT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IndexingTaskResult:
    task: str
    run_id: str
    document_id: str | None
    document_version_id: str | None
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    chunks_scanned: int = 0
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    chunks_deleted: int = 0
    errors: int = 0
    collection: str | None = None
    artifact_path: str | None = None
    message: str | None = None
    error_type: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_iso() -> str:
    return utc_now().isoformat()


def stable_uuid(
    *parts: Any,
) -> str:
    value = ":".join(
        str(part)
        for part in parts
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"legal-index:{value}",
        )
    )


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def ensure_directory(
    path: Path,
) -> Path:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def safe_name(
    value: Any,
) -> str:
    text = str(
        value
    )

    cleaned = "".join(
        char
        if char.isalnum()
        or char in "-_."
        else "_"
        for char in text
    )

    return cleaned[:200] or "unknown"


def write_json(
    path: Path,
    payload: Any,
) -> None:
    ensure_directory(
        path.parent
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def normalize_vector(
    vector: Any,
) -> list[float] | None:
    if vector is None:
        return None

    if hasattr(
        vector,
        "tolist",
    ):
        vector = vector.tolist()

    if not isinstance(
        vector,
        (list, tuple),
    ):
        return None

    try:
        return [
            float(value)
            for value in vector
        ]
    except (
        TypeError,
        ValueError,
    ):
        return None


# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------


def import_module(
    candidates: Iterable[str],
) -> Any | None:
    for module_name in candidates:
        try:
            return importlib.import_module(
                module_name
            )
        except ImportError:
            continue
        except Exception:
            logger.debug(
                "Unable to import %s.",
                module_name,
                exc_info=True,
            )

    return None


def resolve_callable(
    candidates: Iterable[
        tuple[str, str]
    ],
) -> Any | None:
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(
                module_name
            )

            function = getattr(
                module,
                function_name,
                None,
            )

            if callable(function):
                return function

        except ImportError:
            continue
        except Exception:
            logger.debug(
                "Unable to resolve %s.%s.",
                module_name,
                function_name,
                exc_info=True,
            )

    return None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_session_factory() -> Any | None:
    return resolve_callable(
        (
            (
                "backend.db.session",
                "SessionLocal",
            ),
            (
                "backend.db.database",
                "SessionLocal",
            ),
        )
    )


def get_database_session() -> Any:
    factory = get_session_factory()

    if factory is None:
        raise RuntimeError(
            "Unable to locate SessionLocal."
        )

    return factory()


def get_model(
    name: str,
) -> Any | None:
    module = import_module(
        (
            "backend.db.models",
            f"backend.db.models.{name.lower()}",
        )
    )

    if module is None:
        return None

    return getattr(
        module,
        name,
        None,
    )


def get_model_columns(
    model: Any,
) -> set[str]:
    try:
        return {
            column.name
            for column
            in model.__table__.columns
        }
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Chunk normalization
# ---------------------------------------------------------------------------


def first_attribute(
    obj: Any,
    names: Iterable[str],
) -> Any:
    for name in names:
        try:
            value = getattr(
                obj,
                name,
                None,
            )

            if value is not None:
                return value
        except Exception:
            continue

    return None


def chunk_text(
    chunk: Any,
) -> str:
    value = first_attribute(
        chunk,
        (
            "text",
            "content",
            "chunk_text",
            "page_content",
            "body",
        ),
    )

    if value is None:
        return ""

    return str(value).strip()


def chunk_embedding(
    chunk: Any,
) -> list[float] | None:
    value = first_attribute(
        chunk,
        (
            "embedding",
            "vector",
            "embedding_vector",
        ),
    )

    return normalize_vector(
        value
    )


def chunk_id(
    chunk: Any,
    document_id: str,
    document_version_id: str | None,
) -> str:
    existing = first_attribute(
        chunk,
        (
            "id",
            "chunk_id",
            "uuid",
        ),
    )

    if existing:
        return str(existing)

    content = chunk_text(
        chunk
    )

    return stable_uuid(
        document_id,
        document_version_id,
        sha256_text(
            content
        ),
    )


def chunk_metadata(
    chunk: Any,
    *,
    document_id: str,
    document_version_id: str | None,
) -> dict[str, Any]:
    metadata = first_attribute(
        chunk,
        (
            "metadata",
            "meta",
            "attributes",
        ),
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    normalized = dict(
        metadata
    )

    normalized.setdefault(
        "document_id",
        document_id,
    )

    if document_version_id:
        normalized.setdefault(
            "document_version_id",
            document_version_id,
        )

    for field in (
        "page_number",
        "page",
        "section",
        "section_title",
        "heading",
        "chunk_index",
        "start_char",
        "end_char",
        "language",
        "token_count",
    ):
        value = getattr(
            chunk,
            field,
            None,
        )

        if value is not None:
            normalized.setdefault(
                field,
                value,
            )

    tenant_id = first_attribute(
        chunk,
        (
            "tenant_id",
            "tenant",
        ),
    )

    organization_id = first_attribute(
        chunk,
        (
            "organization_id",
            "org_id",
        ),
    )

    if tenant_id is not None:
        normalized.setdefault(
            "tenant_id",
            str(tenant_id),
        )

    if organization_id is not None:
        normalized.setdefault(
            "organization_id",
            str(organization_id),
        )

    return normalized


def normalize_chunk(
    chunk: Any,
    *,
    document_id: str,
    document_version_id: str | None,
) -> dict[str, Any] | None:
    text = chunk_text(
        chunk
    )

    vector = chunk_embedding(
        chunk
    )

    if not text:
        return None

    if not vector:
        return None

    identifier = chunk_id(
        chunk,
        document_id,
        document_version_id,
    )

    metadata = chunk_metadata(
        chunk,
        document_id=document_id,
        document_version_id=document_version_id,
    )

    return {
        "id": identifier,
        "text": text,
        "vector": vector,
        "metadata": metadata,
        "content_hash": sha256_text(
            text
        ),
    }


# ---------------------------------------------------------------------------
# Chunk loading
# ---------------------------------------------------------------------------


def get_document_chunk_model() -> Any | None:
    return get_model(
        "DocumentChunk"
    )


def query_chunks(
    *,
    document_id: str,
    document_version_id: str | None = None,
) -> list[Any]:
    model = get_document_chunk_model()

    if model is None:
        raise RuntimeError(
            "DocumentChunk model is unavailable."
        )

    session = None

    try:
        session = get_database_session()

        query = session.query(
            model
        )

        columns = get_model_columns(
            model
        )

        if (
            "document_id"
            in columns
        ):
            query = query.filter(
                model.document_id
                == document_id
            )

        if (
            document_version_id
            and "document_version_id"
            in columns
        ):
            query = query.filter(
                model.document_version_id
                == document_version_id
            )

        if "chunk_index" in columns:
            query = query.order_by(
                model.chunk_index.asc()
            )

        return query.all()

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


def load_normalized_chunks(
    *,
    document_id: str,
    document_version_id: str | None = None,
) -> list[dict[str, Any]]:
    chunks = query_chunks(
        document_id=document_id,
        document_version_id=document_version_id,
    )

    normalized: list[
        dict[str, Any]
    ] = []

    for chunk in chunks:
        item = normalize_chunk(
            chunk,
            document_id=document_id,
            document_version_id=document_version_id,
        )

        if item is not None:
            normalized.append(
                item
            )

    return normalized


# ---------------------------------------------------------------------------
# Qdrant client
# ---------------------------------------------------------------------------


def get_qdrant_url() -> str:
    return os.getenv(
        "QDRANT_URL",
        os.getenv(
            "VECTOR_DB_URL",
            "http://localhost:6333",
        ),
    )


def get_qdrant_api_key() -> str | None:
    return (
        os.getenv(
            "QDRANT_API_KEY"
        )
        or os.getenv(
            "VECTOR_DB_API_KEY"
        )
        or None
    )


def get_qdrant_client() -> Any:
    """
    Create a Qdrant client.

    A project-specific vector client takes precedence when available.
    """

    project_client = resolve_callable(
        (
            (
                "backend.integrations.vector_db.qdrant",
                "get_client",
            ),
            (
                "backend.integrations.vector_db",
                "get_qdrant_client",
            ),
            (
                "backend.rag.vector_store",
                "get_vector_store",
            ),
        )
    )

    if project_client is not None:
        client = project_client()

        if client is not None:
            return client

    if not QDRANT_AVAILABLE:
        raise RuntimeError(
            "qdrant-client is not installed."
        )

    kwargs: dict[str, Any] = {
        "url": get_qdrant_url(),
    }

    api_key = get_qdrant_api_key()

    if api_key:
        kwargs[
            "api_key"
        ] = api_key

    timeout = float(
        os.getenv(
            "QDRANT_TIMEOUT",
            "30",
        )
    )

    kwargs[
        "timeout"
    ] = timeout

    return QdrantClient(
        **kwargs
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------


def get_vector_dimension(
    vectors: Iterable[
        list[float]
    ],
) -> int:
    for vector in vectors:
        if vector:
            return len(
                vector
            )

    raise ValueError(
        "Unable to determine vector dimension."
    )


def ensure_collection(
    client: Any,
    *,
    collection_name: str = VECTOR_COLLECTION,
    vector_dimension: int,
) -> None:
    """
    Create the Qdrant collection if it does not exist.

    Existing collections are never recreated automatically.
    """

    try:
        collections = (
            client.get_collections()
        )

        names = {
            collection.name
            for collection
            in collections.collections
        }

        if collection_name in names:
            return

    except Exception:
        # Some Qdrant-compatible clients do not expose
        # get_collections in exactly the same form.
        try:
            client.get_collection(
                collection_name
            )
            return
        except Exception:
            pass

    if not QDRANT_AVAILABLE:
        raise RuntimeError(
            "Qdrant models unavailable."
        )

    distance_map = {
        "COSINE": Distance.COSINE,
        "DOT": Distance.DOT,
        "EUCLID": Distance.EUCLID,
        "EUCLIDEAN": Distance.EUCLID,
    }

    distance = distance_map.get(
        VECTOR_DISTANCE.upper(),
        Distance.COSINE,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dimension,
            distance=distance,
        ),
    )


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------


def build_payload(
    chunk: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(
        chunk.get(
            "metadata",
            {},
        )
    )

    payload = {
        "chunk_id": chunk["id"],
        "document_id": metadata.get(
            "document_id"
        ),
        "document_version_id": metadata.get(
            "document_version_id"
        ),
        "text": chunk["text"],
        "content_hash": chunk[
            "content_hash"
        ],
        "metadata": metadata,
        "indexed_at": utc_iso(),
    }

    # Promote frequently filtered fields.
    for field in (
        "tenant_id",
        "organization_id",
        "document_type",
        "contract_type",
        "jurisdiction",
        "language",
        "section",
        "page_number",
    ):
        if field in metadata:
            payload[field] = metadata[
                field
            ]

    return payload


def build_points(
    chunks: list[
        dict[str, Any]
    ],
) -> list[Any]:
    if not QDRANT_AVAILABLE:
        raise RuntimeError(
            "qdrant-client is not installed."
        )

    points = []

    for chunk in chunks:
        point_id = stable_uuid(
            "vector-point",
            chunk["id"],
        )

        points.append(
            PointStruct(
                id=point_id,
                vector=chunk[
                    "vector"
                ],
                payload=build_payload(
                    chunk
                ),
            )
        )

    return points


# ---------------------------------------------------------------------------
# Project-specific indexing integration
# ---------------------------------------------------------------------------


def project_index_chunks(
    chunks: list[
        dict[str, Any]
    ],
    *,
    collection_name: str,
) -> Any | None:
    """
    Prefer the project's own indexing service if available.
    """

    function = resolve_callable(
        (
            (
                "backend.rag.indexing",
                "index_chunks",
            ),
            (
                "backend.rag.pipeline",
                "index_chunks",
            ),
            (
                "backend.integrations.vector_db.qdrant",
                "upsert_chunks",
            ),
        )
    )

    if function is None:
        return None

    attempts = (
        {
            "chunks": chunks,
            "collection_name": collection_name,
        },
        {
            "records": chunks,
            "collection_name": collection_name,
        },
        {
            "chunks": chunks,
        },
        {
            "records": chunks,
        },
    )

    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            return function(
                **kwargs
            )
        except TypeError as exc:
            last_error = exc

    if last_error:
        raise last_error

    return None


# ---------------------------------------------------------------------------
# Qdrant indexing
# ---------------------------------------------------------------------------


def upsert_qdrant_chunks(
    chunks: list[
        dict[str, Any]
    ],
    *,
    collection_name: str = VECTOR_COLLECTION,
) -> int:
    if not chunks:
        return 0

    client = get_qdrant_client()

    dimension = get_vector_dimension(
        chunk["vector"]
        for chunk in chunks
    )

    ensure_collection(
        client,
        collection_name=collection_name,
        vector_dimension=dimension,
    )

    indexed = 0

    for start in range(
        0,
        len(chunks),
        VECTOR_UPSERT_BATCH_SIZE,
    ):
        batch = chunks[
            start : start
            + VECTOR_UPSERT_BATCH_SIZE
        ]

        points = build_points(
            batch
        )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

        indexed += len(
            batch
        )

    return indexed


# ---------------------------------------------------------------------------
# Delete vectors
# ---------------------------------------------------------------------------


def delete_document_vectors(
    *,
    document_id: str,
    document_version_id: str | None = None,
    collection_name: str = VECTOR_COLLECTION,
) -> int:
    client = get_qdrant_client()

    if not QDRANT_AVAILABLE:
        raise RuntimeError(
            "qdrant-client is not installed."
        )

    must_conditions = [
        FieldCondition(
            key="document_id",
            match=MatchValue(
                value=document_id
            ),
        )
    ]

    if document_version_id:
        must_conditions.append(
            FieldCondition(
                key="document_version_id",
                match=MatchValue(
                    value=document_version_id
                ),
            )
        )

    filter_condition = Filter(
        must=must_conditions
    )

    try:
        result = client.count(
            collection_name=collection_name,
            count_filter=filter_condition,
            exact=True,
        )

        count = int(
            getattr(
                result,
                "count",
                0,
            )
        )

    except Exception:
        count = 0

    client.delete(
        collection_name=collection_name,
        points_selector=filter_condition,
        wait=True,
    )

    return count


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------


def manifest_path(
    document_id: str,
    document_version_id: str | None,
) -> Path:
    directory = (
        INDEX_ARTIFACT_ROOT
        / safe_name(document_id)
    )

    if document_version_id:
        directory = (
            directory
            / safe_name(
                document_version_id
            )
        )

    ensure_directory(
        directory
    )

    return (
        directory
        / "index_manifest.json"
    )


def write_index_manifest(
    *,
    document_id: str,
    document_version_id: str | None,
    chunks: list[
        dict[str, Any]
    ],
    indexed_count: int,
    collection_name: str,
    duration_ms: float,
) -> Path:
    path = manifest_path(
        document_id,
        document_version_id,
    )

    payload = {
        "document_id": document_id,
        "document_version_id": document_version_id,
        "collection": collection_name,
        "chunk_count": len(
            chunks
        ),
        "indexed_count": indexed_count,
        "indexed_at": utc_iso(),
        "duration_ms": duration_ms,
        "chunks": [
            {
                "chunk_id": chunk["id"],
                "content_hash": chunk[
                    "content_hash"
                ],
            }
            for chunk in chunks
        ],
    }

    write_json(
        path,
        payload,
    )

    return path


# ---------------------------------------------------------------------------
# Database indexing metadata
# ---------------------------------------------------------------------------


def mark_chunks_indexed(
    *,
    document_id: str,
    document_version_id: str | None,
    indexed_chunk_ids: set[str],
) -> int:
    """
    Best-effort update of chunk indexing state.

    Supports common fields:
        indexed
        is_indexed
        indexed_at
        vector_id
        vector_store_id
    """

    model = get_document_chunk_model()

    if model is None:
        return 0

    session = None
    updated = 0

    try:
        session = get_database_session()

        query = session.query(
            model
        )

        columns = get_model_columns(
            model
        )

        if (
            "document_id"
            in columns
        ):
            query = query.filter(
                model.document_id
                == document_id
            )

        if (
            document_version_id
            and "document_version_id"
            in columns
        ):
            query = query.filter(
                model.document_version_id
                == document_version_id
            )

        chunks = query.all()

        for chunk in chunks:
            identifier = chunk_id(
                chunk,
                document_id,
                document_version_id,
            )

            if identifier not in indexed_chunk_ids:
                continue

            values = {
                "indexed": True,
                "is_indexed": True,
                "indexed_at": utc_now(),
                "index_status": "indexed",
            }

            set_values = False

            for field, value in values.items():
                if field in columns:
                    setattr(
                        chunk,
                        field,
                        value,
                    )
                    set_values = True

            if set_values:
                updated += 1

        session.commit()

    except Exception:
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass

        logger.warning(
            "Unable to update chunk indexing state.",
            exc_info=True,
        )

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    return updated


# ---------------------------------------------------------------------------
# Core indexing workflow
# ---------------------------------------------------------------------------


def index_document(
    *,
    document_id: str,
    document_version_id: str | None = None,
    collection_name: str = VECTOR_COLLECTION,
) -> dict[str, Any]:
    """
    Index one document version.

    Existing vector IDs are deterministic, so repeated execution
    performs idempotent upserts rather than creating duplicates.
    """

    started = time.perf_counter()

    if not document_id:
        raise ValueError(
            "document_id is required."
        )

    run_id = stable_uuid(
        "index-document",
        document_id,
        document_version_id,
        collection_name,
    )

    chunks = load_normalized_chunks(
        document_id=document_id,
        document_version_id=document_version_id,
    )

    if not chunks:
        result = IndexingTaskResult(
            task="index_document",
            run_id=run_id,
            document_id=document_id,
            document_version_id=document_version_id,
            status="skipped",
            started_at=utc_iso(),
            completed_at=utc_iso(),
            duration_ms=round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            ),
            chunks_scanned=0,
            chunks_indexed=0,
            chunks_skipped=0,
            collection=collection_name,
            message=(
                "No embeddable chunks "
                "were found."
            ),
        )

        return result.to_dict()

    indexed_count = 0
    backend_name = "qdrant"

    try:
        project_result = project_index_chunks(
            chunks,
            collection_name=collection_name,
        )

        if project_result is not None:
            backend_name = (
                "project_indexer"
            )

            if isinstance(
                project_result,
                int,
            ):
                indexed_count = (
                    project_result
                )
            elif isinstance(
                project_result,
                dict,
            ):
                indexed_count = int(
                    project_result.get(
                        "indexed_count",
                        project_result.get(
                            "count",
                            len(chunks),
                        ),
                    )
                )
            else:
                indexed_count = len(
                    chunks
                )

        else:
            indexed_count = (
                upsert_qdrant_chunks(
                    chunks,
                    collection_name=collection_name,
                )
            )

        indexed_ids = {
            chunk["id"]
            for chunk in chunks
        }

        mark_chunks_indexed(
            document_id=document_id,
            document_version_id=document_version_id,
            indexed_chunk_ids=indexed_ids,
        )

        duration_ms = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        )

        manifest = (
            write_index_manifest(
                document_id=document_id,
                document_version_id=(
                    document_version_id
                ),
                chunks=chunks,
                indexed_count=indexed_count,
                collection_name=collection_name,
                duration_ms=duration_ms,
            )
        )

        result = IndexingTaskResult(
            task="index_document",
            run_id=run_id,
            document_id=document_id,
            document_version_id=document_version_id,
            status="completed",
            started_at=utc_iso(),
            completed_at=utc_iso(),
            duration_ms=duration_ms,
            chunks_scanned=len(
                chunks
            ),
            chunks_indexed=indexed_count,
            chunks_skipped=max(
                0,
                len(chunks)
                - indexed_count,
            ),
            collection=collection_name,
            artifact_path=str(
                manifest
            ),
            message=(
                "Document indexed successfully."
            ),
            details={
                "backend": backend_name,
                "vector_dimension": len(
                    chunks[0]["vector"]
                ),
            },
        )

        return result.to_dict()

    except Exception as exc:
        logger.exception(
            "Document indexing failed for %s.",
            document_id,
        )

        result = IndexingTaskResult(
            task="index_document",
            run_id=run_id,
            document_id=document_id,
            document_version_id=document_version_id,
            status="failed",
            started_at=utc_iso(),
            completed_at=utc_iso(),
            duration_ms=round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            ),
            chunks_scanned=len(
                chunks
            ),
            collection=collection_name,
            message=str(exc)[:4000],
            error_type=type(
                exc
            ).__name__,
        )

        return result.to_dict()


# ---------------------------------------------------------------------------
# Batch indexing
# ---------------------------------------------------------------------------


def index_documents(
    document_ids: Iterable[str],
    *,
    collection_name: str = VECTOR_COLLECTION,
) -> dict[str, Any]:
    started = time.perf_counter()

    results: list[
        dict[str, Any]
    ] = []

    for document_id in document_ids:
        result = index_document(
            document_id=document_id,
            collection_name=collection_name,
        )

        results.append(
            result
        )

    successful = sum(
        1
        for result in results
        if result.get("status")
        == "completed"
    )

    failed = sum(
        1
        for result in results
        if result.get("status")
        == "failed"
    )

    skipped = sum(
        1
        for result in results
        if result.get("status")
        == "skipped"
    )

    return {
        "task": "index_documents",
        "status": (
            "failed"
            if failed
            else "completed"
        ),
        "document_count": len(
            results
        ),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "duration_ms": round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        ),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Re-indexing
# ---------------------------------------------------------------------------


def reindex_document(
    *,
    document_id: str,
    document_version_id: str | None = None,
    collection_name: str = VECTOR_COLLECTION,
) -> dict[str, Any]:
    """
    Remove existing vectors for the target document and rebuild the index.
    """

    started = time.perf_counter()

    try:
        deleted = delete_document_vectors(
            document_id=document_id,
            document_version_id=document_version_id,
            collection_name=collection_name,
        )

        result = index_document(
            document_id=document_id,
            document_version_id=document_version_id,
            collection_name=collection_name,
        )

        result[
            "task"
        ] = "reindex_document"

        result.setdefault(
            "details",
            {},
        )

        result[
            "details"
        ][
            "vectors_deleted_before_reindex"
        ] = deleted

        result[
            "details"
        ][
            "reindex_duration_ms"
        ] = round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        )

        return result

    except Exception as exc:
        logger.exception(
            "Re-indexing failed for %s.",
            document_id,
        )

        return {
            "task": "reindex_document",
            "document_id": document_id,
            "document_version_id": (
                document_version_id
            ),
            "status": "failed",
            "message": str(exc),
            "error_type": type(
                exc
            ).__name__,
        }


# ---------------------------------------------------------------------------
# Delete indexing
# ---------------------------------------------------------------------------


def remove_document_from_index(
    *,
    document_id: str,
    document_version_id: str | None = None,
    collection_name: str = VECTOR_COLLECTION,
) -> dict[str, Any]:
    started = time.perf_counter()

    try:
        deleted = delete_document_vectors(
            document_id=document_id,
            document_version_id=document_version_id,
            collection_name=collection_name,
        )

        return {
            "task": "remove_document_from_index",
            "document_id": document_id,
            "document_version_id": (
                document_version_id
            ),
            "status": "completed",
            "deleted": deleted,
            "collection": collection_name,
            "duration_ms": round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                2,
            ),
        }

    except Exception as exc:
        logger.exception(
            "Unable to remove document %s from index.",
            document_id,
        )

        return {
            "task": "remove_document_from_index",
            "document_id": document_id,
            "document_version_id": (
                document_version_id
            ),
            "status": "failed",
            "message": str(exc),
            "error_type": type(
                exc
            ).__name__,
        }


# ---------------------------------------------------------------------------
# Collection statistics
# ---------------------------------------------------------------------------


def get_index_statistics(
    collection_name: str = VECTOR_COLLECTION,
) -> dict[str, Any]:
    client = get_qdrant_client()

    collection = client.get_collection(
        collection_name
    )

    return {
        "collection": collection_name,
        "status": str(
            getattr(
                collection,
                "status",
                "unknown",
            )
        ),
        "points_count": getattr(
            collection,
            "points_count",
            None,
        ),
        "vectors_count": getattr(
            collection,
            "vectors_count",
            None,
        ),
        "indexed_vectors_count": getattr(
            collection,
            "indexed_vectors_count",
            None,
        ),
    }


# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------


class IndexingTaskBase(
    Task
):
    max_retries = (
        INDEX_TASK_MAX_RETRIES
    )

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.error(
            "Indexing task failed: "
            "task_id=%s error=%s",
            task_id,
            exc,
        )


def get_celery_app() -> Any | None:
    for module_name in (
        "backend.workers.celery_app",
        "backend.tasks.celery_app",
        "backend.celery_app",
    ):
        try:
            module = importlib.import_module(
                module_name
            )

            for attribute in (
                "celery_app",
                "app",
                "celery",
            ):
                app = getattr(
                    module,
                    attribute,
                    None,
                )

                if app is not None:
                    return app

        except ImportError:
            continue
        except Exception:
            logger.debug(
                "Unable to load Celery app.",
                exc_info=True,
            )

    return None


celery_app = get_celery_app()


# ---------------------------------------------------------------------------
# Celery task registrations
# ---------------------------------------------------------------------------


if celery_app is not None:

    @celery_app.task(
        bind=True,
        base=IndexingTaskBase,
        name="legal.indexing.document",
        max_retries=INDEX_TASK_MAX_RETRIES,
    )
    def index_document_task(
        self: Any,
        document_id: str,
        document_version_id: str | None = None,
        collection_name: str = VECTOR_COLLECTION,
    ) -> dict[str, Any]:
        try:
            result = index_document(
                document_id=document_id,
                document_version_id=(
                    document_version_id
                ),
                collection_name=(
                    collection_name
                ),
            )

            if result.get(
                "status"
            ) == "failed":
                raise RuntimeError(
                    result.get(
                        "message",
                        "Document indexing failed.",
                    )
                )

            return result

        except Exception as exc:
            countdown = (
                INDEX_TASK_RETRY_BACKOFF
                * (
                    2
                    ** self.request.retries
                )
            )

            raise self.retry(
                exc=exc,
                countdown=countdown,
            )

    @celery_app.task(
        bind=True,
        base=IndexingTaskBase,
        name="legal.indexing.reindex",
        max_retries=INDEX_TASK_MAX_RETRIES,
    )
    def reindex_document_task(
        self: Any,
        document_id: str,
        document_version_id: str | None = None,
        collection_name: str = VECTOR_COLLECTION,
    ) -> dict[str, Any]:
        try:
            result = reindex_document(
                document_id=document_id,
                document_version_id=(
                    document_version_id
                ),
                collection_name=(
                    collection_name
                ),
            )

            if result.get(
                "status"
            ) == "failed":
                raise RuntimeError(
                    result.get(
                        "message",
                        "Document re-indexing failed.",
                    )
                )

            return result

        except Exception as exc:
            countdown = (
                INDEX_TASK_RETRY_BACKOFF
                * (
                    2
                    ** self.request.retries
                )
            )

            raise self.retry(
                exc=exc,
                countdown=countdown,
            )

    @celery_app.task(
        bind=True,
        base=IndexingTaskBase,
        name="legal.indexing.delete",
        max_retries=INDEX_TASK_MAX_RETRIES,
    )
    def remove_document_from_index_task(
        self: Any,
        document_id: str,
        document_version_id: str | None = None,
        collection_name: str = VECTOR_COLLECTION,
    ) -> dict[str, Any]:
        try:
            result = (
                remove_document_from_index(
                    document_id=document_id,
                    document_version_id=(
                        document_version_id
                    ),
                    collection_name=(
                        collection_name
                    ),
                )
            )

            if result.get(
                "status"
            ) == "failed":
                raise RuntimeError(
                    result.get(
                        "message",
                        "Index deletion failed.",
                    )
                )

            return result

        except Exception as exc:
            countdown = (
                INDEX_TASK_RETRY_BACKOFF
                * (
                    2
                    ** self.request.retries
                )
            )

            raise self.retry(
                exc=exc,
                countdown=countdown,
            )

else:

    def index_document_task(
        document_id: str,
        document_version_id: str | None = None,
        collection_name: str = VECTOR_COLLECTION,
    ) -> dict[str, Any]:
        return index_document(
            document_id=document_id,
            document_version_id=(
                document_version_id
            ),
            collection_name=collection_name,
        )

    def reindex_document_task(
        document_id: str,
        document_version_id: str | None = None,
        collection_name: str = VECTOR_COLLECTION,
    ) -> dict[str, Any]:
        return reindex_document(
            document_id=document_id,
            document_version_id=(
                document_version_id
            ),
            collection_name=collection_name,
        )

    def remove_document_from_index_task(
        document_id: str,
        document_version_id: str | None = None,
        collection_name: str = VECTOR_COLLECTION,
    ) -> dict[str, Any]:
        return remove_document_from_index(
            document_id=document_id,
            document_version_id=(
                document_version_id
            ),
            collection_name=collection_name,
        )


# ---------------------------------------------------------------------------
# Celery Beat helper
# ---------------------------------------------------------------------------


def get_indexing_beat_schedule() -> dict[str, Any]:
    """
    Optional periodic schedule for rebuilding stale indexes.

    Keep the schedule disabled in production unless the workload
    justifies automatic re-indexing.
    """

    return {
        "legal-indexing-health-check": {
            "task": (
                "legal.indexing."
                "health_check"
            ),
            "schedule": 3600,
            "options": {
                "queue": "legal-indexing",
            },
        }
    }


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "IndexingTaskResult",
    "index_document",
    "index_documents",
    "reindex_document",
    "remove_document_from_index",
    "delete_document_vectors",
    "get_index_statistics",
    "index_document_task",
    "reindex_document_task",
    "remove_document_from_index_task",
    "get_indexing_beat_schedule",
]