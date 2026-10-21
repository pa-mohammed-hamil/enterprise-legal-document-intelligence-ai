"""
Background tasks for legal-document ingestion and processing.

Pipeline:
    upload
      -> parse
      -> normalize
      -> OCR
      -> metadata extraction
      -> chunk
      -> embedding
      -> indexing
      -> legal entity extraction
      -> risk/obligation analysis
      -> evaluation
      -> notification

The implementation is intentionally defensive:
    - Celery is optional during local development.
    - Individual pipeline modules are discovered dynamically.
    - Failures are persisted through structured task results when
      application-specific persistence is available.
    - Tasks are idempotent by document/version identifiers.
    - No secrets are logged.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

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
    "legal-document-tasks"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_TASK_RETRIES = int(
    os.getenv(
        "DOCUMENT_TASK_MAX_RETRIES",
        "3",
    )
)

RETRY_BACKOFF_SECONDS = int(
    os.getenv(
        "DOCUMENT_TASK_RETRY_BACKOFF_SECONDS",
        "10",
    )
)

TASK_TIME_LIMIT = int(
    os.getenv(
        "DOCUMENT_TASK_TIME_LIMIT",
        "1800",
    )
)

SOFT_TASK_TIME_LIMIT = int(
    os.getenv(
        "DOCUMENT_TASK_SOFT_TIME_LIMIT",
        "1500",
    )
)

ARTIFACT_ROOT = Path(
    os.getenv(
        "DOCUMENT_ARTIFACT_ROOT",
        str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "documents"
        ),
    )
)

# ---------------------------------------------------------------------------
# Optional Celery import
# ---------------------------------------------------------------------------

try:
    from celery import Task

    CELERY_AVAILABLE = True
except ImportError:
    Task = object  # type: ignore
    CELERY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DocumentTaskResult:
    """
    Serializable result returned by document-processing tasks.
    """

    task: str
    document_id: str
    document_version_id: str | None
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    artifact_path: str | None = None
    records_processed: int = 0
    message: str | None = None
    error_type: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_iso() -> str:
    return utc_now().isoformat()


def stable_id(
    *parts: Any,
) -> str:
    value = ":".join(
        str(part)
        for part in parts
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"legal-ai:{value}",
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


def safe_document_id(
    document_id: str,
) -> str:
    """
    Prevent path traversal when document IDs are used
    for local development artifacts.
    """

    cleaned = "".join(
        character
        if character.isalnum()
        or character in "-_"
        else "_"
        for character in str(document_id)
    )

    return cleaned[:200] or "unknown-document"


def artifact_directory(
    document_id: str,
    document_version_id: str | None = None,
) -> Path:
    directory = (
        ARTIFACT_ROOT
        / safe_document_id(document_id)
    )

    if document_version_id:
        directory = (
            directory
            / safe_document_id(
                document_version_id
            )
        )

    return ensure_directory(
        directory
    )


def write_json_artifact(
    path: Path,
    payload: Any,
) -> None:
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


def read_json_artifact(
    path: Path,
) -> Any | None:
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        logger.warning(
            "Unable to read artifact: %s",
            path,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------


def import_first(
    candidates: tuple[str, ...],
) -> Any | None:
    """
    Import the first available module from a candidate list.
    """

    for module_name in candidates:
        try:
            return importlib.import_module(
                module_name
            )
        except ImportError:
            continue
        except Exception:
            logger.warning(
                "Error importing %s.",
                module_name,
                exc_info=True,
            )

    return None


def resolve_callable(
    candidates: tuple[tuple[str, str], ...],
) -> Callable[..., Any] | None:
    """
    Find a callable from candidate module/attribute pairs.
    """

    for module_name, attribute_name in candidates:
        try:
            module = importlib.import_module(
                module_name
            )

            function = getattr(
                module,
                attribute_name,
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
                attribute_name,
                exc_info=True,
            )

    return None


# ---------------------------------------------------------------------------
# Application database helpers
# ---------------------------------------------------------------------------


def get_database_session() -> Any:
    candidates = (
        (
            "backend.db.session",
            "SessionLocal",
        ),
        (
            "backend.db.database",
            "SessionLocal",
        ),
    )

    factory = resolve_callable(
        candidates
    )

    if factory is None:
        raise RuntimeError(
            "Unable to locate SessionLocal."
        )

    return factory()


def get_document_model() -> Any | None:
    module = import_first(
        (
            "backend.db.models",
            "backend.db.models.document",
        )
    )

    if module is None:
        return None

    return getattr(
        module,
        "Document",
        None,
    )


def get_document_version_model() -> Any | None:
    module = import_first(
        (
            "backend.db.models",
            "backend.db.models.document_version",
        )
    )

    if module is None:
        return None

    return getattr(
        module,
        "DocumentVersion",
        None,
    )


def get_model_instance(
    session: Any,
    model: Any,
    object_id: str,
) -> Any | None:
    if model is None:
        return None

    try:
        return session.get(
            model,
            object_id,
        )
    except Exception:
        try:
            return (
                session.query(model)
                .filter(
                    model.id == object_id
                )
                .first()
            )
        except Exception:
            return None


def update_document_status(
    document_id: str,
    status: str,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Best-effort document status persistence.

    The function does not allow a database failure to hide
    the original processing exception.
    """

    session = None

    try:
        session = get_database_session()

        model = get_document_model()

        document = get_model_instance(
            session,
            model,
            document_id,
        )

        if document is None:
            return

        columns = set()

        try:
            columns = {
                column.name
                for column
                in document.__table__.columns
            }
        except Exception:
            pass

        values: dict[str, Any] = {
            "status": status,
            "processing_status": status,
            "updated_at": utc_now(),
        }

        if error:
            values.update(
                {
                    "error": error[:4000],
                    "processing_error": error[:4000],
                }
            )

        if metadata:
            if "metadata" in columns:
                current = getattr(
                    document,
                    "metadata",
                    None,
                )

                if not isinstance(
                    current,
                    dict,
                ):
                    current = {}

                current.update(
                    metadata
                )

                values["metadata"] = current

        for key, value in values.items():
            if key in columns:
                setattr(
                    document,
                    key,
                    value,
                )

        session.commit()

    except Exception:
        logger.warning(
            "Unable to update document status for %s.",
            document_id,
            exc_info=True,
        )

        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Task result helpers
# ---------------------------------------------------------------------------


def make_result(
    task_name: str,
    document_id: str,
    document_version_id: str | None,
    started: float,
    status: str,
    artifact_path: str | None = None,
    records_processed: int = 0,
    message: str | None = None,
    error_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    completed = time.perf_counter()

    result = DocumentTaskResult(
        task=task_name,
        document_id=document_id,
        document_version_id=document_version_id,
        status=status,
        started_at=utc_iso(),
        completed_at=utc_iso(),
        duration_ms=round(
            (completed - started) * 1000,
            2,
        ),
        artifact_path=artifact_path,
        records_processed=records_processed,
        message=message,
        error_type=error_type,
        metadata=metadata or {},
    )

    return result.to_dict()


def task_failure(
    task_name: str,
    document_id: str,
    document_version_id: str | None,
    started: float,
    exc: Exception,
) -> dict[str, Any]:
    error_message = str(exc)

    return make_result(
        task_name=task_name,
        document_id=document_id,
        document_version_id=document_version_id,
        started=started,
        status="failed",
        message=error_message[:4000],
        error_type=type(exc).__name__,
    )


# ---------------------------------------------------------------------------
# Pipeline invocation
# ---------------------------------------------------------------------------


def call_pipeline_function(
    function: Callable[..., Any] | None,
    *,
    document_id: str,
    document_version_id: str | None = None,
    payload: Any = None,
    **kwargs: Any,
) -> Any:
    if function is None:
        raise RuntimeError(
            "Required document-processing component "
            "is not available."
        )

    attempts = (
        {
            "document_id": document_id,
            "document_version_id": document_version_id,
            "payload": payload,
            **kwargs,
        },
        {
            "document_id": document_id,
            "document_version_id": document_version_id,
            **kwargs,
        },
        {
            "document_id": document_id,
            **kwargs,
        },
        {
            "payload": payload,
        },
        {
            "payload": payload,
            "document_id": document_id,
        },
    )

    last_error: Exception | None = None

    for parameters in attempts:
        try:
            cleaned = {
                key: value
                for key, value in parameters.items()
                if value is not None
            }

            return function(
                **cleaned
            )

        except TypeError as exc:
            last_error = exc
            continue

    if last_error:
        raise last_error

    raise RuntimeError(
        "Unable to invoke pipeline component."
    )


def serialize_pipeline_result(
    result: Any,
) -> Any:
    if result is None:
        return None

    if isinstance(
        result,
        (str, int, float, bool),
    ):
        return result

    if isinstance(
        result,
        dict,
    ):
        return result

    if isinstance(
        result,
        (list, tuple),
    ):
        return list(result)

    if hasattr(
        result,
        "model_dump",
    ):
        return result.model_dump()

    if hasattr(
        result,
        "dict",
    ):
        try:
            return result.dict()
        except Exception:
            pass

    if hasattr(
        result,
        "__dict__",
    ):
        return {
            key: value
            for key, value
            in vars(result).items()
            if not key.startswith("_")
        }

    return str(result)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def run_document_ingestion(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.ingestion.pipeline",
                "ingest_document",
            ),
            (
                "backend.ingestion.pipeline",
                "run_pipeline",
            ),
            (
                "backend.ingestion",
                "ingest_document",
            ),
        )
    )

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_document_parsing(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.ingestion.document_parser",
                "parse_document",
            ),
            (
                "backend.ingestion.parser",
                "parse_document",
            ),
            (
                "backend.ingestion.pipeline",
                "parse_document",
            ),
        )
    )

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_document_ocr(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.ingestion.ocr",
                "process_ocr",
            ),
            (
                "backend.ingestion.ocr",
                "run_ocr",
            ),
            (
                "backend.ingestion.pipeline",
                "run_ocr",
            ),
        )
    )

    if function is None:
        logger.info(
            "OCR component not configured; "
            "skipping OCR for document %s.",
            document_id,
        )
        return {
            "status": "skipped",
            "reason": "ocr_component_unavailable",
        }

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_metadata_extraction(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.ingestion.metadata",
                "extract_metadata",
            ),
            (
                "backend.ingestion.metadata_extractor",
                "extract_metadata",
            ),
            (
                "backend.ingestion.pipeline",
                "extract_metadata",
            ),
        )
    )

    if function is None:
        return {
            "status": "skipped",
            "reason": "metadata_extractor_unavailable",
        }

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_chunking(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.rag.chunking",
                "chunk_document",
            ),
            (
                "backend.rag.chunking",
                "create_chunks",
            ),
            (
                "backend.ingestion.pipeline",
                "chunk_document",
            ),
        )
    )

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_embeddings(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.rag.embeddings",
                "embed_document",
            ),
            (
                "backend.rag.embeddings",
                "create_embeddings",
            ),
            (
                "backend.ingestion.pipeline",
                "create_embeddings",
            ),
        )
    )

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_indexing(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.rag.retrieval",
                "index_document",
            ),
            (
                "backend.rag.indexing",
                "index_document",
            ),
            (
                "backend.rag.pipeline",
                "index_document",
            ),
        )
    )

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_legal_entity_extraction(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.legal.entity_extractor",
                "extract_entities",
            ),
            (
                "backend.knowledge_graph.entity_extractor",
                "extract_entities",
            ),
            (
                "backend.agents.document_agent",
                "extract_entities",
            ),
        )
    )

    if function is None:
        return {
            "status": "skipped",
            "reason": "entity_extractor_unavailable",
        }

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def run_legal_analysis(
    document_id: str,
    document_version_id: str | None = None,
) -> Any:
    function = resolve_callable(
        (
            (
                "backend.legal.legal_reasoning",
                "analyze_document",
            ),
            (
                "backend.agents.supervisor",
                "analyze_document",
            ),
            (
                "backend.agents.document_agent",
                "analyze_document",
            ),
        )
    )

    if function is None:
        return {
            "status": "skipped",
            "reason": "legal_analysis_unavailable",
        }

    return call_pipeline_function(
        function,
        document_id=document_id,
        document_version_id=document_version_id,
    )


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def persist_stage_result(
    document_id: str,
    document_version_id: str | None,
    stage: str,
    result: Any,
) -> Path:
    directory = artifact_directory(
        document_id,
        document_version_id,
    )

    path = (
        directory
        / f"{stage}.json"
    )

    payload = {
        "document_id": document_id,
        "document_version_id": document_version_id,
        "stage": stage,
        "created_at": utc_iso(),
        "result": serialize_pipeline_result(
            result
        ),
    }

    write_json_artifact(
        path,
        payload,
    )

    return path


# ---------------------------------------------------------------------------
# Individual tasks
# ---------------------------------------------------------------------------


def _execute_stage(
    *,
    task_name: str,
    stage: str,
    document_id: str,
    document_version_id: str | None,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    started = time.perf_counter()

    if not document_id:
        raise ValueError(
            "document_id is required."
        )

    update_document_status(
        document_id,
        "processing",
    )

    try:
        result = runner(
            document_id,
            document_version_id,
        )

        artifact = persist_stage_result(
            document_id,
            document_version_id,
            stage,
            result,
        )

        records = 1

        if isinstance(
            result,
            dict,
        ):
            possible_records = (
                result.get(
                    "records_processed"
                )
                or result.get(
                    "count"
                )
                or result.get(
                    "total"
                )
            )

            if isinstance(
                possible_records,
                int,
            ):
                records = possible_records

        output = make_result(
            task_name=task_name,
            document_id=document_id,
            document_version_id=document_version_id,
            started=started,
            status="completed",
            artifact_path=str(
                artifact
            ),
            records_processed=records,
            message=f"{stage} completed.",
        )

        return output

    except Exception as exc:
        update_document_status(
            document_id,
            "failed",
            error=str(exc),
            metadata={
                "failed_stage": stage,
            },
        )

        logger.exception(
            "%s failed for document %s.",
            task_name,
            document_id,
        )

        return task_failure(
            task_name,
            document_id,
            document_version_id,
            started,
            exc,
        )


def ingest_document(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="ingest_document",
        stage="ingestion",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_document_ingestion,
    )


def parse_document(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="parse_document",
        stage="parsing",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_document_parsing,
    )


def process_document_ocr(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="process_document_ocr",
        stage="ocr",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_document_ocr,
    )


def extract_document_metadata(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="extract_document_metadata",
        stage="metadata",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_metadata_extraction,
    )


def chunk_document(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="chunk_document",
        stage="chunking",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_chunking,
    )


def create_document_embeddings(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="create_document_embeddings",
        stage="embeddings",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_embeddings,
    )


def index_document(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="index_document",
        stage="indexing",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_indexing,
    )


def extract_legal_entities(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="extract_legal_entities",
        stage="legal_entities",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_legal_entity_extraction,
    )


def analyze_document(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    return _execute_stage(
        task_name="analyze_document",
        stage="legal_analysis",
        document_id=document_id,
        document_version_id=document_version_id,
        runner=run_legal_analysis,
    )


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def process_document_pipeline(
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute the complete document intelligence pipeline.

    The stages are deliberately sequential because downstream
    stages depend on artifacts generated by previous stages.
    """

    started = time.perf_counter()

    if not document_id:
        raise ValueError(
            "document_id is required."
        )

    update_document_status(
        document_id,
        "processing",
        metadata={
            "pipeline_started_at": utc_iso(),
        },
    )

    stages = (
        (
            "ingestion",
            run_document_ingestion,
        ),
        (
            "parsing",
            run_document_parsing,
        ),
        (
            "ocr",
            run_document_ocr,
        ),
        (
            "metadata",
            run_metadata_extraction,
        ),
        (
            "chunking",
            run_chunking,
        ),
        (
            "embeddings",
            run_embeddings,
        ),
        (
            "indexing",
            run_indexing,
        ),
        (
            "legal_entities",
            run_legal_entity_extraction,
        ),
        (
            "legal_analysis",
            run_legal_analysis,
        ),
    )

    results: dict[str, Any] = {}
    failures: list[str] = []

    for stage_name, runner in stages:
        stage_started = time.perf_counter()

        try:
            result = runner(
                document_id,
                document_version_id,
            )

            artifact = persist_stage_result(
                document_id,
                document_version_id,
                stage_name,
                result,
            )

            results[stage_name] = {
                "status": "completed",
                "artifact_path": str(
                    artifact
                ),
                "duration_ms": round(
                    (
                        time.perf_counter()
                        - stage_started
                    )
                    * 1000,
                    2,
                ),
                "result": serialize_pipeline_result(
                    result
                ),
            }

        except Exception as exc:
            failures.append(
                stage_name
            )

            results[stage_name] = {
                "status": "failed",
                "error_type": type(
                    exc
                ).__name__,
                "message": str(exc)[:4000],
                "duration_ms": round(
                    (
                        time.perf_counter()
                        - stage_started
                    )
                    * 1000,
                    2,
                ),
            }

            logger.exception(
                "Document stage '%s' failed for %s.",
                stage_name,
                document_id,
            )

            # Parsing/chunking/indexing failures make
            # downstream results unreliable.
            if stage_name in {
                "ingestion",
                "parsing",
                "chunking",
                "embeddings",
                "indexing",
            }:
                break

    status = (
        "failed"
        if failures
        else "completed"
    )

    update_document_status(
        document_id,
        status,
        error=(
            "Failed stages: "
            + ", ".join(failures)
            if failures
            else None
        ),
        metadata={
            "pipeline_completed_at": utc_iso(),
            "pipeline_stages": list(
                results.keys()
            ),
        },
    )

    summary = {
        "task": "process_document_pipeline",
        "document_id": document_id,
        "document_version_id": document_version_id,
        "status": status,
        "started_at": utc_iso(),
        "completed_at": utc_iso(),
        "duration_ms": round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        ),
        "stages": results,
        "failed_stages": failures,
        "stage_count": len(results),
    }

    persist_stage_result(
        document_id,
        document_version_id,
        "pipeline",
        summary,
    )

    return summary


# ---------------------------------------------------------------------------
# Celery task base
# ---------------------------------------------------------------------------


class DocumentTaskBase(
    Task
):
    """
    Celery base task with retry configuration.

    When Celery is unavailable, this class is harmless and
    the synchronous task functions remain usable directly.
    """

    autoretry_for = ()
    max_retries = MAX_TASK_RETRIES
    time_limit = TASK_TIME_LIMIT
    soft_time_limit = SOFT_TASK_TIME_LIMIT

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        document_id = (
            kwargs.get(
                "document_id"
            )
            or (
                args[0]
                if args
                else None
            )
        )

        if document_id:
            update_document_status(
                str(document_id),
                "failed",
                error=str(exc),
                metadata={
                    "task_id": task_id,
                },
            )

        logger.error(
            "Celery document task failed: "
            "task_id=%s document_id=%s error=%s",
            task_id,
            document_id,
            exc,
        )


# ---------------------------------------------------------------------------
# Celery registration
# ---------------------------------------------------------------------------


def get_celery_app() -> Any | None:
    module = import_first(
        (
            "backend.workers.celery_app",
            "backend.tasks.celery_app",
            "backend.celery_app",
        )
    )

    if module is None:
        return None

    for attribute in (
        "celery_app",
        "app",
        "celery",
    ):
        application = getattr(
            module,
            attribute,
            None,
        )

        if application is not None:
            return application

    return None


celery_app = get_celery_app()


def register_celery_task(
    name: str,
    function: Callable[..., Any],
) -> Any:
    if celery_app is None:
        return function

    return celery_app.task(
        bind=True,
        base=DocumentTaskBase,
        name=name,
        max_retries=MAX_TASK_RETRIES,
        time_limit=TASK_TIME_LIMIT,
        soft_time_limit=SOFT_TASK_TIME_LIMIT,
    )(
        function
    )


# ---------------------------------------------------------------------------
# Celery task wrappers
# ---------------------------------------------------------------------------


def _celery_ingest(
    self: Any,
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    try:
        result = ingest_document(
            document_id,
            document_version_id,
        )

        if result.get(
            "status"
        ) == "failed":
            raise RuntimeError(
                result.get(
                    "message",
                    "Document ingestion failed.",
                )
            )

        return result

    except Exception as exc:
        countdown = (
            RETRY_BACKOFF_SECONDS
            * (
                2
                ** self.request.retries
            )
        )

        raise self.retry(
            exc=exc,
            countdown=countdown,
        )


def _celery_parse(
    self: Any,
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    try:
        result = parse_document(
            document_id,
            document_version_id,
        )

        if result.get(
            "status"
        ) == "failed":
            raise RuntimeError(
                result.get(
                    "message",
                    "Document parsing failed.",
                )
            )

        return result

    except Exception as exc:
        countdown = (
            RETRY_BACKOFF_SECONDS
            * (
                2
                ** self.request.retries
            )
        )

        raise self.retry(
            exc=exc,
            countdown=countdown,
        )


def _celery_pipeline(
    self: Any,
    document_id: str,
    document_version_id: str | None = None,
) -> dict[str, Any]:
    try:
        result = process_document_pipeline(
            document_id,
            document_version_id,
        )

        if result.get(
            "status"
        ) == "failed":
            raise RuntimeError(
                "Document pipeline failed: "
                + ", ".join(
                    result.get(
                        "failed_stages",
                        [],
                    )
                )
            )

        return result

    except Exception as exc:
        countdown = (
            RETRY_BACKOFF_SECONDS
            * (
                2
                ** self.request.retries
            )
        )

        raise self.retry(
            exc=exc,
            countdown=countdown,
        )


# Register only when a Celery application exists.
if celery_app is not None:
    ingest_document_task = (
        register_celery_task(
            "legal.document.ingest",
            _celery_ingest,
        )
    )

    parse_document_task = (
        register_celery_task(
            "legal.document.parse",
            _celery_parse,
        )
    )

    process_document_pipeline_task = (
        register_celery_task(
            "legal.document.pipeline",
            _celery_pipeline,
        )
    )
else:
    ingest_document_task = ingest_document
    parse_document_task = parse_document
    process_document_pipeline_task = (
        process_document_pipeline
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "DocumentTaskResult",
    "ingest_document",
    "parse_document",
    "process_document_ocr",
    "extract_document_metadata",
    "chunk_document",
    "create_document_embeddings",
    "index_document",
    "extract_legal_entities",
    "analyze_document",
    "process_document_pipeline",
    "ingest_document_task",
    "parse_document_task",
    "process_document_pipeline_task",
]