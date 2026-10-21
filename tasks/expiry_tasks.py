"""
Background tasks for legal-document expiry, obligation deadlines,
renewals, reminders, escalation, and notification workflows.

Responsibilities:
    - Detect upcoming document/contract expirations
    - Detect overdue obligations
    - Create renewal reminders
    - Escalate critical deadlines
    - Avoid duplicate notifications
    - Support Celery and synchronous/local execution
    - Respect tenant/organization boundaries
    - Produce auditable task results

The implementation is intentionally defensive so it can work with
slightly different SQLAlchemy model/repository layouts while the
application evolves.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
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
    "legal-expiry-tasks"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_REMINDER_DAYS = tuple(
    sorted(
        {
            int(value.strip())
            for value in os.getenv(
                "EXPIRY_REMINDER_DAYS",
                "90,60,30,14,7,3,1",
            ).split(",")
            if value.strip().isdigit()
        },
        reverse=True,
    )
)

CRITICAL_REMINDER_DAYS = int(
    os.getenv(
        "EXPIRY_CRITICAL_DAYS",
        "7",
    )
)

OVERDUE_GRACE_DAYS = int(
    os.getenv(
        "EXPIRY_OVERDUE_GRACE_DAYS",
        "0",
    )
)

TASK_MAX_RETRIES = int(
    os.getenv(
        "EXPIRY_TASK_MAX_RETRIES",
        "3",
    )
)

TASK_RETRY_BACKOFF_SECONDS = int(
    os.getenv(
        "EXPIRY_TASK_RETRY_BACKOFF_SECONDS",
        "30",
    )
)

NOTIFICATION_ENABLED = (
    os.getenv(
        "EXPIRY_NOTIFICATIONS_ENABLED",
        "true",
    ).lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

# ---------------------------------------------------------------------------
# Optional Celery
# ---------------------------------------------------------------------------

try:
    from celery import Task

    CELERY_AVAILABLE = True
except ImportError:
    Task = object  # type: ignore
    CELERY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExpiryTaskResult:
    task: str
    run_id: str
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    scanned: int = 0
    expiring: int = 0
    overdue: int = 0
    reminders_created: int = 0
    notifications_sent: int = 0
    escalations_created: int = 0
    skipped: int = 0
    errors: int = 0
    message: str | None = None
    error_type: str | None = None
    details: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpiryRecord:
    id: str
    entity_type: str
    entity_id: str
    expiry_date: date
    title: str
    tenant_id: str | None = None
    organization_id: str | None = None
    owner_id: str | None = None
    status: str = "active"
    priority: str = "normal"
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_today() -> date:
    return utc_now().date()


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
            f"legal-expiry:{value}",
        )
    )


def parse_date(
    value: Any,
) -> date | None:
    if value is None:
        return None

    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    if isinstance(
        value,
        date,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        value = value.strip()

        if not value:
            return None

        candidates = (
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
        )

        for fmt in candidates:
            try:
                return datetime.strptime(
                    value,
                    fmt,
                ).date()
            except ValueError:
                continue

        try:
            return datetime.fromisoformat(
                value
            ).date()
        except ValueError:
            return None

    return None


def days_until(
    expiry_date: date,
    reference_date: date | None = None,
) -> int:
    reference = (
        reference_date
        or utc_today()
    )

    return (
        expiry_date
        - reference
    ).days


def ensure_list(
    value: Any,
) -> list[Any]:
    if value is None:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        tuple,
    ):
        return list(value)

    return [value]


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
# Database
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
            "Unable to locate database SessionLocal."
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


def model_columns(
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


def object_columns(
    obj: Any,
) -> set[str]:
    try:
        return model_columns(
            type(obj)
        )
    except Exception:
        return set()


def set_existing_attributes(
    obj: Any,
    values: dict[str, Any],
) -> None:
    columns = object_columns(
        obj
    )

    for key, value in values.items():
        if (
            key in columns
            or hasattr(obj, key)
        ):
            try:
                setattr(
                    obj,
                    key,
                    value,
                )
            except Exception:
                logger.debug(
                    "Unable to set %s on %s.",
                    key,
                    type(obj).__name__,
                    exc_info=True,
                )


# ---------------------------------------------------------------------------
# Expiry field discovery
# ---------------------------------------------------------------------------


EXPIRY_FIELDS = (
    "expiry_date",
    "expires_at",
    "expiration_date",
    "valid_until",
    "valid_to",
    "end_date",
    "contract_end_date",
)

STATUS_FIELDS = (
    "status",
    "document_status",
    "contract_status",
)

TENANT_FIELDS = (
    "tenant_id",
    "tenant",
)

ORGANIZATION_FIELDS = (
    "organization_id",
    "org_id",
)

OWNER_FIELDS = (
    "owner_id",
    "user_id",
    "assigned_to",
)


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


def extract_expiry_date(
    obj: Any,
) -> date | None:
    for field in EXPIRY_FIELDS:
        value = getattr(
            obj,
            field,
            None,
        )

        parsed = parse_date(
            value
        )

        if parsed:
            return parsed

    return None


def is_active_record(
    obj: Any,
) -> bool:
    status = first_attribute(
        obj,
        STATUS_FIELDS,
    )

    if status is None:
        return True

    normalized = str(
        status
    ).strip().lower()

    return normalized not in {
        "deleted",
        "archived",
        "cancelled",
        "canceled",
        "inactive",
        "expired",
        "terminated",
    }


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def expiry_models() -> list[tuple[str, Any]]:
    candidates = (
        "Document",
        "Contract",
        "Renewal",
        "Obligation",
    )

    models: list[
        tuple[str, Any]
    ] = []

    for model_name in candidates:
        model = get_model(
            model_name
        )

        if model is not None:
            models.append(
                (
                    model_name,
                    model,
                )
            )

    return models


def query_model_records(
    session: Any,
    model: Any,
    limit: int = 10000,
) -> list[Any]:
    try:
        query = session.query(
            model
        )

        return (
            query.limit(
                limit
            )
            .all()
        )

    except Exception:
        logger.warning(
            "Unable to query %s.",
            getattr(
                model,
                "__name__",
                model,
            ),
            exc_info=True,
        )

        return []


# ---------------------------------------------------------------------------
# Expiry record normalization
# ---------------------------------------------------------------------------


def normalize_expiry_record(
    obj: Any,
    entity_type: str,
) -> ExpiryRecord | None:
    expiry = extract_expiry_date(
        obj
    )

    if expiry is None:
        return None

    object_id = first_attribute(
        obj,
        (
            "id",
            "uuid",
            "document_id",
            "contract_id",
            "obligation_id",
        ),
    )

    if object_id is None:
        return None

    title = first_attribute(
        obj,
        (
            "title",
            "name",
            "document_name",
            "contract_name",
            "subject",
        ),
    )

    if title is None:
        title = (
            f"{entity_type} "
            f"{object_id}"
        )

    tenant_id = first_attribute(
        obj,
        TENANT_FIELDS,
    )

    organization_id = first_attribute(
        obj,
        ORGANIZATION_FIELDS,
    )

    owner_id = first_attribute(
        obj,
        OWNER_FIELDS,
    )

    metadata = first_attribute(
        obj,
        (
            "metadata",
            "extra_data",
            "attributes",
        ),
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    priority = (
        str(
            first_attribute(
                obj,
                (
                    "priority",
                    "risk_level",
                    "severity",
                ),
            )
            or "normal"
        )
        .lower()
    )

    return ExpiryRecord(
        id=str(
            object_id
        ),
        entity_type=entity_type,
        entity_id=str(
            object_id
        ),
        expiry_date=expiry,
        title=str(title),
        tenant_id=(
            str(tenant_id)
            if tenant_id is not None
            else None
        ),
        organization_id=(
            str(organization_id)
            if organization_id is not None
            else None
        ),
        owner_id=(
            str(owner_id)
            if owner_id is not None
            else None
        ),
        status=str(
            first_attribute(
                obj,
                STATUS_FIELDS,
            )
            or "active"
        ),
        priority=priority,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Expiry scanning
# ---------------------------------------------------------------------------


def scan_expiring_records(
    *,
    reference_date: date | None = None,
    reminder_days: Iterable[int] = DEFAULT_REMINDER_DAYS,
    tenant_id: str | None = None,
    organization_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Scan documents/contracts/renewals/obligations for expiry dates.

    Returns normalized dictionaries instead of leaking SQLAlchemy
    objects outside the task layer.
    """

    reference = (
        reference_date
        or utc_today()
    )

    reminder_set = {
        int(days)
        for days in reminder_days
    }

    session = None
    results: list[dict[str, Any]] = []

    try:
        session = get_database_session()

        for entity_type, model in expiry_models():
            records = query_model_records(
                session,
                model,
            )

            for obj in records:
                if not is_active_record(
                    obj
                ):
                    continue

                normalized = (
                    normalize_expiry_record(
                        obj,
                        entity_type,
                    )
                )

                if normalized is None:
                    continue

                if (
                    tenant_id
                    and normalized.tenant_id
                    and normalized.tenant_id
                    != tenant_id
                ):
                    continue

                if (
                    organization_id
                    and normalized.organization_id
                    and normalized.organization_id
                    != organization_id
                ):
                    continue

                remaining = days_until(
                    normalized.expiry_date,
                    reference,
                )

                should_remind = (
                    remaining in reminder_set
                    or (
                        remaining <= 0
                        and abs(remaining)
                        <= OVERDUE_GRACE_DAYS
                    )
                )

                if not should_remind:
                    continue

                results.append(
                    {
                        **asdict(
                            normalized
                        ),
                        "days_until_expiry": remaining,
                        "is_overdue": (
                            remaining < 0
                        ),
                        "is_expiring_today": (
                            remaining == 0
                        ),
                        "reminder_required": True,
                    }
                )

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

    results.sort(
        key=lambda item: (
            item["days_until_expiry"],
            item["entity_type"],
            item["entity_id"],
        )
    )

    return results


# ---------------------------------------------------------------------------
# Priority
# ---------------------------------------------------------------------------


def calculate_priority(
    days_remaining: int,
    existing_priority: str = "normal",
) -> str:
    existing = (
        existing_priority
        or "normal"
    ).lower()

    if days_remaining < 0:
        return "critical"

    if days_remaining <= 1:
        return "critical"

    if days_remaining <= CRITICAL_REMINDER_DAYS:
        return "high"

    if existing in {
        "critical",
        "high",
    }:
        return existing

    if days_remaining <= 30:
        return "medium"

    return "normal"


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------


def get_notification_model() -> Any | None:
    return get_model(
        "Notification"
    )


def notification_exists(
    session: Any,
    *,
    notification_id: str,
) -> bool:
    model = get_notification_model()

    if model is None:
        return False

    try:
        existing = session.get(
            model,
            notification_id,
        )

        return existing is not None

    except Exception:
        try:
            return (
                session.query(model)
                .filter(
                    model.id
                    == notification_id
                )
                .first()
                is not None
            )
        except Exception:
            return False


def create_notification(
    *,
    record: dict[str, Any],
    days_remaining: int,
    notification_type: str = "expiry_reminder",
) -> bool:
    """
    Create an idempotent notification.

    A deterministic notification ID prevents duplicate reminders
    when Celery retries the same task.
    """

    if not NOTIFICATION_ENABLED:
        return False

    model = get_notification_model()

    if model is None:
        logger.warning(
            "Notification model unavailable."
        )
        return False

    notification_id = stable_uuid(
        notification_type,
        record["entity_type"],
        record["entity_id"],
        record["expiry_date"],
        days_remaining,
    )

    session = None

    try:
        session = get_database_session()

        if notification_exists(
            session,
            notification_id=notification_id,
        ):
            return False

        priority = calculate_priority(
            days_remaining,
            record.get(
                "priority",
                "normal",
            ),
        )

        if days_remaining < 0:
            title = (
                f"{record['title']} is overdue"
            )

            message = (
                f"{record['entity_type']} "
                f"'{record['title']}' expired "
                f"{abs(days_remaining)} day(s) ago."
            )

        elif days_remaining == 0:
            title = (
                f"{record['title']} expires today"
            )

            message = (
                f"{record['entity_type']} "
                f"'{record['title']}' expires today."
            )

        else:
            title = (
                f"{record['title']} expires "
                f"in {days_remaining} day(s)"
            )

            message = (
                f"{record['entity_type']} "
                f"'{record['title']}' expires in "
                f"{days_remaining} day(s)."
            )

        values = {
            "id": notification_id,
            "tenant_id": record.get(
                "tenant_id"
            ),
            "organization_id": record.get(
                "organization_id"
            ),
            "user_id": record.get(
                "owner_id"
            ),
            "type": notification_type,
            "notification_type": notification_type,
            "title": title,
            "message": message,
            "priority": priority,
            "severity": priority,
            "status": "pending",
            "is_read": False,
            "entity_type": record.get(
                "entity_type"
            ),
            "entity_id": record.get(
                "entity_id"
            ),
            "metadata": {
                "expiry_date": str(
                    record.get(
                        "expiry_date"
                    )
                ),
                "days_remaining": (
                    days_remaining
                ),
                "generated_at": utc_iso(),
            },
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        columns = model_columns(
            model
        )

        filtered = {
            key: value
            for key, value in values.items()
            if key in columns
        }

        notification = model(
            **filtered
        )

        session.add(
            notification
        )

        session.commit()

        return True

    except Exception:
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass

        logger.warning(
            "Unable to create expiry notification "
            "for %s/%s.",
            record.get("entity_type"),
            record.get("entity_id"),
            exc_info=True,
        )

        return False

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Renewal helpers
# ---------------------------------------------------------------------------


def get_renewal_model() -> Any | None:
    return get_model(
        "Renewal"
    )


def create_renewal_reminder(
    record: dict[str, Any],
    days_remaining: int,
) -> bool:
    """
    Create a renewal record when the project has a Renewal model.

    The deterministic ID keeps retries idempotent.
    """

    model = get_renewal_model()

    if model is None:
        logger.debug(
            "Renewal model unavailable; "
            "notification-only workflow will be used."
        )
        return False

    renewal_id = stable_uuid(
        "renewal",
        record["entity_type"],
        record["entity_id"],
        record["expiry_date"],
    )

    session = None

    try:
        session = get_database_session()

        existing = None

        try:
            existing = session.get(
                model,
                renewal_id,
            )
        except Exception:
            existing = None

        if existing is not None:
            return False

        values = {
            "id": renewal_id,
            "tenant_id": record.get(
                "tenant_id"
            ),
            "organization_id": record.get(
                "organization_id"
            ),
            "document_id": (
                record["entity_id"]
                if record["entity_type"]
                == "Document"
                else None
            ),
            "contract_id": (
                record["entity_id"]
                if record["entity_type"]
                == "Contract"
                else None
            ),
            "expiry_date": record[
                "expiry_date"
            ],
            "renewal_date": record[
                "expiry_date"
            ],
            "status": "pending",
            "priority": calculate_priority(
                days_remaining,
                record.get(
                    "priority",
                    "normal",
                ),
            ),
            "owner_id": record.get(
                "owner_id"
            ),
            "metadata": {
                "source": "expiry_tasks",
                "days_remaining": days_remaining,
                "created_at": utc_iso(),
            },
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        columns = model_columns(
            model
        )

        filtered = {
            key: value
            for key, value in values.items()
            if key in columns
        }

        session.add(
            model(
                **filtered
            )
        )

        session.commit()

        return True

    except Exception:
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass

        logger.warning(
            "Unable to create renewal reminder.",
            exc_info=True,
        )

        return False

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def write_expiry_audit(
    *,
    record: dict[str, Any],
    action: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Best-effort audit logging.
    """

    model = get_model(
        "AuditLog"
    )

    if model is None:
        return

    session = None

    try:
        session = get_database_session()

        values = {
            "id": stable_uuid(
                "expiry-audit",
                record.get(
                    "entity_type"
                ),
                record.get(
                    "entity_id"
                ),
                action,
                utc_today(),
            ),
            "tenant_id": record.get(
                "tenant_id"
            ),
            "organization_id": record.get(
                "organization_id"
            ),
            "action": action,
            "event_type": action,
            "resource_type": record.get(
                "entity_type"
            ),
            "resource_id": record.get(
                "entity_id"
            ),
            "metadata": {
                **(
                    metadata
                    or {}
                ),
                "generated_at": utc_iso(),
            },
            "created_at": utc_now(),
        }

        columns = model_columns(
            model
        )

        filtered = {
            key: value
            for key, value in values.items()
            if key in columns
        }

        session.add(
            model(
                **filtered
            )
        )

        session.commit()

    except Exception:
        if session is not None:
            try:
                session.rollback()
            except Exception:
                pass

        logger.debug(
            "Unable to persist expiry audit.",
            exc_info=True,
        )

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Individual workflows
# ---------------------------------------------------------------------------


def process_expiry_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    days_remaining = int(
        record[
            "days_until_expiry"
        ]
    )

    priority = calculate_priority(
        days_remaining,
        record.get(
            "priority",
            "normal",
        ),
    )

    renewal_created = (
        create_renewal_reminder(
            record,
            days_remaining,
        )
    )

    notification_created = (
        create_notification(
            record=record,
            days_remaining=days_remaining,
        )
    )

    write_expiry_audit(
        record=record,
        action=(
            "legal_document_expiry_detected"
            if days_remaining >= 0
            else "legal_document_expired"
        ),
        metadata={
            "days_remaining": days_remaining,
            "priority": priority,
            "renewal_created": renewal_created,
            "notification_created": (
                notification_created
            ),
        },
    )

    return {
        "entity_type": record[
            "entity_type"
        ],
        "entity_id": record[
            "entity_id"
        ],
        "title": record[
            "title"
        ],
        "expiry_date": str(
            record[
                "expiry_date"
            ]
        ),
        "days_remaining": days_remaining,
        "priority": priority,
        "renewal_created": renewal_created,
        "notification_created": (
            notification_created
        ),
        "overdue": (
            days_remaining < 0
        ),
    }


# ---------------------------------------------------------------------------
# Main scheduled task
# ---------------------------------------------------------------------------


def scan_and_process_expiries(
    *,
    reference_date: date | None = None,
    reminder_days: Iterable[int] = DEFAULT_REMINDER_DAYS,
    tenant_id: str | None = None,
    organization_id: str | None = None,
) -> dict[str, Any]:
    """
    Scan and process all expiring legal records.

    Intended to run daily through Celery Beat.
    """

    started = time.perf_counter()

    run_id = stable_uuid(
        "expiry-scan",
        reference_date
        or utc_today(),
        tenant_id,
        organization_id,
    )

    records = scan_expiring_records(
        reference_date=reference_date,
        reminder_days=reminder_days,
        tenant_id=tenant_id,
        organization_id=organization_id,
    )

    details: list[
        dict[str, Any]
    ] = []

    expiring = 0
    overdue = 0
    reminders_created = 0
    notifications_sent = 0
    errors = 0

    for record in records:
        try:
            result = process_expiry_record(
                record
            )

            details.append(
                result
            )

            expiring += 1

            if result["overdue"]:
                overdue += 1

            if result[
                "renewal_created"
            ]:
                reminders_created += 1

            if result[
                "notification_created"
            ]:
                notifications_sent += 1

        except Exception as exc:
            errors += 1

            logger.exception(
                "Expiry processing failed for "
                "%s/%s.",
                record.get(
                    "entity_type"
                ),
                record.get(
                    "entity_id"
                ),
            )

            details.append(
                {
                    "entity_type": record.get(
                        "entity_type"
                    ),
                    "entity_id": record.get(
                        "entity_id"
                    ),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    status = (
        "failed"
        if errors and not details
        else "partial"
        if errors
        else "completed"
    )

    result = ExpiryTaskResult(
        task="scan_and_process_expiries",
        run_id=run_id,
        status=status,
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
        scanned=len(records),
        expiring=expiring,
        overdue=overdue,
        reminders_created=(
            reminders_created
        ),
        notifications_sent=(
            notifications_sent
        ),
        errors=errors,
        details=details,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# Dedicated workflows
# ---------------------------------------------------------------------------


def scan_expiring_documents(
    days: int = 30,
    tenant_id: str | None = None,
    organization_id: str | None = None,
) -> dict[str, Any]:
    reference = utc_today()

    records = scan_expiring_records(
        reference_date=reference,
        reminder_days=range(
            0,
            days + 1,
        ),
        tenant_id=tenant_id,
        organization_id=organization_id,
    )

    document_records = [
        record
        for record in records
        if record[
            "entity_type"
        ]
        == "Document"
    ]

    return {
        "status": "completed",
        "scanned": len(
            document_records
        ),
        "documents": document_records,
    }


def scan_expiring_contracts(
    days: int = 30,
    tenant_id: str | None = None,
    organization_id: str | None = None,
) -> dict[str, Any]:
    reference = utc_today()

    records = scan_expiring_records(
        reference_date=reference,
        reminder_days=range(
            0,
            days + 1,
        ),
        tenant_id=tenant_id,
        organization_id=organization_id,
    )

    contract_records = [
        record
        for record in records
        if record[
            "entity_type"
        ]
        == "Contract"
    ]

    return {
        "status": "completed",
        "scanned": len(
            contract_records
        ),
        "contracts": contract_records,
    }


def process_overdue_items(
    tenant_id: str | None = None,
    organization_id: str | None = None,
) -> dict[str, Any]:
    records = scan_expiring_records(
        reference_date=utc_today(),
        reminder_days=(
            0,
        ),
        tenant_id=tenant_id,
        organization_id=organization_id,
    )

    overdue_records = [
        record
        for record in records
        if record[
            "days_until_expiry"
        ]
        < 0
    ]

    processed = 0

    for record in overdue_records:
        try:
            process_expiry_record(
                record
            )
            processed += 1
        except Exception:
            logger.exception(
                "Unable to process overdue item."
            )

    return {
        "status": "completed",
        "overdue": len(
            overdue_records
        ),
        "processed": processed,
    }


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def escalate_critical_expiries(
    *,
    tenant_id: str | None = None,
    organization_id: str | None = None,
) -> dict[str, Any]:
    """
    Escalate records that expire within the configured critical window.

    Escalation is represented by a high-priority notification and audit
    event. It remains idempotent because notification IDs are deterministic.
    """

    records = scan_expiring_records(
        reference_date=utc_today(),
        reminder_days=range(
            0,
            CRITICAL_REMINDER_DAYS + 1,
        ),
        tenant_id=tenant_id,
        organization_id=organization_id,
    )

    critical = [
        record
        for record in records
        if 0
        <= record[
            "days_until_expiry"
        ]
        <= CRITICAL_REMINDER_DAYS
    ]

    escalated = 0

    for record in critical:
        try:
            created = create_notification(
                record={
                    **record,
                    "priority": "critical",
                },
                days_remaining=record[
                    "days_until_expiry"
                ],
                notification_type=(
                    "expiry_escalation"
                ),
            )

            write_expiry_audit(
                record=record,
                action="legal_expiry_escalated",
                metadata={
                    "days_remaining": record[
                        "days_until_expiry"
                    ],
                    "priority": "critical",
                },
            )

            if created:
                escalated += 1

        except Exception:
            logger.exception(
                "Unable to escalate expiry %s.",
                record.get(
                    "entity_id"
                ),
            )

    return {
        "status": "completed",
        "critical_records": len(
            critical
        ),
        "escalated": escalated,
    }


# ---------------------------------------------------------------------------
# Celery
# ---------------------------------------------------------------------------


class ExpiryTaskBase(
    Task
):
    """
    Shared Celery configuration for expiry workflows.
    """

    max_retries = TASK_MAX_RETRIES

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.error(
            "Expiry task failed: task_id=%s error=%s",
            task_id,
            exc,
        )


def get_celery_app() -> Any | None:
    candidates = (
        "backend.workers.celery_app",
        "backend.tasks.celery_app",
        "backend.celery_app",
    )

    for module_name in candidates:
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
                "Unable to load Celery module.",
                exc_info=True,
            )

    return None


celery_app = get_celery_app()


# ---------------------------------------------------------------------------
# Celery task wrappers
# ---------------------------------------------------------------------------


if celery_app is not None:

    @celery_app.task(
        bind=True,
        base=ExpiryTaskBase,
        name="legal.expiry.scan_and_process",
        max_retries=TASK_MAX_RETRIES,
    )
    def scan_and_process_expiries_task(
        self: Any,
        tenant_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return scan_and_process_expiries(
                tenant_id=tenant_id,
                organization_id=organization_id,
            )

        except Exception as exc:
            countdown = (
                TASK_RETRY_BACKOFF_SECONDS
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
        base=ExpiryTaskBase,
        name="legal.expiry.process_overdue",
        max_retries=TASK_MAX_RETRIES,
    )
    def process_overdue_items_task(
        self: Any,
        tenant_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return process_overdue_items(
                tenant_id=tenant_id,
                organization_id=organization_id,
            )

        except Exception as exc:
            countdown = (
                TASK_RETRY_BACKOFF_SECONDS
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
        base=ExpiryTaskBase,
        name="legal.expiry.escalate_critical",
        max_retries=TASK_MAX_RETRIES,
    )
    def escalate_critical_expiries_task(
        self: Any,
        tenant_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return escalate_critical_expiries(
                tenant_id=tenant_id,
                organization_id=organization_id,
            )

        except Exception as exc:
            countdown = (
                TASK_RETRY_BACKOFF_SECONDS
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

    def scan_and_process_expiries_task(
        tenant_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        return scan_and_process_expiries(
            tenant_id=tenant_id,
            organization_id=organization_id,
        )

    def process_overdue_items_task(
        tenant_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        return process_overdue_items(
            tenant_id=tenant_id,
            organization_id=organization_id,
        )

    def escalate_critical_expiries_task(
        tenant_id: str | None = None,
        organization_id: str | None = None,
    ) -> dict[str, Any]:
        return escalate_critical_expiries(
            tenant_id=tenant_id,
            organization_id=organization_id,
        )


# ---------------------------------------------------------------------------
# Celery Beat schedule helper
# ---------------------------------------------------------------------------


def get_expiry_beat_schedule() -> dict[str, Any]:
    """
    Return a Celery Beat-compatible schedule.

    The actual Celery application can merge this into beat_schedule.
    """

    return {
        "legal-expiry-daily-scan": {
            "task": (
                "legal.expiry."
                "scan_and_process"
            ),
            "schedule": 86400,
            "options": {
                "queue": "legal-expiry",
            },
        },
        "legal-expiry-overdue-check": {
            "task": (
                "legal.expiry."
                "process_overdue"
            ),
            "schedule": 3600,
            "options": {
                "queue": "legal-expiry",
            },
        },
        "legal-expiry-critical-escalation": {
            "task": (
                "legal.expiry."
                "escalate_critical"
            ),
            "schedule": 21600,
            "options": {
                "queue": "legal-expiry",
            },
        },
    }


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "ExpiryTaskResult",
    "ExpiryRecord",
    "scan_expiring_records",
    "scan_expiring_documents",
    "scan_expiring_contracts",
    "process_expiry_record",
    "process_overdue_items",
    "escalate_critical_expiries",
    "scan_and_process_expiries",
    "scan_and_process_expiries_task",
    "process_overdue_items_task",
    "escalate_critical_expiries_task",
    "get_expiry_beat_schedule",
]