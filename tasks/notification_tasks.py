"""
Celery notification tasks.

Handles asynchronous notification delivery for:
- Contract expiry
- Contract renewal
- Risk alerts
- Review assignments
- General system notifications
"""

from __future__ import annotations

import logging
from typing import Any

from celery import shared_task
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.db.database import SessionLocal
from backend.services.notification_service import NotificationService

logger = get_logger(__name__) if "get_logger" in globals() else logging.getLogger(__name__)


def _get_db() -> Session:
    """Create a database session for a background task."""
    return SessionLocal()


def _process_notification(
    notification_id: int,
    *,
    notification_type: str | None = None,
) -> dict[str, Any]:
    """
    Process a persisted notification.

    The notification service owns the actual delivery logic so Celery
    remains responsible only for asynchronous orchestration.
    """
    db = _get_db()

    try:
        service = NotificationService(db)

        notification = service.get_notification(notification_id)

        if notification is None:
            logger.warning(
                "Notification not found",
                extra={"notification_id": notification_id},
            )

            return {
                "success": False,
                "notification_id": notification_id,
                "status": "not_found",
            }

        if getattr(notification, "status", None) == "sent":
            return {
                "success": True,
                "notification_id": notification_id,
                "status": "already_sent",
            }

        if notification_type:
            logger.info(
                "Processing notification",
                extra={
                    "notification_id": notification_id,
                    "notification_type": notification_type,
                },
            )

        service.send_notification(notification_id)

        return {
            "success": True,
            "notification_id": notification_id,
            "status": "sent",
        }

    except Exception:
        logger.exception(
            "Notification processing failed",
            extra={"notification_id": notification_id},
        )
        raise

    finally:
        db.close()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def send_notification_task(
    self,
    notification_id: int,
) -> dict[str, Any]:
    """Send a single queued notification."""
    return _process_notification(notification_id)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def send_expiry_notification_task(
    self,
    notification_id: int,
) -> dict[str, Any]:
    """Send a contract/document expiry notification."""
    return _process_notification(
        notification_id,
        notification_type="expiry",
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def send_renewal_notification_task(
    self,
    notification_id: int,
) -> dict[str, Any]:
    """Send a contract renewal notification."""
    return _process_notification(
        notification_id,
        notification_type="renewal",
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def send_risk_notification_task(
    self,
    notification_id: int,
) -> dict[str, Any]:
    """Send a high-risk contract or clause notification."""
    return _process_notification(
        notification_id,
        notification_type="risk",
    )


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def send_review_notification_task(
    self,
    notification_id: int,
) -> dict[str, Any]:
    """Send a human-review assignment notification."""
    return _process_notification(
        notification_id,
        notification_type="review",
    )


@shared_task
def process_pending_notifications(
    batch_size: int = 100,
) -> dict[str, Any]:
    """
    Queue pending notifications for asynchronous delivery.

    This task is suitable for periodic Celery Beat execution.
    """
    db = _get_db()

    try:
        service = NotificationService(db)

        notifications = service.get_pending_notifications(
            limit=batch_size,
        )

        queued = 0

        for notification in notifications:
            notification_id = notification.id

            send_notification_task.delay(notification_id)
            queued += 1

        logger.info(
            "Pending notifications queued",
            extra={
                "found": len(notifications),
                "queued": queued,
            },
        )

        return {
            "success": True,
            "found": len(notifications),
            "queued": queued,
        }

    finally:
        db.close()


@shared_task
def retry_failed_notifications(
    batch_size: int = 100,
) -> dict[str, Any]:
    """
    Retry notifications that previously failed delivery.
    """
    db = _get_db()

    try:
        service = NotificationService(db)

        notifications = service.get_failed_notifications(
            limit=batch_size,
        )

        queued = 0

        for notification in notifications:
            send_notification_task.delay(notification.id)
            queued += 1

        logger.info(
            "Failed notifications queued for retry",
            extra={
                "found": len(notifications),
                "queued": queued,
            },
        )

        return {
            "success": True,
            "found": len(notifications),
            "queued": queued,
        }

    finally:
        db.close()


@shared_task
def cleanup_old_notifications(
    days: int = 90,
) -> dict[str, Any]:
    """
    Remove or archive old notification records.

    Retention policy should ultimately be controlled by the
    application's governance configuration.
    """
    db = _get_db()

    try:
        service = NotificationService(db)

        deleted = service.cleanup_old_notifications(
            days=days,
        )

        logger.info(
            "Old notifications cleaned up",
            extra={
                "days": days,
                "deleted": deleted,
            },
        )

        return {
            "success": True,
            "deleted": deleted,
            "retention_days": days,
        }

    finally:
        db.close()