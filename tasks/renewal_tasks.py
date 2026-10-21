"""
Celery tasks for contract renewal management.

Responsibilities:
- Detect upcoming contract renewals.
- Detect overdue renewals.
- Create renewal notifications.
- Queue notifications for asynchronous delivery.
- Retry failed renewal processing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from celery import shared_task
from sqlalchemy.orm import Session

from backend.db.database import SessionLocal
from backend.services.notification_service import NotificationService
from backend.services.renewal_service import RenewalService

logger = logging.getLogger(__name__)


def _get_db() -> Session:
    """Create a database session for a background task."""
    return SessionLocal()


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def process_renewal(
    self,
    renewal_id: int,
) -> dict[str, Any]:
    """
    Process a single contract renewal.

    The renewal service owns the business logic while Celery
    handles asynchronous execution and retry behavior.
    """
    db = _get_db()

    try:
        service = RenewalService(db)

        renewal = service.get_renewal(renewal_id)

        if renewal is None:
            logger.warning(
                "Renewal not found",
                extra={"renewal_id": renewal_id},
            )

            return {
                "success": False,
                "renewal_id": renewal_id,
                "status": "not_found",
            }

        result = service.process_renewal(renewal_id)

        logger.info(
            "Renewal processed",
            extra={
                "renewal_id": renewal_id,
                "result": result,
            },
        )

        return {
            "success": True,
            "renewal_id": renewal_id,
            "status": "processed",
            "result": result,
        }

    except Exception:
        logger.exception(
            "Renewal processing failed",
            extra={"renewal_id": renewal_id},
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
def process_upcoming_renewals(
    self,
    days_ahead: int = 30,
) -> dict[str, Any]:
    """
    Find contracts approaching renewal and process them.

    Intended for periodic Celery Beat execution.
    """
    db = _get_db()

    try:
        renewal_service = RenewalService(db)

        renewals = renewal_service.get_upcoming_renewals(
            days_ahead=days_ahead,
        )

        processed = 0
        failed = 0

        for renewal in renewals:
            try:
                process_renewal.delay(renewal.id)
                processed += 1

            except Exception:
                failed += 1

                logger.exception(
                    "Failed to queue renewal",
                    extra={"renewal_id": renewal.id},
                )

        logger.info(
            "Upcoming renewals queued",
            extra={
                "days_ahead": days_ahead,
                "found": len(renewals),
                "queued": processed,
                "failed": failed,
            },
        )

        return {
            "success": True,
            "days_ahead": days_ahead,
            "found": len(renewals),
            "queued": processed,
            "failed": failed,
        }

    finally:
        db.close()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def process_overdue_renewals(
    self,
) -> dict[str, Any]:
    """
    Detect contracts whose renewal dates have already passed.
    """
    db = _get_db()

    try:
        renewal_service = RenewalService(db)

        renewals = renewal_service.get_overdue_renewals(
            as_of=_utc_now(),
        )

        processed = 0

        for renewal in renewals:
            renewal_service.mark_overdue(renewal.id)
            processed += 1

        logger.info(
            "Overdue renewals processed",
            extra={
                "found": len(renewals),
                "processed": processed,
            },
        )

        return {
            "success": True,
            "found": len(renewals),
            "processed": processed,
        }

    finally:
        db.close()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def create_renewal_notifications(
    self,
    days_ahead: int = 30,
) -> dict[str, Any]:
    """
    Create notifications for upcoming renewals.

    Notification persistence and delivery remain inside the
    notification service.
    """
    db = _get_db()

    try:
        renewal_service = RenewalService(db)
        notification_service = NotificationService(db)

        renewals = renewal_service.get_upcoming_renewals(
            days_ahead=days_ahead,
        )

        created = 0
        skipped = 0

        for renewal in renewals:
            notification = notification_service.create_renewal_notification(
                renewal_id=renewal.id,
            )

            if notification is None:
                skipped += 1
                continue

            created += 1

            send_renewal_notification.delay(
                notification.id,
            )

        logger.info(
            "Renewal notifications created",
            extra={
                "days_ahead": days_ahead,
                "renewals": len(renewals),
                "created": created,
                "skipped": skipped,
            },
        )

        return {
            "success": True,
            "days_ahead": days_ahead,
            "renewals": len(renewals),
            "created": created,
            "skipped": skipped,
        }

    finally:
        db.close()


@shared_task(
    bind=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=300,
    retry_kwargs={"max_retries": 5},
)
def send_renewal_notification(
    self,
    notification_id: int,
) -> dict[str, Any]:
    """
    Deliver a renewal notification.
    """
    db = _get_db()

    try:
        service = NotificationService(db)

        notification = service.get_notification(notification_id)

        if notification is None:
            logger.warning(
                "Renewal notification not found",
                extra={"notification_id": notification_id},
            )

            return {
                "success": False,
                "notification_id": notification_id,
                "status": "not_found",
            }

        service.send_notification(notification_id)

        logger.info(
            "Renewal notification sent",
            extra={
                "notification_id": notification_id,
            },
        )

        return {
            "success": True,
            "notification_id": notification_id,
            "status": "sent",
        }

    finally:
        db.close()


@shared_task
def refresh_renewal_statuses() -> dict[str, Any]:
    """
    Refresh renewal statuses for active contracts.

    Suitable for scheduled execution.
    """
    db = _get_db()

    try:
        service = RenewalService(db)

        updated = service.refresh_statuses(
            as_of=_utc_now(),
        )

        logger.info(
            "Renewal statuses refreshed",
            extra={"updated": updated},
        )

        return {
            "success": True,
            "updated": updated,
        }

    finally:
        db.close()


@shared_task
def cleanup_completed_renewals(
    days: int = 365,
) -> dict[str, Any]:
    """
    Archive or clean up old completed renewal records.

    The actual retention policy should be controlled by the
    application's governance configuration.
    """
    db = _get_db()

    try:
        service = RenewalService(db)

        cleaned = service.cleanup_completed_renewals(
            older_than_days=days,
        )

        logger.info(
            "Completed renewals cleaned up",
            extra={
                "older_than_days": days,
                "cleaned": cleaned,
            },
        )

        return {
            "success": True,
            "older_than_days": days,
            "cleaned": cleaned,
        }

    finally:
        db.close()