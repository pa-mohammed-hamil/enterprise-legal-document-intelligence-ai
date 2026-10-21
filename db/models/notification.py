# backend/db/models/notification.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Notification(Base):
    """
    Enterprise notification and delivery record.

    Supports:
        - in-app notifications
        - email
        - SMS
        - push
        - webhook
        - contract expiry alerts
        - renewal alerts
        - obligation reminders
        - risk escalations
        - review assignments
        - security alerts
        - system/model alerts

    Important design rule:

        Delivery state and user read state are independent.

        Example:
            delivery_status = "delivered"
            is_read = False

    Workers should claim scheduled notifications atomically to
    prevent duplicate sends.
    """

    __tablename__ = "notifications"

    # ========================================================
    # IDENTITY / TENANCY
    # ========================================================

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # RECIPIENT
    # ========================================================

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    recipient_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="user",
        index=True,
    )

    recipient_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    recipient_email: Mapped[Optional[str]] = mapped_column(
        String(320),
        nullable=True,
    )

    recipient_phone: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    recipient_endpoint: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
    )

    # ========================================================
    # NOTIFICATION
    # ========================================================

    notification_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    priority: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="normal",
        index=True,
    )

    severity: Mapped[Optional[str]] = mapped_column(
        String(30),
        nullable=True,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    summary: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # CHANNEL
    # ========================================================

    channel: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="in_app",
        index=True,
    )

    provider: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    provider_message_id: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
    )

    # ========================================================
    # DELIVERY STATE
    # ========================================================

    delivery_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    delivery_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    max_delivery_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
    )

    last_attempt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    next_retry_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    delivered_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    failed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    cancelled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # USER READ STATE
    # ========================================================

    is_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    read_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    is_seen: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # SCHEDULING
    # ========================================================

    scheduled_for: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    timezone_name: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ========================================================
    # WORKER CLAIM / CONCURRENCY
    # ========================================================

    claimed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    claimed_by: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    claimed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    claim_token: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # RELATED LEGAL RESOURCES
    # ========================================================

    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    contract_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "contracts.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    clause_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "clauses.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    obligation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "obligations.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    risk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "risks.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    renewal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "renewals.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    review_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "reviews.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ========================================================
    # ACTION
    # ========================================================

    action_url: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
    )

    action_label: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    requires_action: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    action_completed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    action_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # ESCALATION
    # ========================================================

    escalated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    escalation_level: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
    )

    escalation_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    escalated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # DEDUPLICATION / IDEMPOTENCY
    # ========================================================

    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    deduplication_key: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    event_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # TEMPLATE
    # ========================================================

    template_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    template_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    template_variables: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # PROVIDER RESPONSE
    # ========================================================

    provider_status: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    provider_response: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # ERROR
    # ========================================================

    error_code: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    error_type: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    error_reference: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    # ========================================================
    # REQUEST / TRACE
    # ========================================================

    request_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    trace_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # EMAIL-SPECIFIC
    # ========================================================

    email_subject: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    email_from: Mapped[Optional[str]] = mapped_column(
        String(320),
        nullable=True,
    )

    email_reply_to: Mapped[Optional[str]] = mapped_column(
        String(320),
        nullable=True,
    )

    # ========================================================
    # EXTENSION METADATA
    # ========================================================

    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # TIMESTAMPS
    # ========================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )

    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )

    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    document = relationship(
        "Document",
        foreign_keys=[document_id],
        lazy="selectin",
    )

    contract = relationship(
        "Contract",
        foreign_keys=[contract_id],
        lazy="selectin",
    )

    clause = relationship(
        "Clause",
        foreign_keys=[clause_id],
        lazy="selectin",
    )

    obligation = relationship(
        "Obligation",
        foreign_keys=[obligation_id],
        lazy="selectin",
    )

    risk = relationship(
        "Risk",
        foreign_keys=[risk_id],
        lazy="selectin",
    )

    renewal = relationship(
        "Renewal",
        foreign_keys=[renewal_id],
        lazy="selectin",
    )

    review = relationship(
        "Review",
        foreign_keys=[review_id],
        lazy="selectin",
    )

    # ========================================================
    # CONSTRAINTS / INDEXES
    # ========================================================

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_notification_tenant_idempotency",
        ),

        Index(
            "ix_notifications_tenant_user_created",
            "tenant_id",
            "user_id",
            "created_at",
        ),

        Index(
            "ix_notifications_tenant_unread",
            "tenant_id",
            "user_id",
            "is_read",
            "created_at",
        ),

        Index(
            "ix_notifications_tenant_delivery",
            "tenant_id",
            "delivery_status",
            "scheduled_for",
        ),

        Index(
            "ix_notifications_due_worker",
            "delivery_status",
            "claimed",
            "scheduled_for",
            "next_retry_at",
        ),

        Index(
            "ix_notifications_priority",
            "tenant_id",
            "priority",
            "created_at",
        ),

        Index(
            "ix_notifications_type",
            "tenant_id",
            "notification_type",
            "created_at",
        ),

        Index(
            "ix_notifications_contract",
            "tenant_id",
            "contract_id",
        ),

        Index(
            "ix_notifications_document",
            "tenant_id",
            "document_id",
        ),

        Index(
            "ix_notifications_risk",
            "tenant_id",
            "risk_id",
        ),

        Index(
            "ix_notifications_renewal",
            "tenant_id",
            "renewal_id",
        ),

        Index(
            "ix_notifications_review",
            "tenant_id",
            "review_id",
        ),

        Index(
            "ix_notifications_escalation",
            "tenant_id",
            "escalated",
            "escalation_level",
        ),

        Index(
            "ix_notifications_dedup",
            "tenant_id",
            "deduplication_key",
        ),

        Index(
            "ix_notifications_trace",
            "tenant_id",
            "trace_id",
        ),
    )

    # ========================================================
    # DOMAIN PROPERTIES
    # ========================================================

    @property
    def unread(self) -> bool:
        return not self.is_read

    @property
    def delivered(self) -> bool:
        return (
            self.delivery_status
            or ""
        ).lower() == "delivered"

    @property
    def failed(self) -> bool:
        return (
            self.delivery_status
            or ""
        ).lower() == "failed"

    @property
    def cancelled(self) -> bool:
        return (
            self.delivery_status
            or ""
        ).lower() == "cancelled"

    @property
    def expired(self) -> bool:
        return bool(
            self.expires_at is not None
            and self.expires_at <= utcnow()
        )

    @property
    def retry_available(self) -> bool:
        return bool(
            self.delivery_attempts
            < self.max_delivery_attempts
            and not self.cancelled
            and not self.delivered
            and not self.expired
        )

    @property
    def due(self) -> bool:
        if self.expired:
            return False

        if self.cancelled or self.delivered:
            return False

        now = utcnow()

        if (
            self.scheduled_for is not None
            and self.scheduled_for > now
        ):
            return False

        if (
            self.next_retry_at is not None
            and self.next_retry_at > now
        ):
            return False

        return True

    @property
    def resource_reference(self) -> Optional[str]:
        for resource_type, value in (
            ("document", self.document_id),
            ("contract", self.contract_id),
            ("clause", self.clause_id),
            ("obligation", self.obligation_id),
            ("risk", self.risk_id),
            ("renewal", self.renewal_id),
            ("review", self.review_id),
        ):
            if value is not None:
                return (
                    f"{resource_type}:{value}"
                )

        return None

    # ========================================================
    # READ STATE
    # ========================================================

    def mark_seen(self) -> None:
        if not self.is_seen:
            self.is_seen = True
            self.seen_at = utcnow()

    def mark_read(self) -> None:
        self.is_read = True

        if self.read_at is None:
            self.read_at = utcnow()

        self.mark_seen()

    def mark_unread(self) -> None:
        self.is_read = False
        self.read_at = None

    # ========================================================
    # DELIVERY
    # ========================================================

    def mark_sending(self) -> None:
        self.delivery_status = "sending"
        self.last_attempt_at = utcnow()
        self.delivery_attempts += 1

    def mark_delivered(
        self,
        *,
        provider_message_id: Optional[str] = None,
        provider_status: Optional[str] = None,
        provider_response: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:
        self.delivery_status = "delivered"
        self.delivered_at = utcnow()

        self.failed_at = None
        self.next_retry_at = None

        if provider_message_id is not None:
            self.provider_message_id = (
                provider_message_id
            )

        if provider_status is not None:
            self.provider_status = (
                provider_status
            )

        if provider_response is not None:
            self.provider_response = (
                provider_response
            )

        self.release_claim()

    def mark_failed(
        self,
        *,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_reference: Optional[str] = None,
        next_retry_at: Optional[
            datetime
        ] = None,
    ) -> None:
        self.failed_at = utcnow()

        self.error_code = error_code
        self.error_type = error_type
        self.error_message = error_message
        self.error_reference = error_reference

        if (
            self.delivery_attempts
            >= self.max_delivery_attempts
        ):
            self.delivery_status = "failed"
            self.next_retry_at = None
        else:
            self.delivery_status = "retry"
            self.next_retry_at = next_retry_at

        self.release_claim()

    def cancel(self) -> None:
        if self.delivered:
            raise ValueError(
                "Delivered notification cannot "
                "be cancelled."
            )

        self.delivery_status = "cancelled"
        self.cancelled_at = utcnow()
        self.next_retry_at = None

        self.release_claim()

    # ========================================================
    # WORKER CLAIM
    # ========================================================

    def claim(
        self,
        *,
        worker_id: str,
        token: Optional[str] = None,
    ) -> None:
        if self.claimed:
            raise ValueError(
                "Notification is already claimed."
            )

        if not self.due:
            raise ValueError(
                "Notification is not due."
            )

        self.claimed = True
        self.claimed_by = worker_id
        self.claimed_at = utcnow()
        self.claim_token = (
            token
            or uuid.uuid4().hex
        )

    def release_claim(self) -> None:
        self.claimed = False
        self.claimed_by = None
        self.claimed_at = None
        self.claim_token = None

    # ========================================================
    # ACTION / ESCALATION
    # ========================================================

    def complete_action(self) -> None:
        self.action_completed = True
        self.action_completed_at = utcnow()

    def escalate(
        self,
        *,
        level: str,
        reason: Optional[str] = None,
    ) -> None:
        self.escalated = True
        self.escalation_level = level
        self.escalation_reason = reason
        self.escalated_at = utcnow()

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            "<Notification("
            f"id={self.id}, "
            f"type={self.notification_type!r}, "
            f"channel={self.channel!r}, "
            f"status={self.delivery_status!r}, "
            f"priority={self.priority!r}"
            ")>"
        )