# backend/app/schemas/notification.py

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NotificationType(str, Enum):
    EXPIRY = "expiry"
    RENEWAL = "renewal"
    NOTICE_DEADLINE = "notice_deadline"
    CONTRACT_RISK = "contract_risk"
    CLAUSE_RISK = "clause_risk"
    OBLIGATION = "obligation"
    REVIEW = "review"
    DOCUMENT_PROCESSING = "document_processing"
    SECURITY = "security"
    SYSTEM = "system"
    GENERAL = "general"


class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NotificationChannel(str, Enum):
    IN_APP = "in_app"
    EMAIL = "email"
    WEBHOOK = "webhook"


class NotificationBase(BaseModel):
    notification_type: NotificationType

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )

    priority: NotificationPriority = NotificationPriority.NORMAL

    channels: list[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.IN_APP]
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationCreate(NotificationBase):
    organization_id: UUID

    user_id: UUID | None = None

    contract_id: UUID | None = None

    document_id: UUID | None = None

    clause_id: UUID | None = None

    renewal_id: UUID | None = None

    review_id: UUID | None = None

    scheduled_at: datetime | None = None

    expires_at: datetime | None = None

    action_url: str | None = Field(
        default=None,
        max_length=2000,
    )

    @model_validator(mode="after")
    def validate_dates(self) -> "NotificationCreate":
        if (
            self.scheduled_at is not None
            and self.expires_at is not None
            and self.expires_at <= self.scheduled_at
        ):
            raise ValueError(
                "expires_at must be later than scheduled_at"
            )

        return self


class NotificationUpdate(BaseModel):
    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    message: str | None = Field(
        default=None,
        min_length=1,
        max_length=10_000,
    )

    priority: NotificationPriority | None = None

    status: NotificationStatus | None = None

    scheduled_at: datetime | None = None

    expires_at: datetime | None = None

    action_url: str | None = Field(
        default=None,
        max_length=2000,
    )

    metadata: dict[str, Any] | None = None


class NotificationResponse(NotificationBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    organization_id: UUID

    user_id: UUID | None = None

    contract_id: UUID | None = None

    document_id: UUID | None = None

    clause_id: UUID | None = None

    renewal_id: UUID | None = None

    review_id: UUID | None = None

    status: NotificationStatus

    is_read: bool = False

    action_url: str | None = None

    scheduled_at: datetime | None = None

    sent_at: datetime | None = None

    delivered_at: datetime | None = None

    read_at: datetime | None = None

    expires_at: datetime | None = None

    created_at: datetime

    updated_at: datetime


class NotificationSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    notification_type: NotificationType

    title: str

    priority: NotificationPriority

    status: NotificationStatus

    is_read: bool

    created_at: datetime


class NotificationMarkReadRequest(BaseModel):
    notification_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )


class NotificationMarkReadResponse(BaseModel):
    updated: int = Field(
        default=0,
        ge=0,
    )

    notification_ids: list[UUID] = Field(
        default_factory=list
    )


class NotificationMarkUnreadRequest(BaseModel):
    notification_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )


class NotificationDeleteRequest(BaseModel):
    notification_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )


class NotificationDeleteResponse(BaseModel):
    deleted: int = Field(
        default=0,
        ge=0,
    )

    failed: int = Field(
        default=0,
        ge=0,
    )


class NotificationFilter(BaseModel):
    notification_types: list[NotificationType] | None = None

    priorities: list[NotificationPriority] | None = None

    statuses: list[NotificationStatus] | None = None

    channels: list[NotificationChannel] | None = None

    is_read: bool | None = None

    contract_id: UUID | None = None

    document_id: UUID | None = None

    start_date: datetime | None = None

    end_date: datetime | None = None

    search: str | None = Field(
        default=None,
        max_length=500,
    )


class NotificationListResponse(BaseModel):
    items: list[NotificationResponse] = Field(
        default_factory=list
    )

    total: int = Field(
        default=0,
        ge=0,
    )

    unread: int = Field(
        default=0,
        ge=0,
    )

    page: int = Field(
        default=1,
        ge=1,
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=200,
    )

    pages: int = Field(
        default=0,
        ge=0,
    )


class NotificationCounts(BaseModel):
    total: int = 0

    unread: int = 0

    critical: int = 0

    high_priority: int = 0

    expiry: int = 0

    renewal: int = 0

    risk: int = 0

    review: int = 0

    security: int = 0


class EmailNotificationRequest(BaseModel):
    recipient: str = Field(
        ...,
        min_length=3,
        max_length=320,
    )

    subject: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    body: str = Field(
        ...,
        min_length=1,
    )

    html_body: str | None = None

    notification_id: UUID | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationDeliveryResult(BaseModel):
    channel: NotificationChannel

    success: bool

    provider: str | None = None

    provider_message_id: str | None = None

    error: str | None = None

    delivered_at: datetime | None = None


class NotificationDeliveryResponse(BaseModel):
    notification_id: UUID

    success: bool

    results: list[NotificationDeliveryResult] = Field(
        default_factory=list
    )


class BulkNotificationCreate(BaseModel):
    user_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    notification_type: NotificationType

    title: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
    )

    priority: NotificationPriority = NotificationPriority.NORMAL

    channels: list[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.IN_APP]
    )

    contract_id: UUID | None = None

    document_id: UUID | None = None

    scheduled_at: datetime | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class BulkNotificationResult(BaseModel):
    user_id: UUID

    notification_id: UUID | None = None

    success: bool

    error: str | None = None


class BulkNotificationResponse(BaseModel):
    requested: int = 0

    created: int = 0

    failed: int = 0

    results: list[BulkNotificationResult] = Field(
        default_factory=list
    )


class NotificationPreference(BaseModel):
    notification_type: NotificationType

    in_app_enabled: bool = True

    email_enabled: bool = True

    webhook_enabled: bool = False


class NotificationPreferencesUpdate(BaseModel):
    preferences: list[NotificationPreference] = Field(
        default_factory=list
    )


class NotificationPreferencesResponse(BaseModel):
    user_id: UUID

    preferences: list[NotificationPreference] = Field(
        default_factory=list
    )

    updated_at: datetime | None = None