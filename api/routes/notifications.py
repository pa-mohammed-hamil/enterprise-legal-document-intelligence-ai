# backend/api/routes/notifications.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)
from pydantic import BaseModel, Field

from backend.db.repositories.notification_repository import (
    NotificationChannel,
    NotificationCreate,
    NotificationFilter,
    NotificationPriority,
    NotificationRepository,
    NotificationStatus,
    NotificationType,
    NotificationUpdate,
)


router = APIRouter(
    prefix="/notifications",
    tags=["Notifications"],
)


# ============================================================
# DEPENDENCIES
# Replace with backend.dependencies providers.
# ============================================================

async def get_notification_repository() -> NotificationRepository:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Notification repository dependency is not configured.",
    )


async def get_notification_service() -> Any:
    """
    Recommended NotificationService dependencies:

        NotificationRepository
        UserRepository
        ContractRepository
        RiskRepository
        RenewalRepository
        ReviewRepository
        EmailClient
        PushClient
        SMSClient
        WebhookClient
        AuditService

    Service should own:
        - delivery orchestration
        - retries
        - provider selection
        - preferences
        - quiet hours
        - channel fan-out
        - rate limiting
        - audit logging
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Notification service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class NotificationCreateRequest(BaseModel):
    user_id: Optional[UUID] = None

    notification_type: NotificationType

    title: str = Field(
        min_length=1,
        max_length=500,
    )

    message: str = Field(
        min_length=1,
        max_length=20_000,
    )

    priority: NotificationPriority = (
        NotificationPriority.NORMAL
    )

    channel: NotificationChannel = (
        NotificationChannel.IN_APP
    )

    contract_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    risk_id: Optional[UUID] = None
    renewal_id: Optional[UUID] = None
    review_id: Optional[UUID] = None

    action_url: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    scheduled_at: Optional[datetime] = None

    expires_at: Optional[datetime] = None

    dedup_key: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class NotificationUpdateRequest(BaseModel):
    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500,
    )

    message: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=20_000,
    )

    priority: Optional[
        NotificationPriority
    ] = None

    action_url: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    metadata: Optional[
        dict[str, Any]
    ] = None


class BulkNotificationRequest(BaseModel):
    user_ids: list[UUID] = Field(
        min_length=1,
        max_length=500,
    )

    notification_type: NotificationType

    title: str = Field(
        min_length=1,
        max_length=500,
    )

    message: str = Field(
        min_length=1,
        max_length=20_000,
    )

    priority: NotificationPriority = (
        NotificationPriority.NORMAL
    )

    channel: NotificationChannel = (
        NotificationChannel.IN_APP
    )

    action_url: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RetryNotificationRequest(BaseModel):
    force: bool = False


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class NotificationResponse(BaseModel):
    id: str

    tenant_id: str
    organization_id: Optional[str] = None

    user_id: Optional[str] = None

    notification_type: str

    title: str
    message: str

    priority: str
    status: str
    channel: str

    contract_id: Optional[str] = None
    document_id: Optional[str] = None
    risk_id: Optional[str] = None
    renewal_id: Optional[str] = None
    review_id: Optional[str] = None

    action_url: Optional[str] = None

    scheduled_at: Optional[datetime] = None

    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None

    expires_at: Optional[datetime] = None

    retry_count: int = 0

    error: Optional[str] = None

    dedup_key: Optional[str] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class NotificationPageResponse(BaseModel):
    items: list[NotificationResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class NotificationStatisticsResponse(BaseModel):
    total: int = 0

    unread: int = 0

    pending: int = 0
    scheduled: int = 0
    sent: int = 0
    delivered: int = 0
    failed: int = 0
    cancelled: int = 0

    by_type: dict[str, int] = Field(
        default_factory=dict,
    )

    by_priority: dict[str, int] = Field(
        default_factory=dict,
    )

    by_channel: dict[str, int] = Field(
        default_factory=dict,
    )

    by_status: dict[str, int] = Field(
        default_factory=dict,
    )


class UnreadCountResponse(BaseModel):
    unread: int


class BulkResultResponse(BaseModel):
    requested: int
    created: int
    skipped: int = 0
    failed: int = 0


class MessageResponse(BaseModel):
    message: str


# ============================================================
# HELPERS
# ============================================================

def _attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if isinstance(obj, dict):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def _enum_value(
    value: Any,
) -> Any:

    if value is None:
        return None

    return getattr(
        value,
        "value",
        value,
    )


def _optional_str(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    return str(value)


def _tenant_id(
    user: Any,
) -> str:

    value = _attr(
        user,
        "tenant_id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context is required.",
        )

    return str(value)


def _organization_id(
    user: Any,
) -> Optional[str]:

    return _optional_str(
        _attr(
            user,
            "organization_id",
        )
    )


def _user_id(
    user: Any,
) -> str:

    value = _attr(
        user,
        "id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity is required.",
        )

    return str(value)


def _roles(
    user: Any,
) -> set[str]:

    values = (
        _attr(
            user,
            "roles",
            [],
        )
        or []
    )

    if isinstance(
        values,
        str,
    ):
        values = [
            values
        ]

    single_role = _attr(
        user,
        "role",
    )

    if single_role:
        values = [
            *values,
            single_role,
        ]

    return {
        str(
            _enum_value(value)
        ).lower()
        for value
        in values
    }


def _require_notification_admin(
    user: Any,
) -> None:

    allowed = {
        "admin",
        "super_admin",
        "legal_admin",
        "compliance",
        "compliance_officer",
        "notification_admin",
    }

    if not (
        _roles(user)
        & allowed
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Notification administration "
                "permission is required."
            ),
        )


def _request_id(
    request: Request,
) -> Optional[str]:

    value = (
        getattr(
            request.state,
            "request_id",
            None,
        )
        or request.headers.get(
            "X-Request-ID"
        )
    )

    return _optional_str(
        value
    )


def _metadata(
    notification: Any,
) -> dict[str, Any]:

    return (
        _attr(
            notification,
            "notification_metadata",
        )
        or _attr(
            notification,
            "metadata_json",
        )
        or {}
    )


def _serialize_notification(
    notification: Any,
) -> dict[str, Any]:

    return {
        "id": str(
            _attr(
                notification,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                notification,
                "tenant_id",
                "",
            )
        ),

        "organization_id":
            _optional_str(
                _attr(
                    notification,
                    "organization_id",
                )
            ),

        "user_id":
            _optional_str(
                _attr(
                    notification,
                    "user_id",
                )
            ),

        "notification_type":
            str(
                _enum_value(
                    _attr(
                        notification,
                        "notification_type",
                        _attr(
                            notification,
                            "type",
                            "other",
                        ),
                    )
                )
            ),

        "title":
            str(
                _attr(
                    notification,
                    "title",
                    "",
                )
            ),

        "message":
            str(
                _attr(
                    notification,
                    "message",
                    "",
                )
            ),

        "priority":
            str(
                _enum_value(
                    _attr(
                        notification,
                        "priority",
                        "normal",
                    )
                )
            ),

        "status":
            str(
                _enum_value(
                    _attr(
                        notification,
                        "status",
                        "pending",
                    )
                )
            ),

        "channel":
            str(
                _enum_value(
                    _attr(
                        notification,
                        "channel",
                        "in_app",
                    )
                )
            ),

        "contract_id":
            _optional_str(
                _attr(
                    notification,
                    "contract_id",
                )
            ),

        "document_id":
            _optional_str(
                _attr(
                    notification,
                    "document_id",
                )
            ),

        "risk_id":
            _optional_str(
                _attr(
                    notification,
                    "risk_id",
                )
            ),

        "renewal_id":
            _optional_str(
                _attr(
                    notification,
                    "renewal_id",
                )
            ),

        "review_id":
            _optional_str(
                _attr(
                    notification,
                    "review_id",
                )
            ),

        "action_url":
            _attr(
                notification,
                "action_url",
            ),

        "scheduled_at":
            _attr(
                notification,
                "scheduled_at",
            ),

        "sent_at":
            _attr(
                notification,
                "sent_at",
            ),

        "delivered_at":
            _attr(
                notification,
                "delivered_at",
            ),

        "read_at":
            _attr(
                notification,
                "read_at",
            ),

        "failed_at":
            _attr(
                notification,
                "failed_at",
            ),

        "expires_at":
            _attr(
                notification,
                "expires_at",
            ),

        "retry_count":
            int(
                _attr(
                    notification,
                    "retry_count",
                    0,
                )
                or 0
            ),

        "error":
            _attr(
                notification,
                "error",
                _attr(
                    notification,
                    "error_message",
                ),
            ),

        "dedup_key":
            _attr(
                notification,
                "dedup_key",
            ),

        "metadata":
            _metadata(
                notification
            ),

        "created_at":
            _attr(
                notification,
                "created_at",
            ),

        "updated_at":
            _attr(
                notification,
                "updated_at",
            ),
    }


def _serialize_page(
    page: Any,
) -> dict[str, Any]:

    items = list(
        _attr(
            page,
            "items",
            [],
        )
        or []
    )

    total = int(
        _attr(
            page,
            "total",
            len(items),
        )
        or 0
    )

    limit = int(
        _attr(
            page,
            "limit",
            len(items),
        )
        or len(items)
    )

    offset = int(
        _attr(
            page,
            "offset",
            0,
        )
        or 0
    )

    return {
        "items": [
            _serialize_notification(
                item
            )
            for item
            in items
        ],

        "total": total,
        "limit": limit,
        "offset": offset,

        "has_more": bool(
            _attr(
                page,
                "has_more",
                (
                    offset
                    + len(items)
                    < total
                ),
            )
        ),
    }


def _set_filter(
    filters: Any,
    field: str,
    value: Any,
) -> None:

    if value is None:
        return

    if hasattr(
        filters,
        field,
    ):
        setattr(
            filters,
            field,
            value,
        )


async def _service_call(
    service: Any,
    operation: str,
    **kwargs: Any,
) -> Any:

    method = getattr(
        service,
        operation,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Notification operation "
                f"'{operation}' is not configured."
            ),
        )

    return await method(
        **kwargs
    )


# ============================================================
# LIST CURRENT USER NOTIFICATIONS
# ============================================================

@router.get(
    "",
    response_model=NotificationPageResponse,
    summary="List notifications",
)
async def list_notifications(
    notification_type: Optional[
        NotificationType
    ] = None,

    notification_status: Optional[
        NotificationStatus
    ] = Query(
        default=None,
        alias="status",
    ),

    priority: Optional[
        NotificationPriority
    ] = None,

    channel: Optional[
        NotificationChannel
    ] = None,

    unread_only: bool = False,

    contract_id: Optional[
        UUID
    ] = None,

    document_id: Optional[
        UUID
    ] = None,

    risk_id: Optional[
        UUID
    ] = None,

    renewal_id: Optional[
        UUID
    ] = None,

    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    filters = NotificationFilter(
        tenant_id=_tenant_id(
            current_user
        )
    )

    _set_filter(
        filters,
        "organization_id",
        _organization_id(
            current_user
        ),
    )

    _set_filter(
        filters,
        "user_id",
        _user_id(
            current_user
        ),
    )

    if notification_type:
        if hasattr(
            filters,
            "types",
        ):
            filters.types = [
                notification_type.value
            ]
        else:
            _set_filter(
                filters,
                "notification_type",
                notification_type.value,
            )

    if notification_status:
        if hasattr(
            filters,
            "statuses",
        ):
            filters.statuses = [
                notification_status.value
            ]
        else:
            _set_filter(
                filters,
                "status",
                notification_status.value,
            )

    if priority:
        if hasattr(
            filters,
            "priorities",
        ):
            filters.priorities = [
                priority.value
            ]
        else:
            _set_filter(
                filters,
                "priority",
                priority.value,
            )

    if channel:
        if hasattr(
            filters,
            "channels",
        ):
            filters.channels = [
                channel.value
            ]
        else:
            _set_filter(
                filters,
                "channel",
                channel.value,
            )

    _set_filter(
        filters,
        "unread_only",
        unread_only,
    )

    _set_filter(
        filters,
        "contract_id",
        str(contract_id)
        if contract_id
        else None,
    )

    _set_filter(
        filters,
        "document_id",
        str(document_id)
        if document_id
        else None,
    )

    _set_filter(
        filters,
        "risk_id",
        str(risk_id)
        if risk_id
        else None,
    )

    _set_filter(
        filters,
        "renewal_id",
        str(renewal_id)
        if renewal_id
        else None,
    )

    page = await repository.list(
        filters,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# UNREAD
# ============================================================

@router.get(
    "/unread",
    response_model=NotificationPageResponse,
    summary="List unread notifications",
)
async def unread_notifications(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_unread(
        tenant_id=_tenant_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


@router.get(
    "/unread/count",
    response_model=UnreadCountResponse,
    summary="Get unread notification count",
)
async def unread_count(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, int]:

    count = await repository.count(
        NotificationFilter(
            tenant_id=_tenant_id(
                current_user
            ),
            user_id=_user_id(
                current_user
            ),
            unread_only=True,
        )
    )

    return {
        "unread": int(
            count
        )
    }


# ============================================================
# DUE NOTIFICATIONS
# ============================================================

@router.get(
    "/due",
    response_model=NotificationPageResponse,
    summary="List due notifications",
)
async def due_notifications(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    _require_notification_admin(
        current_user
    )

    page = await repository.list_due(
        tenant_id=_tenant_id(
            current_user
        ),

        now=datetime.now(
            timezone.utc
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# FAILED NOTIFICATIONS
# ============================================================

@router.get(
    "/failed",
    response_model=NotificationPageResponse,
    summary="List failed notifications",
)
async def failed_notifications(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),

    offset: int = Query(
        default=0,
        ge=0,
    ),

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    _require_notification_admin(
        current_user
    )

    page = await repository.list_failed(
        tenant_id=_tenant_id(
            current_user
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# STATISTICS
# ============================================================

@router.get(
    "/statistics",
    response_model=NotificationStatisticsResponse,
    summary="Get notification statistics",
)
async def notification_statistics(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> Any:

    result = await repository.statistics(
        NotificationFilter(
            tenant_id=_tenant_id(
                current_user
            ),
            user_id=_user_id(
                current_user
            ),
        )
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    return {
        "total":
            _attr(
                result,
                "total",
                0,
            ),

        "unread":
            _attr(
                result,
                "unread",
                0,
            ),

        "pending":
            _attr(
                result,
                "pending",
                0,
            ),

        "scheduled":
            _attr(
                result,
                "scheduled",
                0,
            ),

        "sent":
            _attr(
                result,
                "sent",
                0,
            ),

        "delivered":
            _attr(
                result,
                "delivered",
                0,
            ),

        "failed":
            _attr(
                result,
                "failed",
                0,
            ),

        "cancelled":
            _attr(
                result,
                "cancelled",
                0,
            ),

        "by_type":
            _attr(
                result,
                "by_type",
                {},
            ),

        "by_priority":
            _attr(
                result,
                "by_priority",
                {},
            ),

        "by_channel":
            _attr(
                result,
                "by_channel",
                {},
            ),

        "by_status":
            _attr(
                result,
                "by_status",
                {},
            ),
    }


# ============================================================
# MARK ALL READ
# ============================================================

@router.post(
    "/read-all",
    response_model=MessageResponse,
    summary="Mark all notifications as read",
)
async def mark_all_read(
    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, str]:

    await repository.mark_all_read(
        tenant_id=_tenant_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),
    )

    return {
        "message":
            "All notifications marked as read."
    }


# ============================================================
# CREATE
# ============================================================

@router.post(
    "",
    response_model=NotificationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create notification",
)
async def create_notification(
    payload: NotificationCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    _require_notification_admin(
        current_user
    )

    if (
        payload.expires_at
        and payload.scheduled_at
        and payload.expires_at
        <= payload.scheduled_at
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "expires_at must be after "
                "scheduled_at."
            ),
        )

    target_user_id = (
        str(payload.user_id)
        if payload.user_id
        else _user_id(
            current_user
        )
    )

    data = NotificationCreate(
        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=target_user_id,

        notification_type=(
            payload.notification_type.value
        ),

        title=payload.title.strip(),

        message=payload.message,

        priority=payload.priority.value,

        channel=payload.channel.value,

        contract_id=(
            str(payload.contract_id)
            if payload.contract_id
            else None
        ),

        document_id=(
            str(payload.document_id)
            if payload.document_id
            else None
        ),

        risk_id=(
            str(payload.risk_id)
            if payload.risk_id
            else None
        ),

        renewal_id=(
            str(payload.renewal_id)
            if payload.renewal_id
            else None
        ),

        review_id=(
            str(payload.review_id)
            if payload.review_id
            else None
        ),

        action_url=(
            payload.action_url
        ),

        scheduled_at=(
            payload.scheduled_at
        ),

        expires_at=(
            payload.expires_at
        ),

        dedup_key=(
            payload.dedup_key
        ),

        metadata={
            **payload.metadata,
            "created_by":
                _user_id(
                    current_user
                ),
            "request_id":
                _request_id(
                    request
                ),
        },
    )

    notification = (
        await repository.create(
            data
        )
    )

    return _serialize_notification(
        notification
    )


# ============================================================
# BULK CREATE
# ============================================================

@router.post(
    "/bulk",
    response_model=BulkResultResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create notifications for multiple users",
)
async def bulk_create_notifications(
    payload: BulkNotificationRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, int]:

    _require_notification_admin(
        current_user
    )

    unique_users = list(
        dict.fromkeys(
            str(user_id)
            for user_id
            in payload.user_ids
        )
    )

    notifications = [
        NotificationCreate(
            tenant_id=_tenant_id(
                current_user
            ),

            organization_id=_organization_id(
                current_user
            ),

            user_id=user_id,

            notification_type=(
                payload.notification_type.value
            ),

            title=payload.title.strip(),

            message=payload.message,

            priority=(
                payload.priority.value
            ),

            channel=(
                payload.channel.value
            ),

            action_url=(
                payload.action_url
            ),

            metadata={
                **payload.metadata,

                "created_by":
                    _user_id(
                        current_user
                    ),

                "request_id":
                    _request_id(
                        request
                    ),
            },
        )
        for user_id
        in unique_users
    ]

    created = (
        await repository.create_many(
            notifications
        )
    )

    created_count = len(
        created or []
    )

    return {
        "requested":
            len(unique_users),

        "created":
            created_count,

        "skipped":
            len(unique_users)
            - created_count,

        "failed":
            0,
    }


# ============================================================
# GET SINGLE NOTIFICATION
# ============================================================

@router.get(
    "/{notification_id}",
    response_model=NotificationResponse,
    summary="Get notification",
)
async def get_notification(
    notification_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    notification = await repository.get_by_id(
        notification_id,

        tenant_id=_tenant_id(
            current_user
        ),
    )

    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    # A normal user may only read their own notification.
    target_user = _optional_str(
        _attr(
            notification,
            "user_id",
        )
    )

    if (
        target_user
        and target_user
        != _user_id(
            current_user
        )
    ):
        try:
            _require_notification_admin(
                current_user
            )
        except HTTPException:
            # Avoid leaking whether another user's
            # notification exists.
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Notification was not found.",
            )

    return _serialize_notification(
        notification
    )


# ============================================================
# UPDATE
# ============================================================

@router.patch(
    "/{notification_id}",
    response_model=NotificationResponse,
    summary="Update notification",
)
async def update_notification(
    notification_id: UUID,
    payload: NotificationUpdateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    _require_notification_admin(
        current_user
    )

    values = payload.model_dump(
        exclude_unset=True
    )

    for key in (
        "priority",
    ):
        if key in values:
            values[key] = (
                _enum_value(
                    values[key]
                )
            )

    notification = await repository.update(
        notification_id=notification_id,

        tenant_id=_tenant_id(
            current_user
        ),

        data=NotificationUpdate(
            **values
        ),
    )

    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    return _serialize_notification(
        notification
    )


# ============================================================
# MARK READ
# ============================================================

@router.post(
    "/{notification_id}/read",
    response_model=NotificationResponse,
    summary="Mark notification as read",
)
async def mark_notification_read(
    notification_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    notification = (
        await repository.mark_read(
            notification_id=notification_id,

            tenant_id=_tenant_id(
                current_user
            ),

            user_id=_user_id(
                current_user
            ),
        )
    )

    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    return _serialize_notification(
        notification
    )


# ============================================================
# MARK UNREAD
# ============================================================

@router.post(
    "/{notification_id}/unread",
    response_model=NotificationResponse,
    summary="Mark notification as unread",
)
async def mark_notification_unread(
    notification_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    notification = (
        await repository.mark_unread(
            notification_id=notification_id,

            tenant_id=_tenant_id(
                current_user
            ),

            user_id=_user_id(
                current_user
            ),
        )
    )

    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    return _serialize_notification(
        notification
    )


# ============================================================
# SEND
# ============================================================

@router.post(
    "/{notification_id}/send",
    response_model=NotificationResponse,
    summary="Send notification",
)
async def send_notification(
    notification_id: UUID,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_notification_service
    ),
) -> dict[str, Any]:

    _require_notification_admin(
        current_user
    )

    result = await _service_call(
        service,
        "send",

        notification_id=str(
            notification_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        request_id=_request_id(
            request
        ),
    )

    return _serialize_notification(
        result
    )


# ============================================================
# RETRY FAILED DELIVERY
# ============================================================

@router.post(
    "/{notification_id}/retry",
    response_model=NotificationResponse,
    summary="Retry failed notification",
)
async def retry_notification(
    notification_id: UUID,
    payload: RetryNotificationRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_notification_service
    ),
) -> dict[str, Any]:

    _require_notification_admin(
        current_user
    )

    result = await _service_call(
        service,
        "retry",

        notification_id=str(
            notification_id
        ),

        tenant_id=_tenant_id(
            current_user
        ),

        user_id=_user_id(
            current_user
        ),

        force=payload.force,

        request_id=_request_id(
            request
        ),
    )

    return _serialize_notification(
        result
    )


# ============================================================
# CANCEL
# ============================================================

@router.post(
    "/{notification_id}/cancel",
    response_model=NotificationResponse,
    summary="Cancel notification",
)
async def cancel_notification(
    notification_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, Any]:

    _require_notification_admin(
        current_user
    )

    notification = (
        await repository.cancel(
            notification_id=notification_id,

            tenant_id=_tenant_id(
                current_user
            ),
        )
    )

    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    return _serialize_notification(
        notification
    )


# ============================================================
# DELETE
# ============================================================

@router.delete(
    "/{notification_id}",
    response_model=MessageResponse,
    summary="Delete notification",
)
async def delete_notification(
    notification_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: NotificationRepository = Depends(
        get_notification_repository
    ),
) -> dict[str, str]:

    notification = await repository.get_by_id(
        notification_id,

        tenant_id=_tenant_id(
            current_user
        ),
    )

    if notification is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    owner = _optional_str(
        _attr(
            notification,
            "user_id",
        )
    )

    is_owner = (
        owner
        == _user_id(
            current_user
        )
    )

    is_admin = bool(
        _roles(
            current_user
        )
        & {
            "admin",
            "super_admin",
            "legal_admin",
            "notification_admin",
        }
    )

    if not (
        is_owner
        or is_admin
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    deleted = await repository.delete(
        notification_id=notification_id,

        tenant_id=_tenant_id(
            current_user
        ),
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification was not found.",
        )

    return {
        "message":
            "Notification deleted successfully."
    }


# ============================================================
# HEALTH
# ============================================================

@router.get(
    "/health/service",
    summary="Notification subsystem health",
)
async def notification_health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "service": "notifications",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }