# backend/app/schemas/user.py

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class UserStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    DEACTIVATED = "deactivated"


class UserRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    LEGAL_ADMIN = "legal_admin"
    LEGAL_REVIEWER = "legal_reviewer"
    LEGAL_ANALYST = "legal_analyst"
    MEMBER = "member"
    VIEWER = "viewer"


class UserBase(BaseModel):
    email: EmailStr

    first_name: str | None = Field(
        default=None,
        max_length=100,
    )

    last_name: str | None = Field(
        default=None,
        max_length=100,
    )

    display_name: str | None = Field(
        default=None,
        max_length=255,
    )

    job_title: str | None = Field(
        default=None,
        max_length=255,
    )

    department: str | None = Field(
        default=None,
        max_length=255,
    )

    phone: str | None = Field(
        default=None,
        max_length=50,
    )

    timezone: str = Field(
        default="UTC",
        max_length=100,
    )

    locale: str = Field(
        default="en",
        max_length=20,
    )


class UserCreate(UserBase):
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )

    organization_id: UUID | None = None

    role: UserRole = UserRole.MEMBER

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not any(char.isupper() for char in value):
            raise ValueError(
                "password must contain an uppercase letter"
            )

        if not any(char.islower() for char in value):
            raise ValueError(
                "password must contain a lowercase letter"
            )

        if not any(char.isdigit() for char in value):
            raise ValueError(
                "password must contain a number"
            )

        return value


class UserRegister(BaseModel):
    email: EmailStr

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )

    first_name: str | None = Field(
        default=None,
        max_length=100,
    )

    last_name: str | None = Field(
        default=None,
        max_length=100,
    )

    organization_name: str | None = Field(
        default=None,
        max_length=255,
    )

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if not any(char.isupper() for char in value):
            raise ValueError(
                "password must contain an uppercase letter"
            )

        if not any(char.islower() for char in value):
            raise ValueError(
                "password must contain a lowercase letter"
            )

        if not any(char.isdigit() for char in value):
            raise ValueError(
                "password must contain a number"
            )

        return value


class UserUpdate(BaseModel):
    first_name: str | None = Field(
        default=None,
        max_length=100,
    )

    last_name: str | None = Field(
        default=None,
        max_length=100,
    )

    display_name: str | None = Field(
        default=None,
        max_length=255,
    )

    job_title: str | None = Field(
        default=None,
        max_length=255,
    )

    department: str | None = Field(
        default=None,
        max_length=255,
    )

    phone: str | None = Field(
        default=None,
        max_length=50,
    )

    timezone: str | None = Field(
        default=None,
        max_length=100,
    )

    locale: str | None = Field(
        default=None,
        max_length=20,
    )

    avatar_url: str | None = Field(
        default=None,
        max_length=2000,
    )

    metadata: dict[str, Any] | None = None


class UserAdminUpdate(BaseModel):
    role: UserRole | None = None
    status: UserStatus | None = None

    is_active: bool | None = None
    is_verified: bool | None = None

    job_title: str | None = Field(
        default=None,
        max_length=255,
    )

    department: str | None = Field(
        default=None,
        max_length=255,
    )


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    organization_id: UUID | None = None

    role: UserRole

    status: UserStatus

    avatar_url: str | None = None

    is_active: bool = True
    is_verified: bool = False

    email_verified_at: datetime | None = None

    last_login_at: datetime | None = None

    created_at: datetime
    updated_at: datetime


class UserDetailResponse(UserResponse):
    permissions: list[str] = Field(
        default_factory=list
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    document_count: int = 0
    contract_count: int = 0
    review_count: int = 0

    pending_reviews: int = 0
    assigned_obligations: int = 0
    unread_notifications: int = 0


class UserSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID

    email: EmailStr

    first_name: str | None = None
    last_name: str | None = None

    display_name: str | None = None

    role: UserRole

    status: UserStatus

    avatar_url: str | None = None


class UserProfileResponse(UserResponse):
    permissions: list[str] = Field(
        default_factory=list
    )

    organization_name: str | None = None
    organization_role: str | None = None


class UserProfileUpdate(BaseModel):
    first_name: str | None = Field(
        default=None,
        max_length=100,
    )

    last_name: str | None = Field(
        default=None,
        max_length=100,
    )

    display_name: str | None = Field(
        default=None,
        max_length=255,
    )

    job_title: str | None = Field(
        default=None,
        max_length=255,
    )

    department: str | None = Field(
        default=None,
        max_length=255,
    )

    phone: str | None = Field(
        default=None,
        max_length=50,
    )

    timezone: str | None = None
    locale: str | None = None

    avatar_url: str | None = Field(
        default=None,
        max_length=2000,
    )


class PasswordChangeRequest(BaseModel):
    current_password: str

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, value: str) -> str:
        if not any(char.isupper() for char in value):
            raise ValueError(
                "new password must contain an uppercase letter"
            )

        if not any(char.islower() for char in value):
            raise ValueError(
                "new password must contain a lowercase letter"
            )

        if not any(char.isdigit() for char in value):
            raise ValueError(
                "new password must contain a number"
            )

        return value


class PasswordChangeResponse(BaseModel):
    success: bool = True
    message: str = "Password changed successfully"


class PasswordResetRequest(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str = Field(
        ...,
        min_length=1,
    )

    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )


class EmailVerificationRequest(BaseModel):
    token: str = Field(
        ...,
        min_length=1,
    )


class UserRoleUpdate(BaseModel):
    role: UserRole


class UserStatusUpdate(BaseModel):
    status: UserStatus

    reason: str | None = Field(
        default=None,
        max_length=2000,
    )


class UserPermissionsResponse(BaseModel):
    user_id: UUID

    role: UserRole

    permissions: list[str] = Field(
        default_factory=list
    )


class UserActivityStatistics(BaseModel):
    documents_uploaded: int = 0
    contracts_created: int = 0

    reviews_completed: int = 0
    reviews_pending: int = 0

    obligations_completed: int = 0
    obligations_pending: int = 0

    risks_reviewed: int = 0

    searches_performed: int = 0
    chat_queries: int = 0

    last_activity_at: datetime | None = None


class UserDetailWithActivity(UserDetailResponse):
    activity: UserActivityStatistics = Field(
        default_factory=UserActivityStatistics
    )


class UserFilter(BaseModel):
    organization_id: UUID | None = None

    roles: list[UserRole] | None = None
    statuses: list[UserStatus] | None = None

    department: str | None = None

    is_active: bool | None = None
    is_verified: bool | None = None

    search: str | None = Field(
        default=None,
        max_length=500,
    )


class UserListResponse(BaseModel):
    items: list[UserResponse] = Field(
        default_factory=list
    )

    total: int = Field(
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


class UserSessionResponse(BaseModel):
    id: UUID

    user_id: UUID

    ip_address: str | None = None
    user_agent: str | None = None

    created_at: datetime
    last_activity_at: datetime | None = None

    expires_at: datetime | None = None

    is_current: bool = False


class UserSessionListResponse(BaseModel):
    sessions: list[UserSessionResponse] = Field(
        default_factory=list
    )


class UserDeactivateRequest(BaseModel):
    reason: str | None = Field(
        default=None,
        max_length=5000,
    )


class UserDeactivateResponse(BaseModel):
    user_id: UUID

    status: UserStatus = UserStatus.DEACTIVATED

    deactivated: bool = True

    deactivated_at: datetime


class UserBulkRoleUpdateRequest(BaseModel):
    user_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    role: UserRole


class UserBulkStatusUpdateRequest(BaseModel):
    user_ids: list[UUID] = Field(
        ...,
        min_length=1,
    )

    status: UserStatus


class UserBulkResult(BaseModel):
    user_id: UUID

    success: bool

    error: str | None = None


class UserBulkResponse(BaseModel):
    requested: int = 0
    successful: int = 0
    failed: int = 0

    results: list[UserBulkResult] = Field(
        default_factory=list
    )