# backend/db/repositories/audit_repository.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Sequence
from uuid import UUID

from sqlalchemy import and_, delete, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.audit_log import AuditLog


class AuditAction(str, Enum):
    LOGIN = "login"
    LOGOUT = "logout"
    LOGIN_FAILED = "login_failed"

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    UPLOAD = "upload"
    DOWNLOAD = "download"

    SEARCH = "search"
    CHAT = "chat"

    DOCUMENT_CREATE = "document_create"
    DOCUMENT_VIEW = "document_view"
    DOCUMENT_UPDATE = "document_update"
    DOCUMENT_DELETE = "document_delete"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DOWNLOAD = "document_download"

    CONTRACT_CREATE = "contract_create"
    CONTRACT_UPDATE = "contract_update"
    CONTRACT_DELETE = "contract_delete"

    CLAUSE_ANALYSIS = "clause_analysis"
    RISK_ANALYSIS = "risk_analysis"

    EXPIRY_CHECK = "expiry_check"
    RENEWAL_CHECK = "renewal_check"

    REVIEW_CREATE = "review_create"
    REVIEW_APPROVE = "review_approve"
    REVIEW_REJECT = "review_reject"

    RAG_QUERY = "rag_query"
    MODEL_INFERENCE = "model_inference"
    AGENT_EXECUTION = "agent_execution"

    PERMISSION_DENIED = "permission_denied"

    SECURITY_EVENT = "security_event"
    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTED = "pii_detected"

    EXPORT = "export"

    ADMIN_ACTION = "admin_action"

    OTHER = "other"


class AuditSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AuditCreate:
    tenant_id: str
    action: str

    organization_id: Optional[str] = None
    user_id: Optional[str] = None

    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    outcome: str = AuditOutcome.SUCCESS.value
    severity: str = AuditSeverity.INFO.value

    request_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None

    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    endpoint: Optional[str] = None
    http_method: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    latency_ms: Optional[float] = None

    details: dict[str, Any] = field(default_factory=dict)

    created_at: Optional[datetime] = None


@dataclass(slots=True)
class AuditFilter:
    tenant_id: str

    organization_id: Optional[str] = None
    user_id: Optional[str] = None

    actions: list[str] = field(default_factory=list)
    severities: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)

    resource_type: Optional[str] = None
    resource_id: Optional[str] = None

    request_id: Optional[str] = None
    correlation_id: Optional[str] = None

    model_name: Optional[str] = None

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    search: Optional[str] = None


@dataclass(slots=True)
class AuditPage:
    items: list[AuditLog]

    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


@dataclass(slots=True)
class AuditStatistics:
    total_events: int = 0

    successful_events: int = 0
    failed_events: int = 0
    denied_events: int = 0

    critical_events: int = 0
    high_severity_events: int = 0

    security_events: int = 0

    unique_users: int = 0

    actions: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "successful_events": self.successful_events,
            "failed_events": self.failed_events,
            "denied_events": self.denied_events,
            "critical_events": self.critical_events,
            "high_severity_events": self.high_severity_events,
            "security_events": self.security_events,
            "unique_users": self.unique_users,
            "actions": self.actions,
        }


class AuditRepositoryError(RuntimeError):
    pass


class AuditRepository:
    """
    Persistent audit repository.

    Important security properties:

    * tenant-scoped reads
    * append-oriented audit writes
    * explicit privileged deletion
    * no raw prompt/document text required
    * pagination
    * security-event queries
    * AI/model traceability
    * request/correlation tracing

    The repository deliberately does not decide whether an action is
    authorized. Authorization must occur before calling repository
    operations.
    """

    def __init__(
        self,
        session: AsyncSession,
    ) -> None:
        if session is None:
            raise ValueError(
                "AsyncSession is required."
            )

        self.session = session

    # =========================================================
    # CREATE
    # =========================================================

    async def create(
        self,
        data: AuditCreate,
        *,
        flush: bool = True,
    ) -> AuditLog:
        """
        Append a new audit record.

        The method flushes but does not commit. Transaction ownership
        remains with the service/request transaction layer.
        """

        self._validate_create(data)

        record = AuditLog(
            tenant_id=data.tenant_id,
            organization_id=data.organization_id,
            user_id=data.user_id,
            action=data.action,
            resource_type=data.resource_type,
            resource_id=data.resource_id,
            outcome=data.outcome,
            severity=data.severity,
            request_id=data.request_id,
            session_id=data.session_id,
            correlation_id=data.correlation_id,
            ip_address=data.ip_address,
            user_agent=data.user_agent,
            endpoint=data.endpoint,
            http_method=data.http_method,
            model_name=data.model_name,
            model_version=data.model_version,
            latency_ms=data.latency_ms,
            details=self._sanitize_details(
                data.details
            ),
            created_at=(
                data.created_at
                or self._utcnow()
            ),
        )

        self.session.add(record)

        if flush:
            await self.session.flush()

        return record

    async def create_event(
        self,
        *,
        tenant_id: str,
        action: str | AuditAction,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        outcome: str | AuditOutcome = AuditOutcome.SUCCESS,
        severity: str | AuditSeverity = AuditSeverity.INFO,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> AuditLog:

        return await self.create(
            AuditCreate(
                tenant_id=tenant_id,
                organization_id=organization_id,
                user_id=user_id,
                action=self._enum_value(action),
                resource_type=resource_type,
                resource_id=resource_id,
                outcome=self._enum_value(outcome),
                severity=self._enum_value(severity),
                request_id=request_id,
                correlation_id=correlation_id,
                details=details or {},
            )
        )

    async def create_many(
        self,
        events: Sequence[AuditCreate],
    ) -> list[AuditLog]:

        records: list[AuditLog] = []

        for event in events:
            record = await self.create(
                event,
                flush=False,
            )

            records.append(record)

        if records:
            await self.session.flush()

        return records

    # =========================================================
    # GET
    # =========================================================

    async def get_by_id(
        self,
        audit_id: str | UUID,
        *,
        tenant_id: str,
    ) -> Optional[AuditLog]:

        stmt = (
            select(AuditLog)
            .where(
                AuditLog.id == audit_id,
                AuditLog.tenant_id == tenant_id,
            )
            .limit(1)
        )

        result = await self.session.execute(
            stmt
        )

        return result.scalar_one_or_none()

    async def get_by_request_id(
        self,
        request_id: str,
        *,
        tenant_id: str,
    ) -> list[AuditLog]:

        stmt = (
            select(AuditLog)
            .where(
                AuditLog.tenant_id == tenant_id,
                AuditLog.request_id == request_id,
            )
            .order_by(
                AuditLog.created_at.asc()
            )
        )

        result = await self.session.execute(
            stmt
        )

        return list(result.scalars().all())

    async def get_by_correlation_id(
        self,
        correlation_id: str,
        *,
        tenant_id: str,
    ) -> list[AuditLog]:

        stmt = (
            select(AuditLog)
            .where(
                AuditLog.tenant_id == tenant_id,
                AuditLog.correlation_id
                == correlation_id,
            )
            .order_by(
                AuditLog.created_at.asc()
            )
        )

        result = await self.session.execute(
            stmt
        )

        return list(result.scalars().all())

    # =========================================================
    # LIST / SEARCH
    # =========================================================

    async def list(
        self,
        filters: AuditFilter,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditPage:

        limit = self._validate_limit(limit)

        if offset < 0:
            raise ValueError(
                "offset cannot be negative."
            )

        conditions = self._conditions(
            filters
        )

        stmt = (
            select(AuditLog)
            .where(*conditions)
            .order_by(
                desc(AuditLog.created_at)
            )
            .limit(limit)
            .offset(offset)
        )

        count_stmt = (
            select(
                func.count(AuditLog.id)
            )
            .where(*conditions)
        )

        result = await self.session.execute(
            stmt
        )

        total_result = (
            await self.session.execute(
                count_stmt
            )
        )

        return AuditPage(
            items=list(
                result.scalars().all()
            ),
            total=int(
                total_result.scalar_one()
            ),
            limit=limit,
            offset=offset,
        )

    async def list_for_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditPage:

        return await self.list(
            AuditFilter(
                tenant_id=tenant_id,
                user_id=user_id,
            ),
            limit=limit,
            offset=offset,
        )

    async def list_for_resource(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditPage:

        return await self.list(
            AuditFilter(
                tenant_id=tenant_id,
                resource_type=resource_type,
                resource_id=resource_id,
            ),
            limit=limit,
            offset=offset,
        )

    async def list_security_events(
        self,
        *,
        tenant_id: str,
        organization_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditPage:

        return await self.list(
            AuditFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                actions=[
                    AuditAction.SECURITY_EVENT.value,
                    AuditAction.PERMISSION_DENIED.value,
                    AuditAction.PROMPT_INJECTION.value,
                    AuditAction.PII_DETECTED.value,
                ],
            ),
            limit=limit,
            offset=offset,
        )

    async def list_model_events(
        self,
        *,
        tenant_id: str,
        model_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> AuditPage:

        return await self.list(
            AuditFilter(
                tenant_id=tenant_id,
                model_name=model_name,
                actions=[
                    AuditAction.MODEL_INFERENCE.value,
                    AuditAction.RAG_QUERY.value,
                    AuditAction.AGENT_EXECUTION.value,
                ],
            ),
            limit=limit,
            offset=offset,
        )

    # =========================================================
    # STATISTICS
    # =========================================================

    async def statistics(
        self,
        filters: AuditFilter,
    ) -> AuditStatistics:

        conditions = self._conditions(
            filters
        )

        total_stmt = (
            select(
                func.count(AuditLog.id)
            )
            .where(*conditions)
        )

        total = (
            await self.session.execute(
                total_stmt
            )
        ).scalar_one()

        success = await self._count_with(
            conditions,
            AuditLog.outcome
            == AuditOutcome.SUCCESS.value,
        )

        failed = await self._count_with(
            conditions,
            AuditLog.outcome
            == AuditOutcome.FAILURE.value,
        )

        denied = await self._count_with(
            conditions,
            AuditLog.outcome
            == AuditOutcome.DENIED.value,
        )

        critical = await self._count_with(
            conditions,
            AuditLog.severity
            == AuditSeverity.CRITICAL.value,
        )

        high = await self._count_with(
            conditions,
            AuditLog.severity
            == AuditSeverity.HIGH.value,
        )

        security = await self._count_with(
            conditions,
            AuditLog.action.in_(
                [
                    AuditAction.SECURITY_EVENT.value,
                    AuditAction.PERMISSION_DENIED.value,
                    AuditAction.PROMPT_INJECTION.value,
                    AuditAction.PII_DETECTED.value,
                ]
            ),
        )

        users_stmt = (
            select(
                func.count(
                    func.distinct(
                        AuditLog.user_id
                    )
                )
            )
            .where(
                *conditions,
                AuditLog.user_id.is_not(None),
            )
        )

        unique_users = (
            await self.session.execute(
                users_stmt
            )
        ).scalar_one()

        action_stmt = (
            select(
                AuditLog.action,
                func.count(AuditLog.id),
            )
            .where(*conditions)
            .group_by(AuditLog.action)
        )

        action_result = (
            await self.session.execute(
                action_stmt
            )
        )

        actions = {
            str(action): int(count)
            for action, count
            in action_result.all()
        }

        return AuditStatistics(
            total_events=int(total),
            successful_events=success,
            failed_events=failed,
            denied_events=denied,
            critical_events=critical,
            high_severity_events=high,
            security_events=security,
            unique_users=int(
                unique_users or 0
            ),
            actions=actions,
        )

    async def _count_with(
        self,
        base_conditions: Sequence[Any],
        condition: Any,
    ) -> int:

        stmt = (
            select(
                func.count(AuditLog.id)
            )
            .where(
                *base_conditions,
                condition,
            )
        )

        result = await self.session.execute(
            stmt
        )

        return int(
            result.scalar_one()
        )

    # =========================================================
    # RETENTION
    # =========================================================

    async def delete_before(
        self,
        *,
        tenant_id: str,
        before: datetime,
        allow_delete: bool = False,
    ) -> int:
        """
        Explicit retention operation.

        Audit records should normally be immutable. The caller must
        explicitly opt into deletion, ideally from a controlled
        retention job with separate authorization.
        """

        if not allow_delete:
            raise AuditRepositoryError(
                "Audit deletion requires "
                "explicit allow_delete=True."
            )

        if before.tzinfo is None:
            before = before.replace(
                tzinfo=timezone.utc
            )

        stmt = (
            delete(AuditLog)
            .where(
                AuditLog.tenant_id
                == tenant_id,
                AuditLog.created_at
                < before,
            )
        )

        result = await self.session.execute(
            stmt
        )

        await self.session.flush()

        return int(
            result.rowcount or 0
        )

    # =========================================================
    # FILTER BUILDER
    # =========================================================

    def _conditions(
        self,
        filters: AuditFilter,
    ) -> list[Any]:

        if not filters.tenant_id:
            raise AuditRepositoryError(
                "tenant_id is mandatory "
                "for audit queries."
            )

        conditions: list[Any] = [
            AuditLog.tenant_id
            == filters.tenant_id
        ]

        if filters.organization_id:
            conditions.append(
                AuditLog.organization_id
                == filters.organization_id
            )

        if filters.user_id:
            conditions.append(
                AuditLog.user_id
                == filters.user_id
            )

        if filters.actions:
            conditions.append(
                AuditLog.action.in_(
                    filters.actions
                )
            )

        if filters.severities:
            conditions.append(
                AuditLog.severity.in_(
                    filters.severities
                )
            )

        if filters.outcomes:
            conditions.append(
                AuditLog.outcome.in_(
                    filters.outcomes
                )
            )

        if filters.resource_type:
            conditions.append(
                AuditLog.resource_type
                == filters.resource_type
            )

        if filters.resource_id:
            conditions.append(
                AuditLog.resource_id
                == filters.resource_id
            )

        if filters.request_id:
            conditions.append(
                AuditLog.request_id
                == filters.request_id
            )

        if filters.correlation_id:
            conditions.append(
                AuditLog.correlation_id
                == filters.correlation_id
            )

        if filters.model_name:
            conditions.append(
                AuditLog.model_name
                == filters.model_name
            )

        if filters.start_time:
            conditions.append(
                AuditLog.created_at
                >= filters.start_time
            )

        if filters.end_time:
            conditions.append(
                AuditLog.created_at
                <= filters.end_time
            )

        if filters.search:
            term = (
                "%"
                + self._escape_like(
                    filters.search.strip()
                )
                + "%"
            )

            conditions.append(
                or_(
                    AuditLog.action.ilike(
                        term,
                        escape="\\",
                    ),
                    AuditLog.resource_type.ilike(
                        term,
                        escape="\\",
                    ),
                    AuditLog.resource_id.ilike(
                        term,
                        escape="\\",
                    ),
                    AuditLog.request_id.ilike(
                        term,
                        escape="\\",
                    ),
                    AuditLog.correlation_id.ilike(
                        term,
                        escape="\\",
                    ),
                )
            )

        return conditions

    # =========================================================
    # PRIVACY
    # =========================================================

    @classmethod
    def _sanitize_details(
        cls,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Prevent common secrets from being accidentally written to
        the audit JSON field.

        This is defense-in-depth. Services should still avoid passing
        sensitive raw document text, prompts, passwords or tokens.
        """

        if not details:
            return {}

        blocked = {
            "password",
            "password_hash",
            "access_token",
            "refresh_token",
            "authorization",
            "cookie",
            "api_key",
            "secret",
            "client_secret",
        }

        output: dict[str, Any] = {}

        for key, value in details.items():
            normalized = str(
                key
            ).strip().lower()

            if normalized in blocked:
                output[key] = "[REDACTED]"
                continue

            if isinstance(value, dict):
                output[key] = (
                    cls._sanitize_details(
                        value
                    )
                )

            elif isinstance(value, list):
                output[key] = [
                    cls._sanitize_details(
                        item
                    )
                    if isinstance(
                        item,
                        dict,
                    )
                    else item
                    for item in value
                ]

            else:
                output[key] = value

        return output

    # =========================================================
    # VALIDATION / UTILITIES
    # =========================================================

    @staticmethod
    def _validate_create(
        data: AuditCreate,
    ) -> None:

        if not data.tenant_id:
            raise AuditRepositoryError(
                "tenant_id is required."
            )

        if not data.action:
            raise AuditRepositoryError(
                "audit action is required."
            )

    @staticmethod
    def _validate_limit(
        limit: int,
    ) -> int:

        if limit <= 0:
            raise ValueError(
                "limit must be > 0."
            )

        return min(
            limit,
            1000,
        )

    @staticmethod
    def _enum_value(
        value: Any,
    ) -> str:

        if isinstance(value, Enum):
            return str(value.value)

        return str(value)

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(
            timezone.utc
        )

    @staticmethod
    def _escape_like(
        value: str,
    ) -> str:

        return (
            value
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )