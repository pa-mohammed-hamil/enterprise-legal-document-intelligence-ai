    # backend/db/models/audit_log.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuditLog(Base):
    """
    Immutable audit event.

    Audit records should normally be INSERT ONLY.

    Do not use this model as an application log. It is intended
    for security, compliance and business-action provenance.

    Typical events:

        authentication
        authorization
        document access
        contract changes
        legal review decisions
        risk decisions
        RAG queries
        model inference
        agent execution
        prompt injection detection
        PII detection
        exports
        administrative actions
    """

    __tablename__ = "audit_logs"

    # ========================================================
    # IDENTITY
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
    # ACTOR
    # ========================================================

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    actor_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="user",
    )

    actor_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    actor_email: Mapped[Optional[str]] = mapped_column(
        String(320),
        nullable=True,
    )

    actor_role: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    service_name: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    agent_name: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
    )

    # ========================================================
    # EVENT
    # ========================================================

    action: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    event_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    severity: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="info",
        index=True,
    )

    outcome: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="success",
        index=True,
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # RESOURCE
    # ========================================================

    resource_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    resource_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    resource_name: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    contract_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    clause_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    risk_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    review_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    renewal_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # REQUEST CORRELATION
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

    span_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    session_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # HTTP / NETWORK
    # ========================================================

    http_method: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )

    endpoint: Mapped[Optional[str]] = mapped_column(
        String(1000),
        nullable=True,
    )

    status_code: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    ip_address: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    user_agent: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================
    # CHANGE TRACKING
    # ========================================================

    previous_values: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )

    new_values: Mapped[Optional[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )

    changed_fields: Mapped[Optional[list[str]]] = mapped_column(
        JSONB,
        nullable=True,
    )

    # ========================================================
    # RAG / RETRIEVAL
    # ========================================================

    query_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    query_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    retrieval_strategy: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    retrieved_document_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    retrieved_chunk_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    retrieval_scores: Mapped[Optional[list[Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )

    reranker_scores: Mapped[Optional[list[Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )

    citation_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    citation_valid: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    groundedness_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================
    # MODEL / LLM
    # ========================================================

    model_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_provider: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    model_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    prompt_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    completion_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    total_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    estimated_cost: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================
    # AGENT EXECUTION
    # ========================================================

    agent_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    agent_steps: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    tool_calls: Mapped[Optional[list[Any]]] = mapped_column(
        JSONB,
        nullable=True,
    )

    tool_call_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================
    # SECURITY
    # ========================================================

    security_event: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    permission_granted: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    prompt_injection_detected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    pii_detected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    malware_detected: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    guardrail_triggered: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    guardrail_result: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    blocked: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
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

    # Never store raw stack traces if they can contain
    # secrets, prompts, document text, or personal data.
    error_reference: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    # ========================================================
    # AUDIT INTEGRITY
    # ========================================================

    schema_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    event_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    previous_event_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
    )

    immutable: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    # Generic extension point.
    # Avoid SQLAlchemy's reserved declarative attribute name
    # "metadata".
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
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

    # Audit events are immutable, so intentionally no
    # updated_at / deleted_at columns.

    # ========================================================
    # INDEXES
    # ========================================================

    __table_args__ = (
        Index(
            "ix_audit_logs_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_audit_logs_tenant_action",
            "tenant_id",
            "action",
        ),
        Index(
            "ix_audit_logs_tenant_user_created",
            "tenant_id",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_audit_logs_tenant_resource",
            "tenant_id",
            "resource_type",
            "resource_id",
        ),
        Index(
            "ix_audit_logs_tenant_severity",
            "tenant_id",
            "severity",
            "created_at",
        ),
        Index(
            "ix_audit_logs_security_events",
            "tenant_id",
            "security_event",
            "created_at",
        ),
        Index(
            "ix_audit_logs_request",
            "tenant_id",
            "request_id",
        ),
        Index(
            "ix_audit_logs_correlation",
            "tenant_id",
            "correlation_id",
        ),
        Index(
            "ix_audit_logs_model_run",
            "tenant_id",
            "model_run_id",
        ),
        Index(
            "ix_audit_logs_agent_run",
            "tenant_id",
            "agent_run_id",
        ),
    )

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            "<AuditLog("
            f"id={self.id}, "
            f"tenant_id={self.tenant_id}, "
            f"action={self.action!r}, "
            f"severity={self.severity!r}, "
            f"outcome={self.outcome!r}"
            ")>"
        )

    # ========================================================
    # HELPERS
    # ========================================================

    @property
    def is_security_event(self) -> bool:
        return bool(
            self.security_event
            or self.prompt_injection_detected
            or self.pii_detected
            or self.malware_detected
            or self.guardrail_triggered
            or self.blocked
            or self.permission_granted is False
        )

    @property
    def is_ai_event(self) -> bool:
        return bool(
            self.model_name
            or self.model_run_id
            or self.agent_run_id
            or self.query_type
            or self.retrieval_strategy
        )

    @property
    def token_count(self) -> int:
        if self.total_tokens is not None:
            return self.total_tokens

        return (
            (self.prompt_tokens or 0)
            + (self.completion_tokens or 0)
        )

    def resource_reference(
        self,
    ) -> Optional[str]:
        if (
            self.resource_type
            and self.resource_id
        ):
            return (
                f"{self.resource_type}:"
                f"{self.resource_id}"
            )

        for name, value in (
            ("document", self.document_id),
            ("contract", self.contract_id),
            ("clause", self.clause_id),
            ("risk", self.risk_id),
            ("review", self.review_id),
            ("renewal", self.renewal_id),
        ):
            if value:
                return f"{name}:{value}"

        return None