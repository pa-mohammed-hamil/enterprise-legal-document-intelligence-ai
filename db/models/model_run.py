# backend/db/models/model_run.py

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.db.base import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelRun(Base):
    """
    Immutable-ish provenance record for an AI/model execution.

    Examples:
        LLM generation
        embedding generation
        reranking
        document classification
        clause classification
        legal NER
        risk classification
        OCR / VLM inference
        query classification
        RAG answer generation
        agent model call
        evaluation model call

    ModelRun should contain metadata/provenance rather than large
    document contents or unrestricted prompts.

    Sensitive prompt/output content should live in an appropriately
    protected trace store when retention is required.
    """

    __tablename__ = "model_runs"

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

    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    # ========================================================
    # RUN IDENTITY
    # ========================================================

    run_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
    )

    run_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    task_type: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    operation: Mapped[Optional[str]] = mapped_column(
        String(150),
        nullable=True,
        index=True,
    )

    source: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # ========================================================
    # MODEL
    # ========================================================

    model_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    model_version: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    model_provider: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    model_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    model_revision: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    model_endpoint: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    deployment_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    # ========================================================
    # ROUTING
    # ========================================================

    router_mode: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    route_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    fallback_used: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    fallback_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    fallback_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
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

    agent_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    parent_run_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # RESOURCE PROVENANCE
    # ========================================================

    document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
    )

    document_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
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

    chunk_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
    )

    # ========================================================
    # STATUS
    # ========================================================

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="pending",
        index=True,
    )

    success: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        index=True,
    )

    cancelled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    timed_out: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # INPUT
    # ========================================================

    input_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    input_characters: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    input_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    prompt_template: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    prompt_template_version: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    # Store only sanitized/redacted metadata here.
    input_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    output_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    output_characters: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    output_hash: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    finish_reason: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    output_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # GENERATION PARAMETERS
    # ========================================================

    temperature: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    top_p: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    top_k: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    max_tokens: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    seed: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    generation_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    # ========================================================
    # TOKENS
    # ========================================================

    prompt_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    completion_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    total_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    cached_input_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    reasoning_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # PERFORMANCE
    # ========================================================

    latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
    )

    queue_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    inference_latency_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    time_to_first_token_ms: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    tokens_per_second: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================
    # COST
    # ========================================================

    input_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 8),
        nullable=True,
    )

    output_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 8),
        nullable=True,
    )

    total_cost: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(18, 8),
        nullable=True,
        index=True,
    )

    cost_currency: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="USD",
    )

    # ========================================================
    # RAG PROVENANCE
    # ========================================================

    rag_used: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    retrieval_strategy: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    query_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    rewritten_query_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    retrieved_document_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    retrieved_chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    reranked_chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    retrieved_chunk_ids: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    retrieval_scores: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    reranker_scores: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    # ========================================================
    # CITATIONS
    # ========================================================

    citation_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    citation_valid: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
    )

    citation_accuracy: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluated: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    groundedness_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    faithfulness_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    relevance_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    answer_quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    hallucination_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    evaluation_results: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    evaluator_model: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    evaluated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================
    # TOOL / AGENT DATA
    # ========================================================

    tool_call_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    tool_calls: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    agent_steps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ========================================================
    # SECURITY / GUARDRAILS
    # ========================================================

    guardrails_applied: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    guardrail_triggered: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
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

    content_filtered: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    blocked: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    security_findings: Mapped[list[Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
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

    # Reference into protected logs/tracing rather than storing
    # raw stack traces that may contain document text/secrets.
    error_reference: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
    )

    # ========================================================
    # WORKER / INFRASTRUCTURE
    # ========================================================

    worker_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    host_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
    )

    device: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    device_id: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    batch_size: Mapped[Optional[int]] = mapped_column(
        Integer,
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

    queued_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utcnow,
        index=True,
    )

    # ========================================================
    # INDEXES
    # ========================================================

    __table_args__ = (
        Index(
            "ix_model_runs_tenant_created",
            "tenant_id",
            "created_at",
        ),
        Index(
            "ix_model_runs_tenant_model",
            "tenant_id",
            "model_name",
            "created_at",
        ),
        Index(
            "ix_model_runs_tenant_type",
            "tenant_id",
            "run_type",
            "created_at",
        ),
        Index(
            "ix_model_runs_tenant_status",
            "tenant_id",
            "status",
            "created_at",
        ),
        Index(
            "ix_model_runs_tenant_user",
            "tenant_id",
            "user_id",
            "created_at",
        ),
        Index(
            "ix_model_runs_request",
            "tenant_id",
            "request_id",
        ),
        Index(
            "ix_model_runs_trace",
            "tenant_id",
            "trace_id",
        ),
        Index(
            "ix_model_runs_document",
            "tenant_id",
            "document_id",
            "document_version_id",
        ),
        Index(
            "ix_model_runs_contract",
            "tenant_id",
            "contract_id",
        ),
        Index(
            "ix_model_runs_clause",
            "tenant_id",
            "clause_id",
        ),
        Index(
            "ix_model_runs_agent",
            "tenant_id",
            "agent_run_id",
        ),
        Index(
            "ix_model_runs_rag",
            "tenant_id",
            "rag_used",
            "retrieval_strategy",
        ),
        Index(
            "ix_model_runs_security",
            "tenant_id",
            "blocked",
            "guardrail_triggered",
            "prompt_injection_detected",
        ),
    )

    # ========================================================
    # DOMAIN HELPERS
    # ========================================================

    @property
    def duration_ms(self) -> Optional[float]:
        if self.latency_ms is not None:
            return self.latency_ms

        if (
            self.started_at is not None
            and self.completed_at is not None
        ):
            return (
                self.completed_at
                - self.started_at
            ).total_seconds() * 1000.0

        return None

    @property
    def failed(self) -> bool:
        return bool(
            self.success is False
            or self.status.lower()
            in {
                "failed",
                "error",
            }
        )

    @property
    def security_event(self) -> bool:
        return bool(
            self.blocked
            or self.guardrail_triggered
            or self.prompt_injection_detected
            or self.pii_detected
            or self.content_filtered
        )

    @property
    def evaluation_summary(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "groundedness":
                self.groundedness_score,
            "faithfulness":
                self.faithfulness_score,
            "relevance":
                self.relevance_score,
            "answer_quality":
                self.answer_quality_score,
            "hallucination":
                self.hallucination_score,
            "citation_accuracy":
                self.citation_accuracy,
        }

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def mark_started(self) -> None:
        self.status = "running"
        self.started_at = utcnow()
        self.success = None

    def mark_completed(
        self,
        *,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        self.status = "completed"
        self.success = True
        self.completed_at = utcnow()

        if prompt_tokens is not None:
            self.prompt_tokens = max(
                0,
                prompt_tokens,
            )

        if completion_tokens is not None:
            self.completion_tokens = max(
                0,
                completion_tokens,
            )

        if total_tokens is not None:
            self.total_tokens = max(
                0,
                total_tokens,
            )
        else:
            self.total_tokens = (
                self.prompt_tokens
                + self.completion_tokens
            )

        if finish_reason is not None:
            self.finish_reason = finish_reason

        self._calculate_performance()

    def mark_failed(
        self,
        *,
        error_code: Optional[str] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_reference: Optional[str] = None,
    ) -> None:
        self.status = "failed"
        self.success = False
        self.completed_at = utcnow()

        self.error_code = error_code
        self.error_type = error_type
        self.error_message = error_message
        self.error_reference = error_reference

        self._calculate_performance()

    def mark_timeout(self) -> None:
        self.status = "failed"
        self.success = False
        self.timed_out = True
        self.completed_at = utcnow()

        self._calculate_performance()

    def mark_cancelled(self) -> None:
        self.status = "cancelled"
        self.success = False
        self.cancelled = True
        self.completed_at = utcnow()

        self._calculate_performance()

    def _calculate_performance(self) -> None:
        if (
            self.started_at is not None
            and self.completed_at is not None
        ):
            self.latency_ms = max(
                0.0,
                (
                    self.completed_at
                    - self.started_at
                ).total_seconds()
                * 1000.0,
            )

        if (
            self.completion_tokens > 0
            and self.inference_latency_ms
            and self.inference_latency_ms > 0
        ):
            self.tokens_per_second = (
                self.completion_tokens
                / (
                    self.inference_latency_ms
                    / 1000.0
                )
            )

    # ========================================================
    # RAG
    # ========================================================

    def set_retrieval_result(
        self,
        *,
        strategy: str,
        document_count: int,
        chunk_ids: list[str],
        retrieval_scores: Optional[
            list[Any]
        ] = None,
        reranker_scores: Optional[
            list[Any]
        ] = None,
    ) -> None:
        self.rag_used = True
        self.retrieval_strategy = strategy

        self.retrieved_document_count = max(
            0,
            document_count,
        )

        self.retrieved_chunk_count = len(
            chunk_ids
        )

        self.retrieved_chunk_ids = (
            chunk_ids
        )

        self.retrieval_scores = (
            retrieval_scores or []
        )

        self.reranker_scores = (
            reranker_scores or []
        )

        self.reranked_chunk_count = len(
            self.reranker_scores
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    def set_evaluation(
        self,
        *,
        groundedness: Optional[
            float
        ] = None,
        faithfulness: Optional[
            float
        ] = None,
        relevance: Optional[
            float
        ] = None,
        answer_quality: Optional[
            float
        ] = None,
        hallucination: Optional[
            float
        ] = None,
        citation_accuracy: Optional[
            float
        ] = None,
        evaluator_model: Optional[
            str
        ] = None,
        results: Optional[
            dict[str, Any]
        ] = None,
    ) -> None:
        self.groundedness_score = (
            groundedness
        )

        self.faithfulness_score = (
            faithfulness
        )

        self.relevance_score = relevance

        self.answer_quality_score = (
            answer_quality
        )

        self.hallucination_score = (
            hallucination
        )

        self.citation_accuracy = (
            citation_accuracy
        )

        self.evaluator_model = (
            evaluator_model
        )

        self.evaluation_results = (
            results or {}
        )

        self.evaluated = True
        self.evaluated_at = utcnow()

    # ========================================================
    # REPRESENTATION
    # ========================================================

    def __repr__(self) -> str:
        return (
            "<ModelRun("
            f"id={self.id}, "
            f"run_id={self.run_id!r}, "
            f"model={self.model_name!r}, "
            f"type={self.run_type!r}, "
            f"status={self.status!r}"
            ")>"
        )