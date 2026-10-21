# backend/services/analytics_service.py

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence


# ============================================================
# ENUMS
# ============================================================

class AnalyticsPeriod(str, Enum):
    TODAY = "today"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    THIS_MONTH = "this_month"
    THIS_QUARTER = "this_quarter"
    THIS_YEAR = "this_year"
    CUSTOM = "custom"


class TrendDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


# ============================================================
# REPOSITORY PROTOCOLS
# ============================================================

class CountRepository(Protocol):
    async def count(self, filters: Any) -> int:
        ...


class StatisticsRepository(Protocol):
    async def statistics(self, filters: Any) -> Any:
        ...


# ============================================================
# REQUEST / FILTERS
# ============================================================

@dataclass(slots=True)
class AnalyticsContext:
    tenant_id: str

    organization_id: Optional[str] = None
    user_id: Optional[str] = None

    period: AnalyticsPeriod = (
        AnalyticsPeriod.LAST_30_DAYS
    )

    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None

    timezone_name: str = "UTC"

    include_trends: bool = True
    include_ai_metrics: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class DateRange:
    start_at: datetime
    end_at: datetime


# ============================================================
# COMMON METRICS
# ============================================================

@dataclass(slots=True)
class TrendMetric:
    current: float
    previous: float

    absolute_change: float
    percentage_change: Optional[float]

    direction: TrendDirection

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "previous": self.previous,
            "absolute_change":
                self.absolute_change,
            "percentage_change":
                self.percentage_change,
            "direction":
                self.direction.value,
        }


@dataclass(slots=True)
class DistributionItem:
    name: str
    count: int
    percentage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TimeSeriesPoint:
    timestamp: datetime
    value: float

    label: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp":
                self.timestamp.isoformat(),
            "value": self.value,
            "label": self.label,
        }


# ============================================================
# DOCUMENT ANALYTICS
# ============================================================

@dataclass(slots=True)
class DocumentAnalytics:
    total_documents: int = 0

    processed_documents: int = 0
    failed_documents: int = 0
    processing_documents: int = 0

    indexed_documents: int = 0
    pending_indexing: int = 0

    ocr_documents: int = 0

    total_pages: int = 0
    total_chunks: int = 0

    by_type: dict[str, int] = field(
        default_factory=dict
    )

    by_status: dict[str, int] = field(
        default_factory=dict
    )

    by_language: dict[str, int] = field(
        default_factory=dict
    )

    upload_trend: Optional[
        TrendMetric
    ] = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        if self.upload_trend:
            result["upload_trend"] = (
                self.upload_trend.to_dict()
            )

        return result


# ============================================================
# CONTRACT ANALYTICS
# ============================================================

@dataclass(slots=True)
class ContractAnalytics:
    total_contracts: int = 0

    active_contracts: int = 0
    draft_contracts: int = 0
    expired_contracts: int = 0
    terminated_contracts: int = 0

    contracts_expiring_30_days: int = 0
    contracts_expiring_60_days: int = 0
    contracts_expiring_90_days: int = 0

    auto_renewing_contracts: int = 0

    total_contract_value: float = 0.0

    by_type: dict[str, int] = field(
        default_factory=dict
    )

    by_status: dict[str, int] = field(
        default_factory=dict
    )

    by_risk_level: dict[str, int] = field(
        default_factory=dict
    )

    by_currency: dict[str, float] = field(
        default_factory=dict
    )

    contract_trend: Optional[
        TrendMetric
    ] = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        if self.contract_trend:
            result["contract_trend"] = (
                self.contract_trend.to_dict()
            )

        return result


# ============================================================
# CLAUSE ANALYTICS
# ============================================================

@dataclass(slots=True)
class ClauseAnalytics:
    total_clauses: int = 0

    high_risk_clauses: int = 0
    critical_clauses: int = 0

    average_confidence: Optional[
        float
    ] = None

    average_risk_score: Optional[
        float
    ] = None

    by_type: dict[str, int] = field(
        default_factory=dict
    )

    by_risk_level: dict[str, int] = field(
        default_factory=dict
    )

    top_clause_types: list[
        DistributionItem
    ] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["top_clause_types"] = [
            item.to_dict()
            for item
            in self.top_clause_types
        ]

        return result


# ============================================================
# RISK ANALYTICS
# ============================================================

@dataclass(slots=True)
class RiskAnalytics:
    total_risks: int = 0

    open_risks: int = 0
    acknowledged_risks: int = 0
    mitigated_risks: int = 0
    resolved_risks: int = 0

    low_risks: int = 0
    medium_risks: int = 0
    high_risks: int = 0
    critical_risks: int = 0

    pending_review: int = 0
    escalated_risks: int = 0

    average_score: Optional[float] = None
    average_confidence: Optional[
        float
    ] = None

    by_category: dict[str, int] = field(
        default_factory=dict
    )

    by_level: dict[str, int] = field(
        default_factory=dict
    )

    by_status: dict[str, int] = field(
        default_factory=dict
    )

    risk_trend: Optional[
        TrendMetric
    ] = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        if self.risk_trend:
            result["risk_trend"] = (
                self.risk_trend.to_dict()
            )

        return result


# ============================================================
# REVIEW ANALYTICS
# ============================================================

@dataclass(slots=True)
class ReviewAnalytics:
    total_reviews: int = 0

    pending_reviews: int = 0
    assigned_reviews: int = 0
    in_review: int = 0
    completed_reviews: int = 0

    approved_reviews: int = 0
    rejected_reviews: int = 0
    changes_requested: int = 0

    overdue_reviews: int = 0
    unassigned_reviews: int = 0

    high_priority_reviews: int = 0
    critical_priority_reviews: int = 0

    by_type: dict[str, int] = field(
        default_factory=dict
    )

    by_status: dict[str, int] = field(
        default_factory=dict
    )

    by_decision: dict[str, int] = field(
        default_factory=dict
    )

    by_priority: dict[str, int] = field(
        default_factory=dict
    )

    completion_rate: float = 0.0
    approval_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# RENEWAL ANALYTICS
# ============================================================

@dataclass(slots=True)
class RenewalAnalytics:
    total_renewals: int = 0

    upcoming_renewals: int = 0
    overdue_renewals: int = 0

    completed_renewals: int = 0
    cancelled_renewals: int = 0

    renewals_30_days: int = 0
    renewals_60_days: int = 0
    renewals_90_days: int = 0

    notice_deadlines_30_days: int = 0

    auto_renewals: int = 0

    by_status: dict[str, int] = field(
        default_factory=dict
    )

    by_priority: dict[str, int] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# AI / RAG ANALYTICS
# ============================================================

@dataclass(slots=True)
class AIAnalytics:
    total_queries: int = 0

    rag_queries: int = 0
    agent_runs: int = 0
    model_runs: int = 0

    successful_runs: int = 0
    failed_runs: int = 0

    average_latency_ms: Optional[
        float
    ] = None

    p95_latency_ms: Optional[
        float
    ] = None

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0

    estimated_cost: float = 0.0

    average_retrieval_score: Optional[
        float
    ] = None

    average_reranker_score: Optional[
        float
    ] = None

    groundedness_score: Optional[
        float
    ] = None

    citation_accuracy: Optional[
        float
    ] = None

    hallucination_rate: Optional[
        float
    ] = None

    answer_validation_rate: Optional[
        float
    ] = None

    by_model: dict[str, int] = field(
        default_factory=dict
    )

    by_query_type: dict[str, int] = field(
        default_factory=dict
    )

    by_agent: dict[str, int] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# SECURITY ANALYTICS
# ============================================================

@dataclass(slots=True)
class SecurityAnalytics:
    security_events: int = 0

    permission_denials: int = 0

    prompt_injection_events: int = 0
    pii_events: int = 0

    blocked_requests: int = 0

    critical_events: int = 0

    by_event_type: dict[str, int] = field(
        default_factory=dict
    )

    by_severity: dict[str, int] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# DASHBOARD
# ============================================================

@dataclass(slots=True)
class DashboardAnalytics:
    generated_at: datetime

    tenant_id: str
    organization_id: Optional[str]

    period: AnalyticsPeriod
    start_at: datetime
    end_at: datetime

    documents: DocumentAnalytics
    contracts: ContractAnalytics
    clauses: ClauseAnalytics
    risks: RiskAnalytics
    reviews: ReviewAnalytics
    renewals: RenewalAnalytics

    ai: Optional[AIAnalytics] = None
    security: Optional[
        SecurityAnalytics
    ] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at":
                self.generated_at.isoformat(),

            "tenant_id":
                self.tenant_id,

            "organization_id":
                self.organization_id,

            "period":
                self.period.value,

            "start_at":
                self.start_at.isoformat(),

            "end_at":
                self.end_at.isoformat(),

            "documents":
                self.documents.to_dict(),

            "contracts":
                self.contracts.to_dict(),

            "clauses":
                self.clauses.to_dict(),

            "risks":
                self.risks.to_dict(),

            "reviews":
                self.reviews.to_dict(),

            "renewals":
                self.renewals.to_dict(),

            "ai":
                self.ai.to_dict()
                if self.ai
                else None,

            "security":
                self.security.to_dict()
                if self.security
                else None,
        }


# ============================================================
# ANALYTICS SERVICE
# ============================================================

class AnalyticsService:
    """
    Aggregates enterprise legal analytics.

    This service intentionally accepts repositories through
    dependency injection so analytics remains independent from
    concrete SQLAlchemy implementations.

    Expected repositories
    ---------------------
    document_repository
    contract_repository
    clause_repository
    risk_repository
    review_repository
    renewal_repository

    Optional:
    audit_repository
    model_run_repository
    telemetry_repository
    """

    def __init__(
        self,
        *,
        document_repository: Any = None,
        contract_repository: Any = None,
        clause_repository: Any = None,
        risk_repository: Any = None,
        review_repository: Any = None,
        renewal_repository: Any = None,
        audit_repository: Any = None,
        model_run_repository: Any = None,
        telemetry_repository: Any = None,
    ) -> None:

        self.document_repository = (
            document_repository
        )

        self.contract_repository = (
            contract_repository
        )

        self.clause_repository = (
            clause_repository
        )

        self.risk_repository = (
            risk_repository
        )

        self.review_repository = (
            review_repository
        )

        self.renewal_repository = (
            renewal_repository
        )

        self.audit_repository = (
            audit_repository
        )

        self.model_run_repository = (
            model_run_repository
        )

        self.telemetry_repository = (
            telemetry_repository
        )

    # ========================================================
    # DASHBOARD
    # ========================================================

    async def dashboard(
        self,
        context: AnalyticsContext,
    ) -> DashboardAnalytics:

        self._validate_context(
            context
        )

        date_range = (
            self.resolve_date_range(
                context
            )
        )

        documents = (
            await self.document_analytics(
                context,
                date_range,
            )
        )

        contracts = (
            await self.contract_analytics(
                context,
                date_range,
            )
        )

        clauses = (
            await self.clause_analytics(
                context,
                date_range,
            )
        )

        risks = (
            await self.risk_analytics(
                context,
                date_range,
            )
        )

        reviews = (
            await self.review_analytics(
                context,
                date_range,
            )
        )

        renewals = (
            await self.renewal_analytics(
                context,
                date_range,
            )
        )

        ai: Optional[AIAnalytics] = None

        if context.include_ai_metrics:
            ai = await self.ai_analytics(
                context,
                date_range,
            )

        security = (
            await self.security_analytics(
                context,
                date_range,
            )
        )

        return DashboardAnalytics(
            generated_at=self._utcnow(),

            tenant_id=context.tenant_id,

            organization_id=(
                context.organization_id
            ),

            period=context.period,

            start_at=date_range.start_at,
            end_at=date_range.end_at,

            documents=documents,
            contracts=contracts,
            clauses=clauses,
            risks=risks,
            reviews=reviews,
            renewals=renewals,

            ai=ai,
            security=security,
        )

    # ========================================================
    # DOCUMENTS
    # ========================================================

    async def document_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> DocumentAnalytics:

        repository = (
            self.document_repository
        )

        if repository is None:
            return DocumentAnalytics()

        filters = self._make_filter(
            "DocumentFilter",

            tenant_id=context.tenant_id,

            organization_id=(
                context.organization_id
            ),
        )

        stats = await self._statistics(
            repository,
            filters,
        )

        result = DocumentAnalytics(
            total_documents=self._number(
                stats,
                "total",
            ),

            processed_documents=self._number(
                stats,
                "processed",
                "ready",
            ),

            failed_documents=self._number(
                stats,
                "failed",
            ),

            processing_documents=self._number(
                stats,
                "processing",
            ),

            indexed_documents=self._number(
                stats,
                "indexed",
            ),

            pending_indexing=self._number(
                stats,
                "pending_indexing",
            ),

            ocr_documents=self._number(
                stats,
                "ocr_documents",
                "ocr",
            ),

            total_pages=self._number(
                stats,
                "total_pages",
            ),

            total_chunks=self._number(
                stats,
                "total_chunks",
            ),

            by_type=self._mapping(
                stats,
                "by_type",
            ),

            by_status=self._mapping(
                stats,
                "by_status",
            ),

            by_language=self._mapping(
                stats,
                "by_language",
            ),
        )

        return result

    # ========================================================
    # CONTRACTS
    # ========================================================

    async def contract_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> ContractAnalytics:

        repository = (
            self.contract_repository
        )

        if repository is None:
            return ContractAnalytics()

        filters = self._make_filter(
            "ContractFilter",

            tenant_id=context.tenant_id,

            organization_id=(
                context.organization_id
            ),
        )

        stats = await self._statistics(
            repository,
            filters,
        )

        by_status = self._mapping(
            stats,
            "by_status",
        )

        by_risk = self._mapping(
            stats,
            "by_risk",
            "by_risk_level",
        )

        result = ContractAnalytics(
            total_contracts=self._number(
                stats,
                "total",
            ),

            active_contracts=(
                by_status.get(
                    "active",
                    0,
                )
            ),

            draft_contracts=(
                by_status.get(
                    "draft",
                    0,
                )
            ),

            expired_contracts=(
                by_status.get(
                    "expired",
                    0,
                )
            ),

            terminated_contracts=(
                by_status.get(
                    "terminated",
                    0,
                )
            ),

            auto_renewing_contracts=(
                self._number(
                    stats,
                    "auto_renewing",
                    "auto_renewal",
                )
            ),

            total_contract_value=float(
                self._value(
                    stats,
                    "total_contract_value",
                    "total_value",
                    default=0.0,
                )
                or 0.0
            ),

            by_type=self._mapping(
                stats,
                "by_type",
            ),

            by_status=by_status,

            by_risk_level=by_risk,

            by_currency=self._mapping_float(
                stats,
                "by_currency",
            ),
        )

        now = self._utcnow()

        result.contracts_expiring_30_days = (
            await self._count_expiring_contracts(
                repository,
                context,
                now,
                now + timedelta(days=30),
            )
        )

        result.contracts_expiring_60_days = (
            await self._count_expiring_contracts(
                repository,
                context,
                now,
                now + timedelta(days=60),
            )
        )

        result.contracts_expiring_90_days = (
            await self._count_expiring_contracts(
                repository,
                context,
                now,
                now + timedelta(days=90),
            )
        )

        return result

    async def _count_expiring_contracts(
        self,
        repository: Any,
        context: AnalyticsContext,
        start_at: datetime,
        end_at: datetime,
    ) -> int:

        method = getattr(
            repository,
            "list_expiring_between",
            None,
        )

        if method is None:
            return 0

        try:
            result = await method(
                tenant_id=context.tenant_id,
                organization_id=(
                    context.organization_id
                ),
                start_date=start_at.date(),
                end_date=end_at.date(),
                limit=1000,
                offset=0,
            )

            return self._result_count(
                result
            )

        except TypeError:
            try:
                result = await method(
                    tenant_id=(
                        context.tenant_id
                    ),
                    start_date=(
                        start_at.date()
                    ),
                    end_date=(
                        end_at.date()
                    ),
                    limit=1000,
                    offset=0,
                )

                return self._result_count(
                    result
                )

            except TypeError:
                return 0

    # ========================================================
    # CLAUSES
    # ========================================================

    async def clause_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> ClauseAnalytics:

        repository = (
            self.clause_repository
        )

        if repository is None:
            return ClauseAnalytics()

        filters = self._make_filter(
            "ClauseFilter",

            tenant_id=context.tenant_id,

            organization_id=(
                context.organization_id
            ),
        )

        stats = await self._statistics(
            repository,
            filters,
        )

        by_type = self._mapping(
            stats,
            "by_type",
        )

        total = self._number(
            stats,
            "total",
        )

        return ClauseAnalytics(
            total_clauses=total,

            high_risk_clauses=(
                self._number(
                    stats,
                    "high",
                    "high_risk",
                )
            ),

            critical_clauses=(
                self._number(
                    stats,
                    "critical",
                    "critical_risk",
                )
            ),

            average_confidence=(
                self._optional_float(
                    stats,
                    "average_confidence",
                    "avg_confidence",
                )
            ),

            average_risk_score=(
                self._optional_float(
                    stats,
                    "average_risk_score",
                    "avg_risk_score",
                )
            ),

            by_type=by_type,

            by_risk_level=(
                self._mapping(
                    stats,
                    "by_risk",
                    "by_risk_level",
                )
            ),

            top_clause_types=(
                self._distribution(
                    by_type,
                    total,
                    limit=10,
                )
            ),
        )

    # ========================================================
    # RISKS
    # ========================================================

    async def risk_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> RiskAnalytics:

        repository = (
            self.risk_repository
        )

        if repository is None:
            return RiskAnalytics()

        filters = self._make_filter(
            "RiskFilter",

            tenant_id=context.tenant_id,

            organization_id=(
                context.organization_id
            ),
        )

        stats = await self._statistics(
            repository,
            filters,
        )

        return RiskAnalytics(
            total_risks=self._number(
                stats,
                "total",
            ),

            open_risks=self._number(
                stats,
                "open",
            ),

            acknowledged_risks=(
                self._number(
                    stats,
                    "acknowledged",
                )
            ),

            mitigated_risks=(
                self._number(
                    stats,
                    "mitigated",
                )
            ),

            resolved_risks=(
                self._number(
                    stats,
                    "resolved",
                )
            ),

            low_risks=self._number(
                stats,
                "low",
            ),

            medium_risks=self._number(
                stats,
                "medium",
            ),

            high_risks=self._number(
                stats,
                "high",
            ),

            critical_risks=self._number(
                stats,
                "critical",
            ),

            pending_review=self._number(
                stats,
                "unreviewed",
            ),

            escalated_risks=self._number(
                stats,
                "escalated",
            ),

            average_score=(
                self._optional_float(
                    stats,
                    "average_score",
                )
            ),

            average_confidence=(
                self._optional_float(
                    stats,
                    "average_confidence",
                )
            ),

            by_category=self._mapping(
                stats,
                "by_category",
            ),

            by_level=self._mapping(
                stats,
                "by_level",
            ),

            by_status=self._mapping(
                stats,
                "by_status",
            ),
        )

    # ========================================================
    # REVIEWS
    # ========================================================

    async def review_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> ReviewAnalytics:

        repository = (
            self.review_repository
        )

        if repository is None:
            return ReviewAnalytics()

        filters = self._make_filter(
            "ReviewFilter",

            tenant_id=context.tenant_id,

            organization_id=(
                context.organization_id
            ),
        )

        stats = await self._statistics(
            repository,
            filters,
        )

        total = self._number(
            stats,
            "total",
        )

        completed = self._number(
            stats,
            "completed",
        )

        approved = self._number(
            stats,
            "approved",
        )

        completion_rate = (
            (completed / total) * 100
            if total
            else 0.0
        )

        approval_rate = (
            (approved / completed) * 100
            if completed
            else 0.0
        )

        return ReviewAnalytics(
            total_reviews=total,

            pending_reviews=self._number(
                stats,
                "pending",
            ),

            assigned_reviews=self._number(
                stats,
                "assigned",
            ),

            in_review=self._number(
                stats,
                "in_review",
            ),

            completed_reviews=completed,

            approved_reviews=approved,

            rejected_reviews=self._number(
                stats,
                "rejected",
            ),

            changes_requested=self._number(
                stats,
                "changes_requested",
            ),

            overdue_reviews=self._number(
                stats,
                "overdue",
            ),

            unassigned_reviews=self._number(
                stats,
                "unassigned",
            ),

            high_priority_reviews=(
                self._number(
                    stats,
                    "high_priority",
                )
            ),

            critical_priority_reviews=(
                self._number(
                    stats,
                    "critical_priority",
                )
            ),

            by_type=self._mapping(
                stats,
                "by_type",
            ),

            by_status=self._mapping(
                stats,
                "by_status",
            ),

            by_decision=self._mapping(
                stats,
                "by_decision",
            ),

            by_priority=self._mapping(
                stats,
                "by_priority",
            ),

            completion_rate=round(
                completion_rate,
                2,
            ),

            approval_rate=round(
                approval_rate,
                2,
            ),
        )

    # ========================================================
    # RENEWALS
    # ========================================================

    async def renewal_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> RenewalAnalytics:

        repository = (
            self.renewal_repository
        )

        if repository is None:
            return RenewalAnalytics()

        filters = self._make_filter(
            "RenewalFilter",

            tenant_id=context.tenant_id,

            organization_id=(
                context.organization_id
            ),
        )

        stats = await self._statistics(
            repository,
            filters,
        )

        result = RenewalAnalytics(
            total_renewals=self._number(
                stats,
                "total",
            ),

            upcoming_renewals=self._number(
                stats,
                "upcoming",
            ),

            overdue_renewals=self._number(
                stats,
                "overdue",
            ),

            completed_renewals=self._number(
                stats,
                "completed",
            ),

            cancelled_renewals=self._number(
                stats,
                "cancelled",
            ),

            auto_renewals=self._number(
                stats,
                "auto_renewals",
                "auto_renewal",
            ),

            by_status=self._mapping(
                stats,
                "by_status",
            ),

            by_priority=self._mapping(
                stats,
                "by_priority",
            ),
        )

        now = self._utcnow()

        result.renewals_30_days = (
            await self._count_upcoming(
                repository,
                context,
                now,
                30,
            )
        )

        result.renewals_60_days = (
            await self._count_upcoming(
                repository,
                context,
                now,
                60,
            )
        )

        result.renewals_90_days = (
            await self._count_upcoming(
                repository,
                context,
                now,
                90,
            )
        )

        return result

    async def _count_upcoming(
        self,
        repository: Any,
        context: AnalyticsContext,
        now: datetime,
        days: int,
    ) -> int:

        method = getattr(
            repository,
            "list_upcoming",
            None,
        )

        if method is None:
            return 0

        try:
            result = await method(
                tenant_id=context.tenant_id,

                organization_id=(
                    context.organization_id
                ),

                start_date=now.date(),

                end_date=(
                    now
                    + timedelta(days=days)
                ).date(),

                limit=1000,
                offset=0,
            )

            return self._result_count(
                result
            )

        except TypeError:
            return 0

    # ========================================================
    # AI ANALYTICS
    # ========================================================

    async def ai_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> AIAnalytics:

        repository = (
            self.telemetry_repository
            or self.model_run_repository
        )

        if repository is None:
            return AIAnalytics()

        if date_range is None:
            date_range = (
                self.resolve_date_range(
                    context
                )
            )

        method = getattr(
            repository,
            "analytics",
            None,
        )

        if method is None:
            method = getattr(
                repository,
                "statistics",
                None,
            )

        if method is None:
            return AIAnalytics()

        try:
            stats = await method(
                tenant_id=context.tenant_id,

                organization_id=(
                    context.organization_id
                ),

                start_at=(
                    date_range.start_at
                ),

                end_at=(
                    date_range.end_at
                ),
            )

        except TypeError:
            return AIAnalytics()

        return AIAnalytics(
            total_queries=self._number(
                stats,
                "total_queries",
            ),

            rag_queries=self._number(
                stats,
                "rag_queries",
            ),

            agent_runs=self._number(
                stats,
                "agent_runs",
            ),

            model_runs=self._number(
                stats,
                "model_runs",
                "total_runs",
            ),

            successful_runs=self._number(
                stats,
                "successful_runs",
            ),

            failed_runs=self._number(
                stats,
                "failed_runs",
            ),

            average_latency_ms=(
                self._optional_float(
                    stats,
                    "average_latency_ms",
                    "avg_latency_ms",
                )
            ),

            p95_latency_ms=(
                self._optional_float(
                    stats,
                    "p95_latency_ms",
                )
            ),

            total_prompt_tokens=(
                self._number(
                    stats,
                    "prompt_tokens",
                    "total_prompt_tokens",
                )
            ),

            total_completion_tokens=(
                self._number(
                    stats,
                    "completion_tokens",
                    "total_completion_tokens",
                )
            ),

            total_tokens=self._number(
                stats,
                "total_tokens",
            ),

            estimated_cost=float(
                self._value(
                    stats,
                    "estimated_cost",
                    "total_cost",
                    default=0.0,
                )
                or 0.0
            ),

            average_retrieval_score=(
                self._optional_float(
                    stats,
                    "average_retrieval_score",
                )
            ),

            average_reranker_score=(
                self._optional_float(
                    stats,
                    "average_reranker_score",
                )
            ),

            groundedness_score=(
                self._optional_float(
                    stats,
                    "groundedness_score",
                )
            ),

            citation_accuracy=(
                self._optional_float(
                    stats,
                    "citation_accuracy",
                )
            ),

            hallucination_rate=(
                self._optional_float(
                    stats,
                    "hallucination_rate",
                )
            ),

            answer_validation_rate=(
                self._optional_float(
                    stats,
                    "answer_validation_rate",
                )
            ),

            by_model=self._mapping(
                stats,
                "by_model",
            ),

            by_query_type=self._mapping(
                stats,
                "by_query_type",
            ),

            by_agent=self._mapping(
                stats,
                "by_agent",
            ),
        )

    # ========================================================
    # SECURITY
    # ========================================================

    async def security_analytics(
        self,
        context: AnalyticsContext,
        date_range: Optional[
            DateRange
        ] = None,
    ) -> SecurityAnalytics:

        repository = (
            self.audit_repository
        )

        if repository is None:
            return SecurityAnalytics()

        if date_range is None:
            date_range = (
                self.resolve_date_range(
                    context
                )
            )

        method = getattr(
            repository,
            "statistics",
            None,
        )

        if method is None:
            return SecurityAnalytics()

        try:
            filters = self._make_filter(
                "AuditFilter",

                tenant_id=(
                    context.tenant_id
                ),

                organization_id=(
                    context.organization_id
                ),

                created_from=(
                    date_range.start_at
                ),

                created_to=(
                    date_range.end_at
                ),
            )

            stats = await method(filters)

        except (TypeError, AttributeError):
            return SecurityAnalytics()

        by_action = self._mapping(
            stats,
            "by_action",
            "by_event_type",
        )

        by_severity = self._mapping(
            stats,
            "by_severity",
        )

        return SecurityAnalytics(
            security_events=self._number(
                stats,
                "security_events",
            ),

            permission_denials=(
                by_action.get(
                    "permission_denied",
                    0,
                )
            ),

            prompt_injection_events=(
                by_action.get(
                    "prompt_injection",
                    0,
                )
            ),

            pii_events=(
                by_action.get(
                    "pii_detected",
                    0,
                )
            ),

            blocked_requests=(
                self._number(
                    stats,
                    "blocked_requests",
                )
            ),

            critical_events=(
                by_severity.get(
                    "critical",
                    0,
                )
            ),

            by_event_type=by_action,
            by_severity=by_severity,
        )

    # ========================================================
    # TREND CALCULATIONS
    # ========================================================

    @staticmethod
    def calculate_trend(
        current: float,
        previous: float,
    ) -> TrendMetric:

        absolute = current - previous

        if absolute > 0:
            direction = TrendDirection.UP

        elif absolute < 0:
            direction = TrendDirection.DOWN

        else:
            direction = TrendDirection.FLAT

        percentage: Optional[
            float
        ]

        if previous == 0:
            percentage = (
                None
                if current != 0
                else 0.0
            )

        else:
            percentage = (
                absolute
                / abs(previous)
            ) * 100

            percentage = round(
                percentage,
                2,
            )

        return TrendMetric(
            current=current,
            previous=previous,

            absolute_change=round(
                absolute,
                4,
            ),

            percentage_change=percentage,

            direction=direction,
        )

    # ========================================================
    # DATE RANGE
    # ========================================================

    @classmethod
    def resolve_date_range(
        cls,
        context: AnalyticsContext,
    ) -> DateRange:

        now = cls._utcnow()

        if (
            context.period
            == AnalyticsPeriod.CUSTOM
        ):
            if (
                context.start_at is None
                or context.end_at is None
            ):
                raise ValueError(
                    "CUSTOM analytics period "
                    "requires start_at and "
                    "end_at."
                )

            start = cls._ensure_utc(
                context.start_at
            )

            end = cls._ensure_utc(
                context.end_at
            )

            if start >= end:
                raise ValueError(
                    "start_at must be before "
                    "end_at."
                )

            return DateRange(
                start_at=start,
                end_at=end,
            )

        if (
            context.period
            == AnalyticsPeriod.TODAY
        ):
            start = now.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif (
            context.period
            == AnalyticsPeriod.LAST_7_DAYS
        ):
            start = (
                now
                - timedelta(days=7)
            )

        elif (
            context.period
            == AnalyticsPeriod.LAST_90_DAYS
        ):
            start = (
                now
                - timedelta(days=90)
            )

        elif (
            context.period
            == AnalyticsPeriod.THIS_MONTH
        ):
            start = now.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif (
            context.period
            == AnalyticsPeriod.THIS_QUARTER
        ):
            quarter_month = (
                ((now.month - 1) // 3)
                * 3
                + 1
            )

            start = now.replace(
                month=quarter_month,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        elif (
            context.period
            == AnalyticsPeriod.THIS_YEAR
        ):
            start = now.replace(
                month=1,
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

        else:
            start = (
                now
                - timedelta(days=30)
            )

        return DateRange(
            start_at=start,
            end_at=now,
        )

    @staticmethod
    def previous_period(
        current: DateRange,
    ) -> DateRange:

        duration = (
            current.end_at
            - current.start_at
        )

        return DateRange(
            start_at=(
                current.start_at
                - duration
            ),

            end_at=current.start_at,
        )

    # ========================================================
    # DISTRIBUTIONS
    # ========================================================

    @staticmethod
    def _distribution(
        values: Mapping[str, int],
        total: int,
        *,
        limit: Optional[int] = None,
    ) -> list[DistributionItem]:

        ordered = sorted(
            values.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if limit is not None:
            ordered = ordered[:limit]

        result: list[
            DistributionItem
        ] = []

        for name, count in ordered:
            percentage = (
                (count / total) * 100
                if total
                else 0.0
            )

            result.append(
                DistributionItem(
                    name=str(name),
                    count=int(count),
                    percentage=round(
                        percentage,
                        2,
                    ),
                )
            )

        return result

    # ========================================================
    # REPOSITORY ADAPTERS
    # ========================================================

    async def _statistics(
        self,
        repository: Any,
        filters: Any,
    ) -> Any:

        method = getattr(
            repository,
            "statistics",
            None,
        )

        if method is None:
            return {}

        try:
            return await method(filters)

        except TypeError:
            try:
                return await method(
                    filters=filters
                )

            except TypeError:
                return {}

    @staticmethod
    def _result_count(
        result: Any,
    ) -> int:

        if result is None:
            return 0

        total = getattr(
            result,
            "total",
            None,
        )

        if total is not None:
            return int(total)

        items = getattr(
            result,
            "items",
            None,
        )

        if items is not None:
            return len(items)

        if isinstance(
            result,
            Sequence,
        ) and not isinstance(
            result,
            (str, bytes),
        ):
            return len(result)

        return 0

    # ========================================================
    # GENERIC FILTER FACTORY
    # ========================================================

    @staticmethod
    def _make_filter(
        filter_name: str,
        **kwargs: Any,
    ) -> Any:
        """
        Creates a lightweight filter object.

        Your concrete repository dataclass filters can replace
        this adapter directly through normal imports.

        Example:
            from ...risk_repository import RiskFilter
        """

        values = {
            key: value
            for key, value
            in kwargs.items()
            if value is not None
        }

        return type(
            filter_name,
            (),
            values,
        )()

    # ========================================================
    # VALUE ADAPTERS
    # ========================================================

    @classmethod
    def _value(
        cls,
        source: Any,
        *names: str,
        default: Any = None,
    ) -> Any:

        if source is None:
            return default

        for name in names:

            if isinstance(
                source,
                Mapping,
            ):
                if name in source:
                    return source[name]

            elif hasattr(
                source,
                name,
            ):
                return getattr(
                    source,
                    name,
                )

        return default

    @classmethod
    def _number(
        cls,
        source: Any,
        *names: str,
    ) -> int:

        value = cls._value(
            source,
            *names,
            default=0,
        )

        if value is None:
            return 0

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return 0

    @classmethod
    def _optional_float(
        cls,
        source: Any,
        *names: str,
    ) -> Optional[float]:

        value = cls._value(
            source,
            *names,
            default=None,
        )

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @classmethod
    def _mapping(
        cls,
        source: Any,
        *names: str,
    ) -> dict[str, int]:

        value = cls._value(
            source,
            *names,
            default={},
        )

        if not isinstance(
            value,
            Mapping,
        ):
            return {}

        result: dict[str, int] = {}

        for key, item in value.items():
            try:
                result[str(key)] = int(item)

            except (
                TypeError,
                ValueError,
            ):
                continue

        return result

    @classmethod
    def _mapping_float(
        cls,
        source: Any,
        *names: str,
    ) -> dict[str, float]:

        value = cls._value(
            source,
            *names,
            default={},
        )

        if not isinstance(
            value,
            Mapping,
        ):
            return {}

        result: dict[
            str,
            float
        ] = {}

        for key, item in value.items():
            try:
                result[str(key)] = (
                    float(item)
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

        return result

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_context(
        context: AnalyticsContext,
    ) -> None:

        if not context.tenant_id:
            raise ValueError(
                "tenant_id is mandatory "
                "for analytics."
            )

    # ========================================================
    # DATETIME
    # ========================================================

    @staticmethod
    def _ensure_utc(
        value: Optional[datetime],
    ) -> Optional[datetime]:

        if value is None:
            return None

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )

    @staticmethod
    def _utcnow() -> datetime:

        return datetime.now(
            timezone.utc
        )