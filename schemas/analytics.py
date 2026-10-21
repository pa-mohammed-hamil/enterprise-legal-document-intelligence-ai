"""
backend/app/services/analytics.py

Analytics service for Enterprise Legal Document Intelligence AI.

Provides:
- Document analytics
- Clause analytics
- Risk analytics
- Expiry / renewal analytics
- Obligation analytics
- Review analytics
- RAG / AI analytics
- Portfolio summary
- Dashboard-ready metrics

The service accepts dictionaries, ORM objects, or Pydantic-like objects
so it remains decoupled from the persistence layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from statistics import mean
from typing import Any, Mapping, Sequence


# ============================================================================
# Models
# ============================================================================


@dataclass(slots=True)
class Metric:
    name: str
    value: int | float
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
        }


@dataclass(slots=True)
class AnalyticsSnapshot:
    generated_at: datetime

    documents: dict[str, Any] = field(default_factory=dict)
    clauses: dict[str, Any] = field(default_factory=dict)
    risks: dict[str, Any] = field(default_factory=dict)
    expiries: dict[str, Any] = field(default_factory=dict)
    renewals: dict[str, Any] = field(default_factory=dict)
    obligations: dict[str, Any] = field(default_factory=dict)
    reviews: dict[str, Any] = field(default_factory=dict)
    rag: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "documents": self.documents,
            "clauses": self.clauses,
            "risks": self.risks,
            "expiries": self.expiries,
            "renewals": self.renewals,
            "obligations": self.obligations,
            "reviews": self.reviews,
            "rag": self.rag,
        }


# ============================================================================
# Analytics Service
# ============================================================================


class AnalyticsService:
    """
    Builds analytics from normalized domain records.

    Designed for use by:
        API endpoints
        admin dashboard
        analytics dashboard
        scheduled reports
        BI export
        portfolio metrics
    """

    RISK_ORDER = {
        "critical": 4,
        "high": 3,
        "medium": 2,
        "low": 1,
        "none": 0,
    }

    def snapshot(
        self,
        *,
        documents: Sequence[Any] = (),
        clauses: Sequence[Any] = (),
        risks: Sequence[Any] = (),
        expiries: Sequence[Any] = (),
        renewals: Sequence[Any] = (),
        obligations: Sequence[Any] = (),
        reviews: Sequence[Any] = (),
        rag_events: Sequence[Any] = (),
        now: datetime | None = None,
    ) -> AnalyticsSnapshot:

        now = now or datetime.now(timezone.utc)

        return AnalyticsSnapshot(
            generated_at=now,
            documents=self.document_analytics(documents),
            clauses=self.clause_analytics(clauses),
            risks=self.risk_analytics(risks),
            expiries=self.expiry_analytics(
                expiries,
                now=now,
            ),
            renewals=self.renewal_analytics(
                renewals,
                now=now,
            ),
            obligations=self.obligation_analytics(
                obligations,
                now=now,
            ),
            reviews=self.review_analytics(reviews),
            rag=self.rag_analytics(rag_events),
        )

    # ========================================================================
    # Documents
    # ========================================================================

    def document_analytics(
        self,
        documents: Sequence[Any],
    ) -> dict[str, Any]:

        total = len(documents)

        status_counter: Counter[str] = Counter()
        type_counter: Counter[str] = Counter()
        language_counter: Counter[str] = Counter()

        page_counts: list[int] = []
        processing_times: list[float] = []

        for document in documents:

            status_counter[
                self._string(
                    self._get(
                        document,
                        "status",
                        default="unknown",
                    )
                )
            ] += 1

            type_counter[
                self._string(
                    self._get(
                        document,
                        "document_type",
                        "type",
                        default="unknown",
                    )
                )
            ] += 1

            language_counter[
                self._string(
                    self._get(
                        document,
                        "language",
                        default="unknown",
                    )
                )
            ] += 1

            pages = self._number(
                self._get(
                    document,
                    "page_count",
                    "pages",
                    default=None,
                )
            )

            if pages is not None:
                page_counts.append(int(pages))

            processing = self._number(
                self._get(
                    document,
                    "processing_time_ms",
                    default=None,
                )
            )

            if processing is not None:
                processing_times.append(processing)

        processed = sum(
            status_counter.get(status, 0)
            for status in (
                "processed",
                "completed",
                "indexed",
                "ready",
            )
        )

        return {
            "total_documents": total,
            "processed_documents": processed,
            "processing_rate": self._ratio(
                processed,
                total,
            ),
            "total_pages": sum(page_counts),
            "average_pages": self._average(page_counts),
            "average_processing_time_ms": self._average(
                processing_times
            ),
            "by_status": dict(status_counter),
            "by_type": dict(type_counter),
            "by_language": dict(language_counter),
        }

    # ========================================================================
    # Clauses
    # ========================================================================

    def clause_analytics(
        self,
        clauses: Sequence[Any],
    ) -> dict[str, Any]:

        clause_types: Counter[str] = Counter()
        risk_levels: Counter[str] = Counter()

        confidence_scores: list[float] = []

        for clause in clauses:

            clause_types[
                self._string(
                    self._get(
                        clause,
                        "clause_type",
                        "type",
                        "category",
                        default="unknown",
                    )
                )
            ] += 1

            risk_levels[
                self._string(
                    self._get(
                        clause,
                        "risk_level",
                        "severity",
                        default="unknown",
                    )
                )
            ] += 1

            confidence = self._number(
                self._get(
                    clause,
                    "confidence",
                    "confidence_score",
                    default=None,
                )
            )

            if confidence is not None:
                confidence_scores.append(
                    self._normalize_probability(
                        confidence
                    )
                )

        return {
            "total_clauses": len(clauses),
            "unique_clause_types": len(clause_types),
            "by_type": dict(
                clause_types.most_common()
            ),
            "by_risk_level": dict(risk_levels),
            "average_confidence": self._average(
                confidence_scores
            ),
        }

    # ========================================================================
    # Risk
    # ========================================================================

    def risk_analytics(
        self,
        risks: Sequence[Any],
    ) -> dict[str, Any]:

        severity_counter: Counter[str] = Counter()
        category_counter: Counter[str] = Counter()

        risk_scores: list[float] = []

        critical_or_high = 0

        for risk in risks:

            severity = self._string(
                self._get(
                    risk,
                    "severity",
                    "risk_level",
                    default="unknown",
                )
            )

            category = self._string(
                self._get(
                    risk,
                    "category",
                    "risk_type",
                    "type",
                    default="unknown",
                )
            )

            severity_counter[severity] += 1
            category_counter[category] += 1

            if severity in {"critical", "high"}:
                critical_or_high += 1

            score = self._number(
                self._get(
                    risk,
                    "score",
                    "risk_score",
                    default=None,
                )
            )

            if score is not None:
                risk_scores.append(
                    self._normalize_probability(score)
                )

        return {
            "total_risks": len(risks),
            "high_or_critical": critical_or_high,
            "high_risk_rate": self._ratio(
                critical_or_high,
                len(risks),
            ),
            "average_risk_score": self._average(
                risk_scores
            ),
            "by_severity": dict(severity_counter),
            "by_category": dict(
                category_counter.most_common()
            ),
        }

    # ========================================================================
    # Expiry
    # ========================================================================

    def expiry_analytics(
        self,
        expiries: Sequence[Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:

        now = now or datetime.now(timezone.utc)
        today = now.date()

        expired = 0
        next_30 = 0
        next_60 = 0
        next_90 = 0
        future = 0
        missing_date = 0

        for item in expiries:

            expiry_date = self._date(
                self._get(
                    item,
                    "expiry_date",
                    "expiration_date",
                    "end_date",
                    default=None,
                )
            )

            if expiry_date is None:
                missing_date += 1
                continue

            days = (
                expiry_date - today
            ).days

            if days < 0:
                expired += 1

            elif days <= 30:
                next_30 += 1

            elif days <= 60:
                next_60 += 1

            elif days <= 90:
                next_90 += 1

            else:
                future += 1

        return {
            "total": len(expiries),
            "expired": expired,
            "expiring_0_30_days": next_30,
            "expiring_31_60_days": next_60,
            "expiring_61_90_days": next_90,
            "beyond_90_days": future,
            "missing_expiry_date": missing_date,
            "attention_required": (
                expired
                + next_30
            ),
        }

    # ========================================================================
    # Renewals
    # ========================================================================

    def renewal_analytics(
        self,
        renewals: Sequence[Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:

        now = now or datetime.now(timezone.utc)
        today = now.date()

        automatic = 0
        manual = 0
        upcoming_30 = 0
        overdue = 0

        status_counter: Counter[str] = Counter()

        for renewal in renewals:

            auto_renew = self._get(
                renewal,
                "auto_renew",
                "automatic_renewal",
                default=None,
            )

            if auto_renew is True:
                automatic += 1

            elif auto_renew is False:
                manual += 1

            status_counter[
                self._string(
                    self._get(
                        renewal,
                        "status",
                        default="unknown",
                    )
                )
            ] += 1

            renewal_date = self._date(
                self._get(
                    renewal,
                    "renewal_date",
                    "next_renewal_date",
                    default=None,
                )
            )

            if renewal_date:

                days = (
                    renewal_date - today
                ).days

                if days < 0:
                    overdue += 1

                elif days <= 30:
                    upcoming_30 += 1

        return {
            "total": len(renewals),
            "automatic_renewals": automatic,
            "manual_renewals": manual,
            "upcoming_30_days": upcoming_30,
            "overdue": overdue,
            "by_status": dict(status_counter),
        }

    # ========================================================================
    # Obligations
    # ========================================================================

    def obligation_analytics(
        self,
        obligations: Sequence[Any],
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:

        now = now or datetime.now(timezone.utc)
        today = now.date()

        status_counter: Counter[str] = Counter()
        party_counter: Counter[str] = Counter()

        overdue = 0
        upcoming = 0

        for obligation in obligations:

            status = self._string(
                self._get(
                    obligation,
                    "status",
                    default="unknown",
                )
            )

            status_counter[status] += 1

            party = self._string(
                self._get(
                    obligation,
                    "party",
                    "obligated_party",
                    "responsible_party",
                    default="unknown",
                )
            )

            party_counter[party] += 1

            deadline = self._date(
                self._get(
                    obligation,
                    "deadline",
                    "due_date",
                    default=None,
                )
            )

            if (
                deadline is not None
                and status
                not in {
                    "completed",
                    "fulfilled",
                    "closed",
                }
            ):

                days = (
                    deadline - today
                ).days

                if days < 0:
                    overdue += 1

                elif days <= 30:
                    upcoming += 1

        return {
            "total_obligations": len(obligations),
            "overdue": overdue,
            "due_next_30_days": upcoming,
            "by_status": dict(status_counter),
            "by_party": dict(
                party_counter.most_common()
            ),
        }

    # ========================================================================
    # Reviews
    # ========================================================================

    def review_analytics(
        self,
        reviews: Sequence[Any],
    ) -> dict[str, Any]:

        status_counter: Counter[str] = Counter()
        decision_counter: Counter[str] = Counter()

        review_times: list[float] = []

        for review in reviews:

            status_counter[
                self._string(
                    self._get(
                        review,
                        "status",
                        default="unknown",
                    )
                )
            ] += 1

            decision_counter[
                self._string(
                    self._get(
                        review,
                        "decision",
                        "outcome",
                        default="unknown",
                    )
                )
            ] += 1

            duration = self._number(
                self._get(
                    review,
                    "review_time_seconds",
                    "duration_seconds",
                    default=None,
                )
            )

            if duration is not None:
                review_times.append(duration)

        pending = sum(
            status_counter.get(value, 0)
            for value in (
                "pending",
                "in_review",
                "open",
            )
        )

        return {
            "total_reviews": len(reviews),
            "pending_reviews": pending,
            "average_review_time_seconds": self._average(
                review_times
            ),
            "by_status": dict(status_counter),
            "by_decision": dict(decision_counter),
        }

    # ========================================================================
    # RAG / LLM
    # ========================================================================

    def rag_analytics(
        self,
        events: Sequence[Any],
    ) -> dict[str, Any]:

        latencies: list[float] = []
        retrieval_counts: list[float] = []
        reranked_counts: list[float] = []
        confidence_scores: list[float] = []
        validation_scores: list[float] = []
        uncertainty_scores: list[float] = []

        query_types: Counter[str] = Counter()

        failures = 0
        human_review = 0

        for event in events:

            query_types[
                self._string(
                    self._get(
                        event,
                        "query_type",
                        default="unknown",
                    )
                )
            ] += 1

            latency = self._number(
                self._get(
                    event,
                    "latency_ms",
                    default=None,
                )
            )

            if latency is not None:
                latencies.append(latency)

            retrieved = self._number(
                self._get(
                    event,
                    "retrieved_count",
                    default=None,
                )
            )

            if retrieved is not None:
                retrieval_counts.append(retrieved)

            reranked = self._number(
                self._get(
                    event,
                    "reranked_count",
                    default=None,
                )
            )

            if reranked is not None:
                reranked_counts.append(reranked)

            confidence = self._number(
                self._get(
                    event,
                    "confidence_score",
                    default=None,
                )
            )

            if confidence is not None:
                confidence_scores.append(
                    self._normalize_probability(
                        confidence
                    )
                )

            validation = self._number(
                self._get_nested(
                    event,
                    ("validation", "score"),
                    fallback_names=(
                        "validation_score",
                    ),
                )
            )

            if validation is not None:
                validation_scores.append(
                    self._normalize_probability(
                        validation
                    )
                )

            uncertainty = self._number(
                self._get_nested(
                    event,
                    (
                        "uncertainty",
                        "uncertainty_score",
                    ),
                    fallback_names=(
                        "uncertainty_score",
                    ),
                )
            )

            if uncertainty is not None:
                uncertainty_scores.append(
                    self._normalize_probability(
                        uncertainty
                    )
                )

            status = self._string(
                self._get(
                    event,
                    "status",
                    default="success",
                )
            )

            if status in {
                "failed",
                "error",
                "failure",
            }:
                failures += 1

            review_required = self._get_nested(
                event,
                (
                    "uncertainty",
                    "requires_review",
                ),
                fallback_names=(
                    "requires_review",
                ),
            )

            if review_required is True:
                human_review += 1

        total = len(events)

        return {
            "total_queries": total,
            "successful_queries": max(
                0,
                total - failures,
            ),
            "failed_queries": failures,
            "success_rate": self._ratio(
                total - failures,
                total,
            ),
            "average_latency_ms": self._average(
                latencies
            ),
            "average_retrieved_chunks": self._average(
                retrieval_counts
            ),
            "average_reranked_chunks": self._average(
                reranked_counts
            ),
            "average_confidence": self._average(
                confidence_scores
            ),
            "average_validation_score": self._average(
                validation_scores
            ),
            "average_uncertainty": self._average(
                uncertainty_scores
            ),
            "human_review_required": human_review,
            "human_review_rate": self._ratio(
                human_review,
                total,
            ),
            "by_query_type": dict(
                query_types.most_common()
            ),
        }

    # ========================================================================
    # Portfolio / executive KPIs
    # ========================================================================

    def executive_summary(
        self,
        snapshot: AnalyticsSnapshot,
    ) -> dict[str, Any]:

        return {
            "documents": snapshot.documents.get(
                "total_documents",
                0,
            ),
            "clauses": snapshot.clauses.get(
                "total_clauses",
                0,
            ),
            "high_or_critical_risks": snapshot.risks.get(
                "high_or_critical",
                0,
            ),
            "expiring_within_30_days": snapshot.expiries.get(
                "expiring_0_30_days",
                0,
            ),
            "overdue_obligations": snapshot.obligations.get(
                "overdue",
                0,
            ),
            "pending_reviews": snapshot.reviews.get(
                "pending_reviews",
                0,
            ),
            "rag_queries": snapshot.rag.get(
                "total_queries",
                0,
            ),
            "rag_success_rate": snapshot.rag.get(
                "success_rate",
                0.0,
            ),
            "rag_average_confidence": snapshot.rag.get(
                "average_confidence",
                0.0,
            ),
        }

    # ========================================================================
    # Helpers
    # ========================================================================

    @staticmethod
    def _get(
        obj: Any,
        *names: str,
        default: Any = None,
    ) -> Any:

        if isinstance(obj, Mapping):

            for name in names:
                if name in obj:
                    return obj[name]

            return default

        for name in names:

            if hasattr(obj, name):
                return getattr(
                    obj,
                    name,
                )

        return default

    def _get_nested(
        self,
        obj: Any,
        path: Sequence[str],
        *,
        fallback_names: Sequence[str] = (),
    ) -> Any:

        current = obj

        for name in path:

            current = self._get(
                current,
                name,
                default=None,
            )

            if current is None:
                break

        if current is not None:
            return current

        return self._get(
            obj,
            *fallback_names,
            default=None,
        )

    @staticmethod
    def _string(
        value: Any,
    ) -> str:

        if value is None:
            return "unknown"

        value = getattr(
            value,
            "value",
            value,
        )

        text = str(value).strip().lower()

        return text or "unknown"

    @staticmethod
    def _number(
        value: Any,
    ) -> float | None:

        if value is None:
            return None

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _normalize_probability(
        value: float,
    ) -> float:

        value = float(value)

        if value < 0:
            return 0.0

        if value <= 1:
            return value

        # Percentage-style value.
        if value <= 100:
            return value / 100.0

        return 1.0

    @staticmethod
    def _average(
        values: Sequence[int | float],
    ) -> float:

        if not values:
            return 0.0

        return round(
            float(mean(values)),
            4,
        )

    @staticmethod
    def _ratio(
        numerator: int | float,
        denominator: int | float,
    ) -> float:

        if not denominator:
            return 0.0

        return round(
            float(numerator)
            / float(denominator),
            4,
        )

    @staticmethod
    def _date(
        value: Any,
    ) -> date | None:

        if value is None:
            return None

        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        if isinstance(value, str):

            value = value.strip()

            if not value:
                return None

            # ISO date.
            try:
                return date.fromisoformat(
                    value[:10]
                )
            except ValueError:
                pass

            # Common legal-document formats.
            for fmt in (
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%m-%d-%Y",
                "%d %B %Y",
                "%B %d, %Y",
            ):

                try:
                    return datetime.strptime(
                        value,
                        fmt,
                    ).date()

                except ValueError:
                    continue

        return None