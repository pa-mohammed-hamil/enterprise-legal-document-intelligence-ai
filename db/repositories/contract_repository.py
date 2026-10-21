# backend/db/repositories/contract_repository.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional, Sequence
from uuid import UUID

from sqlalchemy import asc, desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.contract import Contract


# ============================================================
# DTOs
# ============================================================

@dataclass(slots=True)
class ContractCreate:
    tenant_id: str
    organization_id: str
    document_id: str

    title: str

    contract_type: Optional[str] = None
    status: str = "draft"

    contract_number: Optional[str] = None

    counterparty_name: Optional[str] = None
    counterparty_id: Optional[str] = None

    owner_user_id: Optional[str] = None

    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None

    signed_date: Optional[date] = None

    renewal_date: Optional[date] = None
    notice_date: Optional[date] = None

    auto_renewal: bool = False

    renewal_term_days: Optional[int] = None
    notice_period_days: Optional[int] = None

    governing_law: Optional[str] = None
    jurisdiction: Optional[str] = None

    currency: Optional[str] = None
    contract_value: Optional[float] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    review_status: Optional[str] = None

    source: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ContractUpdate:
    title: Optional[str] = None

    contract_type: Optional[str] = None
    status: Optional[str] = None

    contract_number: Optional[str] = None

    counterparty_name: Optional[str] = None
    counterparty_id: Optional[str] = None

    owner_user_id: Optional[str] = None

    effective_date: Optional[date] = None
    expiry_date: Optional[date] = None
    signed_date: Optional[date] = None

    renewal_date: Optional[date] = None
    notice_date: Optional[date] = None

    auto_renewal: Optional[bool] = None

    renewal_term_days: Optional[int] = None
    notice_period_days: Optional[int] = None

    governing_law: Optional[str] = None
    jurisdiction: Optional[str] = None

    currency: Optional[str] = None
    contract_value: Optional[float] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    review_status: Optional[str] = None

    source: Optional[str] = None

    metadata: Optional[
        dict[str, Any]
    ] = None


@dataclass(slots=True)
class ContractFilter:
    tenant_id: str

    organization_id: Optional[str] = None

    document_id: Optional[str] = None
    document_ids: list[str] = field(
        default_factory=list
    )

    contract_ids: list[str] = field(
        default_factory=list
    )

    contract_type: Optional[str] = None
    contract_types: list[str] = field(
        default_factory=list
    )

    status: Optional[str] = None
    statuses: list[str] = field(
        default_factory=list
    )

    counterparty_name: Optional[str] = None
    counterparty_id: Optional[str] = None

    owner_user_id: Optional[str] = None

    risk_level: Optional[str] = None
    risk_levels: list[str] = field(
        default_factory=list
    )

    minimum_risk_score: Optional[float] = None
    maximum_risk_score: Optional[float] = None

    review_status: Optional[str] = None

    auto_renewal: Optional[bool] = None

    effective_from: Optional[date] = None
    effective_to: Optional[date] = None

    expiry_from: Optional[date] = None
    expiry_to: Optional[date] = None

    renewal_from: Optional[date] = None
    renewal_to: Optional[date] = None

    notice_from: Optional[date] = None
    notice_to: Optional[date] = None

    governing_law: Optional[str] = None
    jurisdiction: Optional[str] = None

    minimum_value: Optional[float] = None
    maximum_value: Optional[float] = None

    currency: Optional[str] = None

    search: Optional[str] = None

    created_from: Optional[datetime] = None
    created_to: Optional[datetime] = None


@dataclass(slots=True)
class ContractPage:
    items: list[Contract]

    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return (
            self.offset
            + len(self.items)
            < self.total
        )


@dataclass(slots=True)
class ContractStatistics:
    total: int = 0

    active: int = 0
    draft: int = 0
    expired: int = 0
    terminated: int = 0

    auto_renewing: int = 0

    high_risk: int = 0
    critical_risk: int = 0

    total_contract_value: float = 0.0
    average_risk_score: Optional[float] = None

    by_status: dict[str, int] = field(
        default_factory=dict
    )

    by_type: dict[str, int] = field(
        default_factory=dict
    )

    by_risk_level: dict[str, int] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "active": self.active,
            "draft": self.draft,
            "expired": self.expired,
            "terminated": self.terminated,
            "auto_renewing": self.auto_renewing,
            "high_risk": self.high_risk,
            "critical_risk": self.critical_risk,
            "total_contract_value":
                self.total_contract_value,
            "average_risk_score":
                self.average_risk_score,
            "by_status": self.by_status,
            "by_type": self.by_type,
            "by_risk_level":
                self.by_risk_level,
        }


# ============================================================
# Exceptions
# ============================================================

class ContractRepositoryError(RuntimeError):
    pass


class ContractNotFoundError(
    ContractRepositoryError
):
    pass


# ============================================================
# Repository
# ============================================================

class ContractRepository:
    """
    Async persistence layer for enterprise contracts.

    Responsibilities:
        - CRUD
        - tenant isolation
        - organization filtering
        - contract lifecycle lookup
        - expiry lookup
        - renewal lookup
        - notice lookup
        - counterparty lookup
        - risk lookup
        - review-status lookup
        - contract value analytics
        - dashboard statistics

    Business logic such as renewal decisions, risk interpretation,
    expiry escalation and legal recommendations should remain in
    service/agent layers.
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

    # ========================================================
    # CREATE
    # ========================================================

    async def create(
        self,
        data: ContractCreate,
        *,
        flush: bool = True,
    ) -> Contract:

        self._validate_create(data)

        contract = Contract(
            tenant_id=data.tenant_id,
            organization_id=data.organization_id,
            document_id=data.document_id,

            title=data.title,

            contract_type=data.contract_type,
            status=data.status,

            contract_number=data.contract_number,

            counterparty_name=(
                data.counterparty_name
            ),
            counterparty_id=(
                data.counterparty_id
            ),

            owner_user_id=data.owner_user_id,

            effective_date=data.effective_date,
            expiry_date=data.expiry_date,
            signed_date=data.signed_date,

            renewal_date=data.renewal_date,
            notice_date=data.notice_date,

            auto_renewal=data.auto_renewal,

            renewal_term_days=(
                data.renewal_term_days
            ),
            notice_period_days=(
                data.notice_period_days
            ),

            governing_law=data.governing_law,
            jurisdiction=data.jurisdiction,

            currency=data.currency,
            contract_value=data.contract_value,

            risk_level=data.risk_level,
            risk_score=data.risk_score,

            review_status=data.review_status,

            source=data.source,

            metadata=data.metadata,
        )

        self.session.add(contract)

        if flush:
            await self.session.flush()

        return contract

    async def create_many(
        self,
        contracts: Sequence[
            ContractCreate
        ],
    ) -> list[Contract]:

        records: list[Contract] = []

        for data in contracts:
            record = await self.create(
                data,
                flush=False,
            )

            records.append(record)

        if records:
            await self.session.flush()

        return records

    # ========================================================
    # GET
    # ========================================================

    async def get_by_id(
        self,
        contract_id: str | UUID,
        *,
        tenant_id: str,
    ) -> Optional[Contract]:

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(Contract)
            .where(
                Contract.id == contract_id,
                Contract.tenant_id
                == tenant_id,
            )
            .limit(1)
        )

        result = await self.session.execute(
            stmt
        )

        return result.scalar_one_or_none()

    async def require_by_id(
        self,
        contract_id: str | UUID,
        *,
        tenant_id: str,
    ) -> Contract:

        contract = await self.get_by_id(
            contract_id,
            tenant_id=tenant_id,
        )

        if contract is None:
            raise ContractNotFoundError(
                f"Contract {contract_id} "
                "was not found."
            )

        return contract

    async def get_by_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> Optional[Contract]:

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(Contract)
            .where(
                Contract.tenant_id
                == tenant_id,
                Contract.document_id
                == document_id,
            )
            .limit(1)
        )

        result = await self.session.execute(
            stmt
        )

        return result.scalar_one_or_none()

    async def get_by_contract_number(
        self,
        *,
        tenant_id: str,
        contract_number: str,
        organization_id: Optional[
            str
        ] = None,
    ) -> Optional[Contract]:

        self._require_tenant(
            tenant_id
        )

        conditions = [
            Contract.tenant_id
            == tenant_id,
            Contract.contract_number
            == contract_number,
        ]

        if organization_id:
            conditions.append(
                Contract.organization_id
                == organization_id
            )

        stmt = (
            select(Contract)
            .where(*conditions)
            .limit(1)
        )

        result = await self.session.execute(
            stmt
        )

        return result.scalar_one_or_none()

    # ========================================================
    # GENERIC LIST
    # ========================================================

    async def list(
        self,
        filters: ContractFilter,
        *,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "newest",
    ) -> ContractPage:

        self._require_tenant(
            filters.tenant_id
        )

        limit = self._validate_limit(
            limit
        )

        if offset < 0:
            raise ValueError(
                "offset cannot be negative."
            )

        conditions = self._conditions(
            filters
        )

        stmt = (
            select(Contract)
            .where(*conditions)
        )

        stmt = self._apply_order(
            stmt,
            order_by,
        )

        stmt = (
            stmt
            .limit(limit)
            .offset(offset)
        )

        count_stmt = (
            select(
                func.count(
                    Contract.id
                )
            )
            .where(*conditions)
        )

        result = await self.session.execute(
            stmt
        )

        count_result = (
            await self.session.execute(
                count_stmt
            )
        )

        return ContractPage(
            items=list(
                result.scalars().all()
            ),
            total=int(
                count_result.scalar_one()
            ),
            limit=limit,
            offset=offset,
        )

    # ========================================================
    # ORGANIZATION
    # ========================================================

    async def list_for_organization(
        self,
        *,
        tenant_id: str,
        organization_id: str,
        statuses: Optional[
            Sequence[str]
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ContractPage:

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                statuses=list(
                    statuses or []
                ),
            ),
            limit=limit,
            offset=offset,
        )

    # ========================================================
    # COUNTERPARTY
    # ========================================================

    async def list_for_counterparty(
        self,
        *,
        tenant_id: str,
        counterparty_name: Optional[
            str
        ] = None,
        counterparty_id: Optional[
            str
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ContractPage:

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                counterparty_name=(
                    counterparty_name
                ),
                counterparty_id=(
                    counterparty_id
                ),
            ),
            limit=limit,
            offset=offset,
        )

    # ========================================================
    # OWNER
    # ========================================================

    async def list_for_owner(
        self,
        *,
        tenant_id: str,
        owner_user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> ContractPage:

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                owner_user_id=(
                    owner_user_id
                ),
            ),
            limit=limit,
            offset=offset,
        )

    # ========================================================
    # EXPIRY
    # ========================================================

    async def list_expiring_between(
        self,
        *,
        tenant_id: str,
        start_date: date,
        end_date: date,
        organization_id: Optional[
            str
        ] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ContractPage:

        if start_date > end_date:
            raise ValueError(
                "start_date cannot be "
                "after end_date."
            )

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                expiry_from=start_date,
                expiry_to=end_date,
            ),
            limit=limit,
            offset=offset,
            order_by="expiry",
        )

    async def list_expired(
        self,
        *,
        tenant_id: str,
        as_of: Optional[date] = None,
        organization_id: Optional[
            str
        ] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ContractPage:

        as_of = as_of or date.today()

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                expiry_to=as_of,
            ),
            limit=limit,
            offset=offset,
            order_by="expiry",
        )

    # ========================================================
    # RENEWALS
    # ========================================================

    async def list_renewing_between(
        self,
        *,
        tenant_id: str,
        start_date: date,
        end_date: date,
        organization_id: Optional[
            str
        ] = None,
        auto_renewal: Optional[
            bool
        ] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ContractPage:

        if start_date > end_date:
            raise ValueError(
                "start_date cannot be "
                "after end_date."
            )

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                renewal_from=start_date,
                renewal_to=end_date,
                auto_renewal=(
                    auto_renewal
                ),
            ),
            limit=limit,
            offset=offset,
            order_by="renewal",
        )

    async def list_auto_renewing(
        self,
        *,
        tenant_id: str,
        organization_id: Optional[
            str
        ] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ContractPage:

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                auto_renewal=True,
            ),
            limit=limit,
            offset=offset,
            order_by="renewal",
        )

    # ========================================================
    # NOTICE DEADLINES
    # ========================================================

    async def list_notice_deadlines(
        self,
        *,
        tenant_id: str,
        start_date: date,
        end_date: date,
        organization_id: Optional[
            str
        ] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ContractPage:

        if start_date > end_date:
            raise ValueError(
                "start_date cannot be "
                "after end_date."
            )

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                notice_from=start_date,
                notice_to=end_date,
            ),
            limit=limit,
            offset=offset,
            order_by="notice",
        )

    # ========================================================
    # RISK
    # ========================================================

    async def list_high_risk(
        self,
        *,
        tenant_id: str,
        organization_id: Optional[
            str
        ] = None,
        minimum_risk_score: Optional[
            float
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ContractPage:

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                risk_levels=[
                    "high",
                    "critical",
                ],
                minimum_risk_score=(
                    minimum_risk_score
                ),
            ),
            limit=limit,
            offset=offset,
            order_by="risk",
        )

    # ========================================================
    # REVIEW
    # ========================================================

    async def list_pending_review(
        self,
        *,
        tenant_id: str,
        organization_id: Optional[
            str
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ContractPage:

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                review_status="pending",
            ),
            limit=limit,
            offset=offset,
        )

    # ========================================================
    # SEARCH
    # ========================================================

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        organization_id: Optional[
            str
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ContractPage:

        return await self.list(
            ContractFilter(
                tenant_id=tenant_id,
                organization_id=(
                    organization_id
                ),
                search=query,
            ),
            limit=limit,
            offset=offset,
        )

    # ========================================================
    # UPDATE
    # ========================================================

    async def update(
        self,
        contract_id: str | UUID,
        data: ContractUpdate,
        *,
        tenant_id: str,
    ) -> Contract:

        contract = (
            await self.require_by_id(
                contract_id,
                tenant_id=tenant_id,
            )
        )

        values = self._update_values(
            data
        )

        for key, value in (
            values.items()
        ):
            setattr(
                contract,
                key,
                value,
            )

        await self.session.flush()

        return contract

    async def update_status(
        self,
        contract_id: str | UUID,
        *,
        tenant_id: str,
        status: str,
    ) -> Contract:

        if not status.strip():
            raise ValueError(
                "status cannot be empty."
            )

        return await self.update(
            contract_id,
            ContractUpdate(
                status=status.strip()
            ),
            tenant_id=tenant_id,
        )

    async def update_risk(
        self,
        contract_id: str | UUID,
        *,
        tenant_id: str,
        risk_level: str,
        risk_score: Optional[
            float
        ] = None,
    ) -> Contract:

        self._validate_score(
            risk_score
        )

        return await self.update(
            contract_id,
            ContractUpdate(
                risk_level=risk_level,
                risk_score=risk_score,
            ),
            tenant_id=tenant_id,
        )

    async def update_review_status(
        self,
        contract_id: str | UUID,
        *,
        tenant_id: str,
        review_status: str,
    ) -> Contract:

        return await self.update(
            contract_id,
            ContractUpdate(
                review_status=(
                    review_status
                )
            ),
            tenant_id=tenant_id,
        )

    # ========================================================
    # DELETE
    # ========================================================

    async def delete(
        self,
        contract_id: str | UUID,
        *,
        tenant_id: str,
    ) -> bool:

        contract = await self.get_by_id(
            contract_id,
            tenant_id=tenant_id,
        )

        if contract is None:
            return False

        await self.session.delete(
            contract
        )

        await self.session.flush()

        return True

    # ========================================================
    # EXISTS / COUNT
    # ========================================================

    async def exists(
        self,
        contract_id: str | UUID,
        *,
        tenant_id: str,
    ) -> bool:

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(
                func.count(
                    Contract.id
                )
            )
            .where(
                Contract.id
                == contract_id,
                Contract.tenant_id
                == tenant_id,
            )
        )

        result = await self.session.execute(
            stmt
        )

        return (
            int(
                result.scalar_one()
            )
            > 0
        )

    async def count(
        self,
        filters: ContractFilter,
    ) -> int:

        conditions = self._conditions(
            filters
        )

        stmt = (
            select(
                func.count(
                    Contract.id
                )
            )
            .where(*conditions)
        )

        result = await self.session.execute(
            stmt
        )

        return int(
            result.scalar_one()
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    async def statistics(
        self,
        filters: ContractFilter,
    ) -> ContractStatistics:

        conditions = self._conditions(
            filters
        )

        summary_stmt = (
            select(
                func.count(
                    Contract.id
                ),
                func.sum(
                    Contract.contract_value
                ),
                func.avg(
                    Contract.risk_score
                ),
            )
            .where(*conditions)
        )

        result = await self.session.execute(
            summary_stmt
        )

        row = result.one()

        total = int(row[0] or 0)

        total_value = float(
            row[1] or 0.0
        )

        average_risk = (
            float(row[2])
            if row[2] is not None
            else None
        )

        status_stmt = (
            select(
                Contract.status,
                func.count(
                    Contract.id
                ),
            )
            .where(*conditions)
            .group_by(
                Contract.status
            )
        )

        status_result = (
            await self.session.execute(
                status_stmt
            )
        )

        by_status = {
            str(status or "unknown"):
                int(count)
            for status, count
            in status_result.all()
        }

        type_stmt = (
            select(
                Contract.contract_type,
                func.count(
                    Contract.id
                ),
            )
            .where(*conditions)
            .group_by(
                Contract.contract_type
            )
        )

        type_result = (
            await self.session.execute(
                type_stmt
            )
        )

        by_type = {
            str(contract_type or "unknown"):
                int(count)
            for contract_type, count
            in type_result.all()
        }

        risk_stmt = (
            select(
                Contract.risk_level,
                func.count(
                    Contract.id
                ),
            )
            .where(*conditions)
            .group_by(
                Contract.risk_level
            )
        )

        risk_result = (
            await self.session.execute(
                risk_stmt
            )
        )

        by_risk = {
            str(risk_level or "unknown"):
                int(count)
            for risk_level, count
            in risk_result.all()
        }

        auto_stmt = (
            select(
                func.count(
                    Contract.id
                )
            )
            .where(
                *conditions,
                Contract.auto_renewal
                .is_(True),
            )
        )

        auto_result = (
            await self.session.execute(
                auto_stmt
            )
        )

        auto_renewing = int(
            auto_result.scalar_one()
        )

        return ContractStatistics(
            total=total,

            active=by_status.get(
                "active",
                0,
            ),
            draft=by_status.get(
                "draft",
                0,
            ),
            expired=by_status.get(
                "expired",
                0,
            ),
            terminated=by_status.get(
                "terminated",
                0,
            ),

            auto_renewing=(
                auto_renewing
            ),

            high_risk=by_risk.get(
                "high",
                0,
            ),
            critical_risk=by_risk.get(
                "critical",
                0,
            ),

            total_contract_value=(
                total_value
            ),

            average_risk_score=(
                average_risk
            ),

            by_status=by_status,
            by_type=by_type,
            by_risk_level=by_risk,
        )

    # ========================================================
    # FILTER CONDITIONS
    # ========================================================

    def _conditions(
        self,
        filters: ContractFilter,
    ) -> list[Any]:

        self._require_tenant(
            filters.tenant_id
        )

        conditions: list[Any] = [
            Contract.tenant_id
            == filters.tenant_id
        ]

        if filters.organization_id:
            conditions.append(
                Contract.organization_id
                == filters.organization_id
            )

        if filters.document_id:
            conditions.append(
                Contract.document_id
                == filters.document_id
            )

        if filters.document_ids:
            conditions.append(
                Contract.document_id.in_(
                    filters.document_ids
                )
            )

        if filters.contract_ids:
            conditions.append(
                Contract.id.in_(
                    filters.contract_ids
                )
            )

        if filters.contract_type:
            conditions.append(
                Contract.contract_type
                == filters.contract_type
            )

        if filters.contract_types:
            conditions.append(
                Contract.contract_type.in_(
                    filters.contract_types
                )
            )

        if filters.status:
            conditions.append(
                Contract.status
                == filters.status
            )

        if filters.statuses:
            conditions.append(
                Contract.status.in_(
                    filters.statuses
                )
            )

        if filters.counterparty_name:
            conditions.append(
                Contract.counterparty_name
                == filters.counterparty_name
            )

        if filters.counterparty_id:
            conditions.append(
                Contract.counterparty_id
                == filters.counterparty_id
            )

        if filters.owner_user_id:
            conditions.append(
                Contract.owner_user_id
                == filters.owner_user_id
            )

        if filters.risk_level:
            conditions.append(
                Contract.risk_level
                == filters.risk_level
            )

        if filters.risk_levels:
            conditions.append(
                Contract.risk_level.in_(
                    filters.risk_levels
                )
            )

        if (
            filters.minimum_risk_score
            is not None
        ):
            conditions.append(
                Contract.risk_score
                >= filters.minimum_risk_score
            )

        if (
            filters.maximum_risk_score
            is not None
        ):
            conditions.append(
                Contract.risk_score
                <= filters.maximum_risk_score
            )

        if filters.review_status:
            conditions.append(
                Contract.review_status
                == filters.review_status
            )

        if filters.auto_renewal is not None:
            conditions.append(
                Contract.auto_renewal
                .is_(
                    filters.auto_renewal
                )
            )

        if filters.effective_from:
            conditions.append(
                Contract.effective_date
                >= filters.effective_from
            )

        if filters.effective_to:
            conditions.append(
                Contract.effective_date
                <= filters.effective_to
            )

        if filters.expiry_from:
            conditions.append(
                Contract.expiry_date
                >= filters.expiry_from
            )

        if filters.expiry_to:
            conditions.append(
                Contract.expiry_date
                <= filters.expiry_to
            )

        if filters.renewal_from:
            conditions.append(
                Contract.renewal_date
                >= filters.renewal_from
            )

        if filters.renewal_to:
            conditions.append(
                Contract.renewal_date
                <= filters.renewal_to
            )

        if filters.notice_from:
            conditions.append(
                Contract.notice_date
                >= filters.notice_from
            )

        if filters.notice_to:
            conditions.append(
                Contract.notice_date
                <= filters.notice_to
            )

        if filters.governing_law:
            conditions.append(
                Contract.governing_law
                == filters.governing_law
            )

        if filters.jurisdiction:
            conditions.append(
                Contract.jurisdiction
                == filters.jurisdiction
            )

        if filters.minimum_value is not None:
            conditions.append(
                Contract.contract_value
                >= filters.minimum_value
            )

        if filters.maximum_value is not None:
            conditions.append(
                Contract.contract_value
                <= filters.maximum_value
            )

        if filters.currency:
            conditions.append(
                Contract.currency
                == filters.currency
            )

        if filters.created_from:
            conditions.append(
                Contract.created_at
                >= filters.created_from
            )

        if filters.created_to:
            conditions.append(
                Contract.created_at
                <= filters.created_to
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
                    Contract.title.ilike(
                        term,
                        escape="\\",
                    ),
                    Contract.contract_number.ilike(
                        term,
                        escape="\\",
                    ),
                    Contract.counterparty_name.ilike(
                        term,
                        escape="\\",
                    ),
                    Contract.contract_type.ilike(
                        term,
                        escape="\\",
                    ),
                    Contract.governing_law.ilike(
                        term,
                        escape="\\",
                    ),
                    Contract.jurisdiction.ilike(
                        term,
                        escape="\\",
                    ),
                )
            )

        return conditions

    # ========================================================
    # ORDERING
    # ========================================================

    @staticmethod
    def _apply_order(
        stmt: Any,
        order_by: str,
    ) -> Any:

        if order_by == "expiry":
            return stmt.order_by(
                asc(
                    Contract.expiry_date
                ),
                asc(
                    Contract.title
                ),
            )

        if order_by == "renewal":
            return stmt.order_by(
                asc(
                    Contract.renewal_date
                ),
                asc(
                    Contract.title
                ),
            )

        if order_by == "notice":
            return stmt.order_by(
                asc(
                    Contract.notice_date
                ),
                asc(
                    Contract.title
                ),
            )

        if order_by == "risk":
            return stmt.order_by(
                desc(
                    Contract.risk_score
                ),
                desc(
                    Contract.created_at
                ),
            )

        if order_by == "value":
            return stmt.order_by(
                desc(
                    Contract.contract_value
                )
            )

        if order_by == "title":
            return stmt.order_by(
                asc(
                    Contract.title
                )
            )

        if order_by == "oldest":
            return stmt.order_by(
                asc(
                    Contract.created_at
                )
            )

        return stmt.order_by(
            desc(
                Contract.created_at
            )
        )

    # ========================================================
    # UPDATE MAPPING
    # ========================================================

    def _update_values(
        self,
        data: ContractUpdate,
    ) -> dict[str, Any]:

        values: dict[str, Any] = {}

        fields = (
            "title",
            "contract_type",
            "status",
            "contract_number",
            "counterparty_name",
            "counterparty_id",
            "owner_user_id",
            "effective_date",
            "expiry_date",
            "signed_date",
            "renewal_date",
            "notice_date",
            "auto_renewal",
            "renewal_term_days",
            "notice_period_days",
            "governing_law",
            "jurisdiction",
            "currency",
            "contract_value",
            "risk_level",
            "risk_score",
            "review_status",
            "source",
            "metadata",
        )

        for field_name in fields:
            value = getattr(
                data,
                field_name,
            )

            if value is not None:
                values[
                    field_name
                ] = value

        if "risk_score" in values:
            self._validate_score(
                values["risk_score"]
            )

        if (
            "contract_value" in values
            and values[
                "contract_value"
            ] < 0
        ):
            raise ValueError(
                "contract_value cannot "
                "be negative."
            )

        if (
            "renewal_term_days"
            in values
            and values[
                "renewal_term_days"
            ] < 0
        ):
            raise ValueError(
                "renewal_term_days cannot "
                "be negative."
            )

        if (
            "notice_period_days"
            in values
            and values[
                "notice_period_days"
            ] < 0
        ):
            raise ValueError(
                "notice_period_days cannot "
                "be negative."
            )

        return values

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_create(
        self,
        data: ContractCreate,
    ) -> None:

        self._require_tenant(
            data.tenant_id
        )

        if not data.organization_id:
            raise ContractRepositoryError(
                "organization_id is required."
            )

        if not data.document_id:
            raise ContractRepositoryError(
                "document_id is required."
            )

        if not data.title.strip():
            raise ContractRepositoryError(
                "contract title cannot "
                "be empty."
            )

        self._validate_score(
            data.risk_score
        )

        if (
            data.contract_value
            is not None
            and data.contract_value < 0
        ):
            raise ValueError(
                "contract_value cannot "
                "be negative."
            )

        if (
            data.renewal_term_days
            is not None
            and data.renewal_term_days < 0
        ):
            raise ValueError(
                "renewal_term_days cannot "
                "be negative."
            )

        if (
            data.notice_period_days
            is not None
            and data.notice_period_days < 0
        ):
            raise ValueError(
                "notice_period_days cannot "
                "be negative."
            )

        if (
            data.effective_date
            and data.expiry_date
            and data.expiry_date
            < data.effective_date
        ):
            raise ValueError(
                "expiry_date cannot be "
                "before effective_date."
            )

    @staticmethod
    def _validate_score(
        score: Optional[float],
    ) -> None:

        if score is None:
            return

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "risk_score must be "
                "between 0 and 1."
            )

    @staticmethod
    def _require_tenant(
        tenant_id: str,
    ) -> None:

        if not tenant_id:
            raise ContractRepositoryError(
                "tenant_id is mandatory "
                "for contract operations."
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
    def _escape_like(
        value: str,
    ) -> str:

        return (
            value
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    @staticmethod
    def utcnow() -> datetime:
        return datetime.now(
            timezone.utc
        )