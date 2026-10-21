# backend/db/repositories/clause_repository.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence
from uuid import UUID

from sqlalchemy import (
    and_,
    asc,
    desc,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models.clause import Clause


@dataclass(slots=True)
class ClauseCreate:
    tenant_id: str
    document_id: str

    clause_type: str

    text: str

    organization_id: Optional[str] = None
    contract_id: Optional[str] = None

    document_version_id: Optional[str] = None

    parent_clause_id: Optional[str] = None

    clause_number: Optional[str] = None
    title: Optional[str] = None

    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    confidence: Optional[float] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    source: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ClauseUpdate:
    clause_type: Optional[str] = None

    text: Optional[str] = None

    parent_clause_id: Optional[str] = None

    clause_number: Optional[str] = None
    title: Optional[str] = None

    page_number: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    start_offset: Optional[int] = None
    end_offset: Optional[int] = None

    confidence: Optional[float] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    source: Optional[str] = None

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    metadata: Optional[
        dict[str, Any]
    ] = None


@dataclass(slots=True)
class ClauseFilter:
    tenant_id: str

    organization_id: Optional[str] = None

    document_id: Optional[str] = None

    document_ids: list[str] = field(
        default_factory=list
    )

    contract_id: Optional[str] = None

    contract_ids: list[str] = field(
        default_factory=list
    )

    document_version_id: Optional[str] = None

    clause_type: Optional[str] = None

    clause_types: list[str] = field(
        default_factory=list
    )

    risk_level: Optional[str] = None

    risk_levels: list[str] = field(
        default_factory=list
    )

    parent_clause_id: Optional[str] = None

    page_number: Optional[int] = None

    minimum_confidence: Optional[
        float
    ] = None

    minimum_risk_score: Optional[
        float
    ] = None

    maximum_risk_score: Optional[
        float
    ] = None

    source: Optional[str] = None

    model_name: Optional[str] = None

    search: Optional[str] = None

    created_from: Optional[
        datetime
    ] = None

    created_to: Optional[
        datetime
    ] = None


@dataclass(slots=True)
class ClausePage:
    items: list[Clause]

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
class ClauseStatistics:
    total: int = 0

    by_type: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    by_risk_level: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    average_confidence: Optional[
        float
    ] = None

    average_risk_score: Optional[
        float
    ] = None

    high_risk_count: int = 0
    critical_risk_count: int = 0

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "total": self.total,
            "by_type":
                self.by_type,
            "by_risk_level":
                self.by_risk_level,
            "average_confidence":
                self.average_confidence,
            "average_risk_score":
                self.average_risk_score,
            "high_risk_count":
                self.high_risk_count,
            "critical_risk_count":
                self.critical_risk_count,
        }


class ClauseRepositoryError(
    RuntimeError
):
    pass


class ClauseNotFoundError(
    ClauseRepositoryError
):
    pass


class ClauseRepository:
    """
    Async SQLAlchemy repository for legal clauses.

    Responsibilities
    ----------------
    - tenant-isolated clause persistence
    - document/contract clause lookup
    - clause-type filtering
    - risk filtering
    - page/section lookup
    - text search
    - bulk persistence
    - clause statistics
    - model provenance

    Security
    --------
    Every read/update/delete operation requires tenant_id.

    Authorization to a specific document/contract should still be
    enforced by the service/security layer before this repository is
    called.
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
        data: ClauseCreate,
        *,
        flush: bool = True,
    ) -> Clause:

        self._validate_create(
            data
        )

        clause = Clause(
            tenant_id=data.tenant_id,
            organization_id=(
                data.organization_id
            ),
            document_id=(
                data.document_id
            ),
            document_version_id=(
                data.document_version_id
            ),
            contract_id=(
                data.contract_id
            ),
            parent_clause_id=(
                data.parent_clause_id
            ),
            clause_type=(
                data.clause_type
            ),
            clause_number=(
                data.clause_number
            ),
            title=data.title,
            text=data.text,
            page_number=(
                data.page_number
            ),
            page_start=(
                data.page_start
            ),
            page_end=data.page_end,
            start_offset=(
                data.start_offset
            ),
            end_offset=(
                data.end_offset
            ),
            confidence=(
                data.confidence
            ),
            risk_level=(
                data.risk_level
            ),
            risk_score=(
                data.risk_score
            ),
            source=data.source,
            model_name=(
                data.model_name
            ),
            model_version=(
                data.model_version
            ),
            metadata=data.metadata,
        )

        self.session.add(
            clause
        )

        if flush:
            await self.session.flush()

        return clause

    async def create_many(
        self,
        clauses: Sequence[
            ClauseCreate
        ],
    ) -> list[Clause]:

        records: list[
            Clause
        ] = []

        for data in clauses:
            record = (
                await self.create(
                    data,
                    flush=False,
                )
            )

            records.append(
                record
            )

        if records:
            await self.session.flush()

        return records

    # =========================================================
    # GET BY ID
    # =========================================================

    async def get_by_id(
        self,
        clause_id: str | UUID,
        *,
        tenant_id: str,
    ) -> Optional[Clause]:

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(Clause)
            .where(
                Clause.id
                == clause_id,
                Clause.tenant_id
                == tenant_id,
            )
            .limit(1)
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        return (
            result
            .scalar_one_or_none()
        )

    async def require_by_id(
        self,
        clause_id: str | UUID,
        *,
        tenant_id: str,
    ) -> Clause:

        clause = (
            await self.get_by_id(
                clause_id,
                tenant_id=tenant_id,
            )
        )

        if clause is None:
            raise ClauseNotFoundError(
                f"Clause {clause_id} "
                "was not found."
            )

        return clause

    # =========================================================
    # DOCUMENT LOOKUPS
    # =========================================================

    async def list_for_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
        clause_types: Optional[
            Sequence[str]
        ] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ClausePage:

        return await self.list(
            ClauseFilter(
                tenant_id=tenant_id,
                document_id=(
                    document_id
                ),
                clause_types=list(
                    clause_types or []
                ),
            ),
            limit=limit,
            offset=offset,
            order_by="document",
        )

    async def list_for_document_version(
        self,
        *,
        tenant_id: str,
        document_version_id: str,
        limit: int = 500,
        offset: int = 0,
    ) -> ClausePage:

        return await self.list(
            ClauseFilter(
                tenant_id=tenant_id,
                document_version_id=(
                    document_version_id
                ),
            ),
            limit=limit,
            offset=offset,
            order_by="document",
        )

    # =========================================================
    # CONTRACT LOOKUPS
    # =========================================================

    async def list_for_contract(
        self,
        *,
        tenant_id: str,
        contract_id: str,
        clause_types: Optional[
            Sequence[str]
        ] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> ClausePage:

        return await self.list(
            ClauseFilter(
                tenant_id=tenant_id,
                contract_id=(
                    contract_id
                ),
                clause_types=list(
                    clause_types or []
                ),
            ),
            limit=limit,
            offset=offset,
            order_by="document",
        )

    # =========================================================
    # CLAUSE TYPE
    # =========================================================

    async def list_by_type(
        self,
        *,
        tenant_id: str,
        clause_type: str,
        document_id: Optional[
            str
        ] = None,
        contract_id: Optional[
            str
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ClausePage:

        return await self.list(
            ClauseFilter(
                tenant_id=tenant_id,
                document_id=(
                    document_id
                ),
                contract_id=(
                    contract_id
                ),
                clause_type=(
                    clause_type
                ),
            ),
            limit=limit,
            offset=offset,
        )

    # =========================================================
    # RISK LOOKUPS
    # =========================================================

    async def list_high_risk(
        self,
        *,
        tenant_id: str,
        organization_id: Optional[
            str
        ] = None,
        document_id: Optional[
            str
        ] = None,
        contract_id: Optional[
            str
        ] = None,
        minimum_risk_score: Optional[
            float
        ] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ClausePage:

        filters = ClauseFilter(
            tenant_id=tenant_id,
            organization_id=(
                organization_id
            ),
            document_id=(
                document_id
            ),
            contract_id=(
                contract_id
            ),
            risk_levels=[
                "high",
                "critical",
            ],
            minimum_risk_score=(
                minimum_risk_score
            ),
        )

        return await self.list(
            filters,
            limit=limit,
            offset=offset,
            order_by="risk",
        )

    # =========================================================
    # CHILD CLAUSES
    # =========================================================

    async def list_children(
        self,
        *,
        tenant_id: str,
        parent_clause_id: str,
    ) -> list[Clause]:

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(Clause)
            .where(
                Clause.tenant_id
                == tenant_id,
                Clause.parent_clause_id
                == parent_clause_id,
            )
            .order_by(
                asc(
                    Clause.start_offset
                ),
                asc(
                    Clause.clause_number
                ),
            )
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        return list(
            result.scalars().all()
        )

    # =========================================================
    # PAGE LOOKUP
    # =========================================================

    async def list_for_page(
        self,
        *,
        tenant_id: str,
        document_id: str,
        page_number: int,
    ) -> list[Clause]:

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(Clause)
            .where(
                Clause.tenant_id
                == tenant_id,
                Clause.document_id
                == document_id,
                or_(
                    Clause.page_number
                    == page_number,
                    and_(
                        Clause.page_start
                        <= page_number,
                        Clause.page_end
                        >= page_number,
                    ),
                ),
            )
            .order_by(
                asc(
                    Clause.start_offset
                )
            )
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        return list(
            result.scalars().all()
        )

    # =========================================================
    # SEARCH
    # =========================================================

    async def search_text(
        self,
        *,
        tenant_id: str,
        query: str,
        document_id: Optional[
            str
        ] = None,
        contract_id: Optional[
            str
        ] = None,
        clause_types: Optional[
            Sequence[str]
        ] = None,
        limit: int = 100,
    ) -> list[Clause]:

        self._require_tenant(
            tenant_id
        )

        query = query.strip()

        if not query:
            return []

        term = (
            "%"
            + self._escape_like(
                query
            )
            + "%"
        )

        conditions: list[
            Any
        ] = [
            Clause.tenant_id
            == tenant_id,
            or_(
                Clause.text.ilike(
                    term,
                    escape="\\",
                ),
                Clause.title.ilike(
                    term,
                    escape="\\",
                ),
                Clause.clause_number.ilike(
                    term,
                    escape="\\",
                ),
            ),
        ]

        if document_id:
            conditions.append(
                Clause.document_id
                == document_id
            )

        if contract_id:
            conditions.append(
                Clause.contract_id
                == contract_id
            )

        if clause_types:
            conditions.append(
                Clause.clause_type.in_(
                    list(
                        clause_types
                    )
                )
            )

        stmt = (
            select(Clause)
            .where(*conditions)
            .order_by(
                desc(
                    Clause.confidence
                )
            )
            .limit(
                self._validate_limit(
                    limit
                )
            )
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        return list(
            result.scalars().all()
        )

    # =========================================================
    # GENERIC LIST
    # =========================================================

    async def list(
        self,
        filters: ClauseFilter,
        *,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "newest",
    ) -> ClausePage:

        self._require_tenant(
            filters.tenant_id
        )

        limit = (
            self._validate_limit(
                limit
            )
        )

        if offset < 0:
            raise ValueError(
                "offset cannot be "
                "negative."
            )

        conditions = (
            self._conditions(
                filters
            )
        )

        stmt = (
            select(Clause)
            .where(
                *conditions
            )
        )

        stmt = (
            self._apply_order(
                stmt,
                order_by,
            )
        )

        stmt = (
            stmt
            .limit(limit)
            .offset(offset)
        )

        count_stmt = (
            select(
                func.count(
                    Clause.id
                )
            )
            .where(
                *conditions
            )
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        count_result = (
            await self.session.execute(
                count_stmt
            )
        )

        return ClausePage(
            items=list(
                result.scalars().all()
            ),
            total=int(
                count_result
                .scalar_one()
            ),
            limit=limit,
            offset=offset,
        )

    # =========================================================
    # UPDATE
    # =========================================================

    async def update(
        self,
        clause_id: str | UUID,
        data: ClauseUpdate,
        *,
        tenant_id: str,
    ) -> Clause:

        clause = (
            await self.require_by_id(
                clause_id,
                tenant_id=tenant_id,
            )
        )

        values = (
            self._update_values(
                data
            )
        )

        for key, value in (
            values.items()
        ):
            setattr(
                clause,
                key,
                value,
            )

        await self.session.flush()

        return clause

    async def update_risk(
        self,
        clause_id: str | UUID,
        *,
        tenant_id: str,
        risk_level: str,
        risk_score: Optional[
            float
        ] = None,
    ) -> Clause:

        if (
            risk_score is not None
            and not (
                0.0
                <= risk_score
                <= 1.0
            )
        ):
            raise ValueError(
                "risk_score must be "
                "between 0 and 1."
            )

        return await self.update(
            clause_id,
            ClauseUpdate(
                risk_level=(
                    risk_level
                ),
                risk_score=(
                    risk_score
                ),
            ),
            tenant_id=tenant_id,
        )

    # =========================================================
    # DELETE
    # =========================================================

    async def delete(
        self,
        clause_id: str | UUID,
        *,
        tenant_id: str,
    ) -> bool:

        clause = (
            await self.get_by_id(
                clause_id,
                tenant_id=tenant_id,
            )
        )

        if clause is None:
            return False

        await self.session.delete(
            clause
        )

        await self.session.flush()

        return True

    async def delete_for_document(
        self,
        *,
        tenant_id: str,
        document_id: str,
    ) -> int:
        """
        Used when rebuilding clause extraction for a document.

        The service layer should ensure related risks/obligations and
        vector indexes are rebuilt consistently.
        """

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(Clause)
            .where(
                Clause.tenant_id
                == tenant_id,
                Clause.document_id
                == document_id,
            )
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        clauses = list(
            result.scalars().all()
        )

        for clause in clauses:
            await self.session.delete(
                clause
            )

        if clauses:
            await self.session.flush()

        return len(clauses)

    # =========================================================
    # EXISTS / COUNT
    # =========================================================

    async def exists(
        self,
        clause_id: str | UUID,
        *,
        tenant_id: str,
    ) -> bool:

        self._require_tenant(
            tenant_id
        )

        stmt = (
            select(
                func.count(
                    Clause.id
                )
            )
            .where(
                Clause.id
                == clause_id,
                Clause.tenant_id
                == tenant_id,
            )
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        return (
            int(
                result.scalar_one()
            )
            > 0
        )

    async def count(
        self,
        filters: ClauseFilter,
    ) -> int:

        conditions = (
            self._conditions(
                filters
            )
        )

        stmt = (
            select(
                func.count(
                    Clause.id
                )
            )
            .where(
                *conditions
            )
        )

        result = (
            await self.session.execute(
                stmt
            )
        )

        return int(
            result.scalar_one()
        )

    # =========================================================
    # STATISTICS
    # =========================================================

    async def statistics(
        self,
        filters: ClauseFilter,
    ) -> ClauseStatistics:

        conditions = (
            self._conditions(
                filters
            )
        )

        total_stmt = (
            select(
                func.count(
                    Clause.id
                ),
                func.avg(
                    Clause.confidence
                ),
                func.avg(
                    Clause.risk_score
                ),
            )
            .where(
                *conditions
            )
        )

        result = (
            await self.session.execute(
                total_stmt
            )
        )

        row = result.one()

        total = int(
            row[0] or 0
        )

        avg_confidence = (
            float(row[1])
            if row[1] is not None
            else None
        )

        avg_risk = (
            float(row[2])
            if row[2] is not None
            else None
        )

        type_stmt = (
            select(
                Clause.clause_type,
                func.count(
                    Clause.id
                ),
            )
            .where(
                *conditions
            )
            .group_by(
                Clause.clause_type
            )
        )

        type_result = (
            await self.session.execute(
                type_stmt
            )
        )

        by_type = {
            str(
                clause_type
                or "unknown"
            ): int(count)
            for (
                clause_type,
                count,
            ) in type_result.all()
        }

        risk_stmt = (
            select(
                Clause.risk_level,
                func.count(
                    Clause.id
                ),
            )
            .where(
                *conditions
            )
            .group_by(
                Clause.risk_level
            )
        )

        risk_result = (
            await self.session.execute(
                risk_stmt
            )
        )

        by_risk_level = {
            str(
                risk_level
                or "unknown"
            ): int(count)
            for (
                risk_level,
                count,
            ) in risk_result.all()
        }

        return ClauseStatistics(
            total=total,
            by_type=by_type,
            by_risk_level=(
                by_risk_level
            ),
            average_confidence=(
                avg_confidence
            ),
            average_risk_score=(
                avg_risk
            ),
            high_risk_count=(
                by_risk_level.get(
                    "high",
                    0,
                )
            ),
            critical_risk_count=(
                by_risk_level.get(
                    "critical",
                    0,
                )
            ),
        )

    # =========================================================
    # FILTER BUILDER
    # =========================================================

    def _conditions(
        self,
        filters: ClauseFilter,
    ) -> list[Any]:

        self._require_tenant(
            filters.tenant_id
        )

        conditions: list[
            Any
        ] = [
            Clause.tenant_id
            == filters.tenant_id
        ]

        if filters.organization_id:
            conditions.append(
                Clause.organization_id
                == filters.organization_id
            )

        if filters.document_id:
            conditions.append(
                Clause.document_id
                == filters.document_id
            )

        if filters.document_ids:
            conditions.append(
                Clause.document_id.in_(
                    filters.document_ids
                )
            )

        if filters.contract_id:
            conditions.append(
                Clause.contract_id
                == filters.contract_id
            )

        if filters.contract_ids:
            conditions.append(
                Clause.contract_id.in_(
                    filters.contract_ids
                )
            )

        if filters.document_version_id:
            conditions.append(
                Clause.document_version_id
                == filters.document_version_id
            )

        if filters.clause_type:
            conditions.append(
                Clause.clause_type
                == filters.clause_type
            )

        if filters.clause_types:
            conditions.append(
                Clause.clause_type.in_(
                    filters.clause_types
                )
            )

        if filters.risk_level:
            conditions.append(
                Clause.risk_level
                == filters.risk_level
            )

        if filters.risk_levels:
            conditions.append(
                Clause.risk_level.in_(
                    filters.risk_levels
                )
            )

        if filters.parent_clause_id:
            conditions.append(
                Clause.parent_clause_id
                == filters.parent_clause_id
            )

        if filters.page_number is not None:
            conditions.append(
                or_(
                    Clause.page_number
                    == filters.page_number,
                    and_(
                        Clause.page_start
                        <= filters.page_number,
                        Clause.page_end
                        >= filters.page_number,
                    ),
                )
            )

        if (
            filters.minimum_confidence
            is not None
        ):
            conditions.append(
                Clause.confidence
                >= filters.minimum_confidence
            )

        if (
            filters.minimum_risk_score
            is not None
        ):
            conditions.append(
                Clause.risk_score
                >= filters.minimum_risk_score
            )

        if (
            filters.maximum_risk_score
            is not None
        ):
            conditions.append(
                Clause.risk_score
                <= filters.maximum_risk_score
            )

        if filters.source:
            conditions.append(
                Clause.source
                == filters.source
            )

        if filters.model_name:
            conditions.append(
                Clause.model_name
                == filters.model_name
            )

        if filters.created_from:
            conditions.append(
                Clause.created_at
                >= filters.created_from
            )

        if filters.created_to:
            conditions.append(
                Clause.created_at
                <= filters.created_to
            )

        if filters.search:
            term = (
                "%"
                + self._escape_like(
                    filters.search
                    .strip()
                )
                + "%"
            )

            conditions.append(
                or_(
                    Clause.text.ilike(
                        term,
                        escape="\\",
                    ),
                    Clause.title.ilike(
                        term,
                        escape="\\",
                    ),
                    Clause.clause_number.ilike(
                        term,
                        escape="\\",
                    ),
                    Clause.clause_type.ilike(
                        term,
                        escape="\\",
                    ),
                )
            )

        return conditions

    # =========================================================
    # ORDERING
    # =========================================================

    @staticmethod
    def _apply_order(
        stmt: Any,
        order_by: str,
    ) -> Any:

        if order_by == "document":
            return stmt.order_by(
                asc(
                    Clause.page_start
                ),
                asc(
                    Clause.page_number
                ),
                asc(
                    Clause.start_offset
                ),
                asc(
                    Clause.clause_number
                ),
            )

        if order_by == "risk":
            return stmt.order_by(
                desc(
                    Clause.risk_score
                ),
                desc(
                    Clause.confidence
                ),
            )

        if order_by == "confidence":
            return stmt.order_by(
                desc(
                    Clause.confidence
                )
            )

        if order_by == "oldest":
            return stmt.order_by(
                asc(
                    Clause.created_at
                )
            )

        return stmt.order_by(
            desc(
                Clause.created_at
            )
        )

    # =========================================================
    # UPDATE HELPERS
    # =========================================================

    @staticmethod
    def _update_values(
        data: ClauseUpdate,
    ) -> dict[str, Any]:

        values: dict[
            str,
            Any,
        ] = {}

        fields = (
            "clause_type",
            "text",
            "parent_clause_id",
            "clause_number",
            "title",
            "page_number",
            "page_start",
            "page_end",
            "start_offset",
            "end_offset",
            "confidence",
            "risk_level",
            "risk_score",
            "source",
            "model_name",
            "model_version",
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

        if (
            "confidence" in values
            and not (
                0.0
                <= values[
                    "confidence"
                ]
                <= 1.0
            )
        ):
            raise ValueError(
                "confidence must be "
                "between 0 and 1."
            )

        if (
            "risk_score" in values
            and not (
                0.0
                <= values[
                    "risk_score"
                ]
                <= 1.0
            )
        ):
            raise ValueError(
                "risk_score must be "
                "between 0 and 1."
            )

        return values

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_create(
        data: ClauseCreate,
    ) -> None:

        if not data.tenant_id:
            raise (
                ClauseRepositoryError(
                    "tenant_id is "
                    "required."
                )
            )

        if not data.document_id:
            raise (
                ClauseRepositoryError(
                    "document_id is "
                    "required."
                )
            )

        if not data.clause_type:
            raise (
                ClauseRepositoryError(
                    "clause_type is "
                    "required."
                )
            )

        if not data.text.strip():
            raise (
                ClauseRepositoryError(
                    "clause text cannot "
                    "be empty."
                )
            )

        if (
            data.confidence
            is not None
            and not (
                0.0
                <= data.confidence
                <= 1.0
            )
        ):
            raise ValueError(
                "confidence must be "
                "between 0 and 1."
            )

        if (
            data.risk_score
            is not None
            and not (
                0.0
                <= data.risk_score
                <= 1.0
            )
        ):
            raise ValueError(
                "risk_score must be "
                "between 0 and 1."
            )

        if (
            data.page_start
            is not None
            and data.page_end
            is not None
            and data.page_start
            > data.page_end
        ):
            raise ValueError(
                "page_start cannot be "
                "greater than page_end."
            )

        if (
            data.start_offset
            is not None
            and data.end_offset
            is not None
            and data.start_offset
            > data.end_offset
        ):
            raise ValueError(
                "start_offset cannot be "
                "greater than end_offset."
            )

    @staticmethod
    def _require_tenant(
        tenant_id: str,
    ) -> None:

        if not tenant_id:
            raise (
                ClauseRepositoryError(
                    "tenant_id is mandatory "
                    "for clause operations."
                )
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
            .replace(
                "\\",
                "\\\\",
            )
            .replace(
                "%",
                "\\%",
            )
            .replace(
                "_",
                "\\_",
            )
        )