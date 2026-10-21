# backend/api/routes/clauses.py

from __future__ import annotations

from datetime import datetime
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
from pydantic import BaseModel, ConfigDict, Field

from backend.db.repositories.clause_repository import (
    ClauseCreate,
    ClauseFilter,
    ClauseNotFoundError,
    ClauseRepository,
    ClauseUpdate,
)


router = APIRouter(
    prefix="/clauses",
    tags=["Clauses"],
)


# ============================================================
# DEPENDENCIES
# ============================================================

async def get_clause_repository() -> ClauseRepository:
    """
    Replace with the real provider from backend.dependencies:

        async def get_clause_repository(
            session: AsyncSession = Depends(get_db),
        ) -> ClauseRepository:
            return ClauseRepository(session)
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Clause repository dependency is not configured.",
    )


async def get_clause_service() -> Any:
    """
    Optional service layer used for:
        - clause classification
        - risk analysis
        - re-analysis
        - model provenance
        - audit events
        - agent orchestration
    """
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Clause service dependency is not configured.",
    )


async def get_current_user() -> Any:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication dependency is not configured.",
    )


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ClauseCreateRequest(BaseModel):
    document_id: UUID

    contract_id: Optional[UUID] = None
    document_version_id: Optional[UUID] = None
    parent_clause_id: Optional[UUID] = None

    clause_type: str = Field(
        min_length=1,
        max_length=100,
    )

    clause_number: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    title: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    text: str = Field(
        min_length=1,
        max_length=500_000,
    )

    page_number: Optional[int] = Field(
        default=None,
        ge=1,
    )

    page_start: Optional[int] = Field(
        default=None,
        ge=1,
    )

    page_end: Optional[int] = Field(
        default=None,
        ge=1,
    )

    start_offset: Optional[int] = Field(
        default=None,
        ge=0,
    )

    end_offset: Optional[int] = Field(
        default=None,
        ge=0,
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    risk_level: Optional[str] = None

    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    source: str = Field(
        default="manual",
        max_length=100,
    )

    model_name: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    model_version: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ClauseUpdateRequest(BaseModel):
    clause_type: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    clause_number: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    title: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    text: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=500_000,
    )

    parent_clause_id: Optional[UUID] = None

    page_number: Optional[int] = Field(
        default=None,
        ge=1,
    )

    page_start: Optional[int] = Field(
        default=None,
        ge=1,
    )

    page_end: Optional[int] = Field(
        default=None,
        ge=1,
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    risk_level: Optional[str] = None

    risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    metadata: Optional[dict[str, Any]] = None


class ClauseRiskUpdateRequest(BaseModel):
    risk_level: str = Field(
        min_length=1,
        max_length=50,
    )

    risk_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    reason: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class ClauseAnalyzeRequest(BaseModel):
    run_classification: bool = True
    run_risk_analysis: bool = True

    model: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    force: bool = False


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class ClauseResponse(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        extra="ignore",
    )

    id: str

    tenant_id: str
    organization_id: Optional[str] = None

    document_id: str
    contract_id: Optional[str] = None
    document_version_id: Optional[str] = None

    parent_clause_id: Optional[str] = None

    clause_type: str

    clause_number: Optional[str] = None
    title: Optional[str] = None

    text: str

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

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ClausePageResponse(BaseModel):
    items: list[ClauseResponse]

    total: int
    limit: int
    offset: int

    has_more: bool


class ClauseStatisticsResponse(BaseModel):
    total: int = 0

    by_type: dict[str, int] = Field(
        default_factory=dict,
    )

    by_risk: dict[str, int] = Field(
        default_factory=dict,
    )

    average_confidence: Optional[float] = None
    average_risk_score: Optional[float] = None

    high_risk: int = 0
    critical: int = 0

    model_config = ConfigDict(
        extra="allow",
    )


class ClauseAnalysisResponse(BaseModel):
    clause_id: str

    clause_type: Optional[str] = None

    classification_confidence: Optional[float] = None

    risk_level: Optional[str] = None
    risk_score: Optional[float] = None

    risk_reasons: list[str] = Field(
        default_factory=list,
    )

    model_name: Optional[str] = None
    model_version: Optional[str] = None

    analyzed_at: datetime


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
        return obj.get(name, default)

    return getattr(obj, name, default)


def _optional_str(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    return str(value)


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


def _tenant_id(
    current_user: Any,
) -> str:

    tenant_id = _attr(
        current_user,
        "tenant_id",
    )

    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tenant context is required.",
        )

    return str(tenant_id)


def _organization_id(
    current_user: Any,
) -> Optional[str]:

    value = _attr(
        current_user,
        "organization_id",
    )

    return (
        str(value)
        if value
        else None
    )


def _user_id(
    current_user: Any,
) -> str:

    value = _attr(
        current_user,
        "id",
    )

    if not value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity is required.",
        )

    return str(value)


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

    return (
        str(value)
        if value
        else None
    )


def _metadata(
    clause: Any,
) -> dict[str, Any]:

    return (
        _attr(
            clause,
            "metadata_json",
        )
        or _attr(
            clause,
            "clause_metadata",
        )
        or {}
    )


def _serialize_clause(
    clause: Any,
) -> dict[str, Any]:

    return {
        "id": str(
            _attr(
                clause,
                "id",
                "",
            )
        ),

        "tenant_id": str(
            _attr(
                clause,
                "tenant_id",
                "",
            )
        ),

        "organization_id": _optional_str(
            _attr(
                clause,
                "organization_id",
            )
        ),

        "document_id": str(
            _attr(
                clause,
                "document_id",
                "",
            )
        ),

        "contract_id": _optional_str(
            _attr(
                clause,
                "contract_id",
            )
        ),

        "document_version_id": _optional_str(
            _attr(
                clause,
                "document_version_id",
            )
        ),

        "parent_clause_id": _optional_str(
            _attr(
                clause,
                "parent_clause_id",
            )
        ),

        "clause_type": str(
            _enum_value(
                _attr(
                    clause,
                    "clause_type",
                    "unknown",
                )
            )
        ),

        "clause_number": _attr(
            clause,
            "clause_number",
        ),

        "title": _attr(
            clause,
            "title",
        ),

        "text": str(
            _attr(
                clause,
                "text",
                "",
            )
        ),

        "page_number": _attr(
            clause,
            "page_number",
        ),

        "page_start": _attr(
            clause,
            "page_start",
        ),

        "page_end": _attr(
            clause,
            "page_end",
        ),

        "start_offset": _attr(
            clause,
            "start_offset",
        ),

        "end_offset": _attr(
            clause,
            "end_offset",
        ),

        "confidence": _attr(
            clause,
            "confidence",
        ),

        "risk_level": _optional_str(
            _enum_value(
                _attr(
                    clause,
                    "risk_level",
                )
            )
        ),

        "risk_score": _attr(
            clause,
            "risk_score",
        ),

        "source": _optional_str(
            _enum_value(
                _attr(
                    clause,
                    "source",
                )
            )
        ),

        "model_name": _attr(
            clause,
            "model_name",
        ),

        "model_version": _attr(
            clause,
            "model_version",
        ),

        "metadata": _metadata(
            clause
        ),

        "created_at": _attr(
            clause,
            "created_at",
        ),

        "updated_at": _attr(
            clause,
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
    )

    limit = int(
        _attr(
            page,
            "limit",
            len(items),
        )
    )

    offset = int(
        _attr(
            page,
            "offset",
            0,
        )
    )

    return {
        "items": [
            _serialize_clause(item)
            for item in items
        ],

        "total": total,
        "limit": limit,
        "offset": offset,

        "has_more": bool(
            _attr(
                page,
                "has_more",
                offset + len(items) < total,
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
    name: str,
    **kwargs: Any,
) -> Any:

    method = getattr(
        service,
        name,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Clause operation "
                f"'{name}' is not configured."
            ),
        )

    return await method(**kwargs)


# ============================================================
# LIST CLAUSES
# ============================================================

@router.get(
    "",
    response_model=ClausePageResponse,
    summary="List legal clauses",
)
async def list_clauses(
    document_id: Optional[UUID] = Query(
        default=None
    ),

    contract_id: Optional[UUID] = Query(
        default=None
    ),

    document_version_id: Optional[UUID] = Query(
        default=None
    ),

    parent_clause_id: Optional[UUID] = Query(
        default=None
    ),

    clause_type: Optional[str] = Query(
        default=None
    ),

    risk_level: Optional[str] = Query(
        default=None
    ),

    page_number: Optional[int] = Query(
        default=None,
        ge=1,
    ),

    min_confidence: Optional[float] = Query(
        default=None,
        ge=0,
        le=1,
    ),

    min_risk_score: Optional[float] = Query(
        default=None,
        ge=0,
        le=1,
    ),

    max_risk_score: Optional[float] = Query(
        default=None,
        ge=0,
        le=1,
    ),

    source: Optional[str] = Query(
        default=None
    ),

    model_name: Optional[str] = Query(
        default=None
    ),

    search: Optional[str] = Query(
        default=None,
        max_length=1000,
    ),

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

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    filters = ClauseFilter(
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
        "document_id",
        str(document_id)
        if document_id
        else None,
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
        "document_version_id",
        str(document_version_id)
        if document_version_id
        else None,
    )

    _set_filter(
        filters,
        "parent_clause_id",
        str(parent_clause_id)
        if parent_clause_id
        else None,
    )

    if clause_type:
        if hasattr(
            filters,
            "clause_types",
        ):
            filters.clause_types = [
                clause_type
            ]
        else:
            _set_filter(
                filters,
                "clause_type",
                clause_type,
            )

    if risk_level:
        if hasattr(
            filters,
            "risk_levels",
        ):
            filters.risk_levels = [
                risk_level
            ]
        else:
            _set_filter(
                filters,
                "risk_level",
                risk_level,
            )

    _set_filter(
        filters,
        "page_number",
        page_number,
    )

    _set_filter(
        filters,
        "min_confidence",
        min_confidence,
    )

    _set_filter(
        filters,
        "min_risk_score",
        min_risk_score,
    )

    _set_filter(
        filters,
        "max_risk_score",
        max_risk_score,
    )

    _set_filter(
        filters,
        "source",
        source,
    )

    _set_filter(
        filters,
        "model_name",
        model_name,
    )

    _set_filter(
        filters,
        "search",
        search,
    )

    page = await repository.list(
        filters,
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# STATISTICS
# Keep before /{clause_id}
# ============================================================

@router.get(
    "/statistics",
    response_model=ClauseStatisticsResponse,
    summary="Get clause statistics",
)
async def clause_statistics(
    document_id: Optional[UUID] = None,
    contract_id: Optional[UUID] = None,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> Any:

    filters = ClauseFilter(
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
        "document_id",
        str(document_id)
        if document_id
        else None,
    )

    _set_filter(
        filters,
        "contract_id",
        str(contract_id)
        if contract_id
        else None,
    )

    result = await repository.statistics(
        filters
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    if hasattr(
        result,
        "to_dict",
    ):
        return result.to_dict()

    return {
        "total": _attr(
            result,
            "total",
            0,
        ),

        "by_type": _attr(
            result,
            "by_type",
            {},
        ),

        "by_risk": _attr(
            result,
            "by_risk",
            {},
        ),

        "average_confidence": (
            _attr(
                result,
                "average_confidence",
            )
            or _attr(
                result,
                "avg_confidence",
            )
        ),

        "average_risk_score": (
            _attr(
                result,
                "average_risk_score",
            )
            or _attr(
                result,
                "avg_risk_score",
            )
        ),

        "high_risk": _attr(
            result,
            "high_risk",
            0,
        ),

        "critical": _attr(
            result,
            "critical",
            0,
        ),
    }


# ============================================================
# HIGH-RISK CLAUSES
# ============================================================

@router.get(
    "/high-risk",
    response_model=ClausePageResponse,
    summary="List high-risk clauses",
)
async def list_high_risk_clauses(
    min_score: float = Query(
        default=0.70,
        ge=0,
        le=1,
    ),

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

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    try:
        page = await repository.list_high_risk(
            tenant_id=_tenant_id(
                current_user
            ),
            organization_id=_organization_id(
                current_user
            ),
            min_score=min_score,
            limit=limit,
            offset=offset,
        )

    except TypeError:
        filters = ClauseFilter(
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
            "min_risk_score",
            min_score,
        )

        if hasattr(
            filters,
            "risk_levels",
        ):
            filters.risk_levels = [
                "high",
                "critical",
            ]

        page = await repository.list(
            filters,
            limit=limit,
            offset=offset,
        )

    return _serialize_page(page)


# ============================================================
# DOCUMENT CLAUSES
# ============================================================

@router.get(
    "/document/{document_id}",
    response_model=ClausePageResponse,
    summary="List clauses for document",
)
async def document_clauses(
    document_id: UUID,

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

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_for_document(
        tenant_id=_tenant_id(
            current_user
        ),
        document_id=str(
            document_id
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# CONTRACT CLAUSES
# ============================================================

@router.get(
    "/contract/{contract_id}",
    response_model=ClausePageResponse,
    summary="List clauses for contract",
)
async def contract_clauses(
    contract_id: UUID,

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

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    page = await repository.list_for_contract(
        tenant_id=_tenant_id(
            current_user
        ),
        contract_id=str(
            contract_id
        ),
        limit=limit,
        offset=offset,
    )

    return _serialize_page(page)


# ============================================================
# CREATE CLAUSE
# ============================================================

@router.post(
    "",
    response_model=ClauseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create clause",
)
async def create_clause(
    payload: ClauseCreateRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    if (
        payload.page_start is not None
        and payload.page_end is not None
        and payload.page_start
        > payload.page_end
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "page_start cannot be greater "
                "than page_end."
            ),
        )

    if (
        payload.start_offset is not None
        and payload.end_offset is not None
        and payload.start_offset
        > payload.end_offset
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_offset cannot be greater "
                "than end_offset."
            ),
        )

    create_data = ClauseCreate(
        tenant_id=_tenant_id(
            current_user
        ),

        organization_id=_organization_id(
            current_user
        ),

        document_id=str(
            payload.document_id
        ),

        contract_id=(
            str(payload.contract_id)
            if payload.contract_id
            else None
        ),

        document_version_id=(
            str(
                payload.document_version_id
            )
            if payload.document_version_id
            else None
        ),

        parent_clause_id=(
            str(payload.parent_clause_id)
            if payload.parent_clause_id
            else None
        ),

        clause_type=(
            payload.clause_type
        ),

        clause_number=(
            payload.clause_number
        ),

        title=payload.title,

        text=payload.text,

        page_number=(
            payload.page_number
        ),

        page_start=(
            payload.page_start
        ),

        page_end=(
            payload.page_end
        ),

        start_offset=(
            payload.start_offset
        ),

        end_offset=(
            payload.end_offset
        ),

        confidence=(
            payload.confidence
        ),

        risk_level=(
            payload.risk_level
        ),

        risk_score=(
            payload.risk_score
        ),

        source=payload.source,

        model_name=(
            payload.model_name
        ),

        model_version=(
            payload.model_version
        ),

        metadata=payload.metadata,
    )

    clause = await repository.create(
        create_data
    )

    return _serialize_clause(
        clause
    )


# ============================================================
# GET CLAUSE
# ============================================================

@router.get(
    "/{clause_id}",
    response_model=ClauseResponse,
    summary="Get clause",
)
async def get_clause(
    clause_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    clause = await repository.get_by_id(
        clause_id,
        tenant_id=_tenant_id(
            current_user
        ),
    )

    if clause is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clause was not found.",
        )

    return _serialize_clause(
        clause
    )


# ============================================================
# UPDATE CLAUSE
# ============================================================

@router.patch(
    "/{clause_id}",
    response_model=ClauseResponse,
    summary="Update clause",
)
async def update_clause(
    clause_id: UUID,
    payload: ClauseUpdateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    values = payload.model_dump(
        exclude_unset=True
    )

    if (
        "parent_clause_id"
        in values
        and values[
            "parent_clause_id"
        ]
        is not None
    ):
        values[
            "parent_clause_id"
        ] = str(
            values[
                "parent_clause_id"
            ]
        )

    try:
        updated = await repository.update(
            clause_id=clause_id,

            tenant_id=_tenant_id(
                current_user
            ),

            data=ClauseUpdate(
                **values
            ),
        )

    except ClauseNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clause was not found.",
        ) from exc

    return _serialize_clause(
        updated
    )


# ============================================================
# UPDATE RISK
# ============================================================

@router.patch(
    "/{clause_id}/risk",
    response_model=ClauseResponse,
    summary="Update clause risk",
)
async def update_clause_risk(
    clause_id: UUID,
    payload: ClauseRiskUpdateRequest,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    try:
        clause = await repository.update_risk(
            clause_id=clause_id,

            tenant_id=_tenant_id(
                current_user
            ),

            risk_level=(
                payload.risk_level
            ),

            risk_score=(
                payload.risk_score
            ),
        )

    except ClauseNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clause was not found.",
        ) from exc

    return _serialize_clause(
        clause
    )


# ============================================================
# CHILD CLAUSES
# ============================================================

@router.get(
    "/{clause_id}/children",
    response_model=ClausePageResponse,
    summary="Get child clauses",
)
async def clause_children(
    clause_id: UUID,

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

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, Any]:

    page = await repository.children(
        tenant_id=_tenant_id(
            current_user
        ),

        parent_clause_id=str(
            clause_id
        ),

        limit=limit,
        offset=offset,
    )

    return _serialize_page(
        page
    )


# ============================================================
# AI CLAUSE ANALYSIS
# ============================================================

@router.post(
    "/{clause_id}/analyze",
    response_model=ClauseAnalysisResponse,
    summary="Analyze clause with AI",
)
async def analyze_clause(
    clause_id: UUID,
    payload: ClauseAnalyzeRequest,
    request: Request,

    current_user: Any = Depends(
        get_current_user
    ),

    service: Any = Depends(
        get_clause_service
    ),
) -> Any:

    result = await _service_call(
        service,
        "analyze",

        clause_id=str(
            clause_id
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

        run_classification=(
            payload.run_classification
        ),

        run_risk_analysis=(
            payload.run_risk_analysis
        ),

        model=payload.model,

        force=payload.force,

        request_id=_request_id(
            request
        ),
    )

    if isinstance(
        result,
        dict,
    ):
        return result

    return {
        "clause_id": str(
            clause_id
        ),

        "clause_type": _optional_str(
            _enum_value(
                _attr(
                    result,
                    "clause_type",
                )
            )
        ),

        "classification_confidence":
            _attr(
                result,
                "classification_confidence",
            ),

        "risk_level": _optional_str(
            _enum_value(
                _attr(
                    result,
                    "risk_level",
                )
            )
        ),

        "risk_score": _attr(
            result,
            "risk_score",
        ),

        "risk_reasons": list(
            _attr(
                result,
                "risk_reasons",
                [],
            )
            or []
        ),

        "model_name": _attr(
            result,
            "model_name",
        ),

        "model_version": _attr(
            result,
            "model_version",
        ),

        "analyzed_at": (
            _attr(
                result,
                "analyzed_at",
            )
            or datetime.utcnow()
        ),
    }


# ============================================================
# DELETE CLAUSE
# ============================================================

@router.delete(
    "/{clause_id}",
    response_model=MessageResponse,
    summary="Delete clause",
)
async def delete_clause(
    clause_id: UUID,

    current_user: Any = Depends(
        get_current_user
    ),

    repository: ClauseRepository = Depends(
        get_clause_repository
    ),
) -> dict[str, str]:

    deleted = await repository.delete(
        clause_id=clause_id,

        tenant_id=_tenant_id(
            current_user
        ),
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Clause was not found.",
        )

    return {
        "message":
            "Clause deleted successfully."
    }