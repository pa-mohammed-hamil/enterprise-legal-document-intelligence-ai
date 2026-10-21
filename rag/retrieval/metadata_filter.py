# backend/app/rag/retrieval/metadata_filter.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================


class MetadataFilterError(Exception):
    """Base metadata filtering error."""


class MetadataFilterValidationError(MetadataFilterError):
    """Raised when a metadata filter is invalid."""


class MetadataFilterSecurityError(MetadataFilterError):
    """Raised when tenant isolation rules are violated."""


# ============================================================
# Operators
# ============================================================


class FilterOperator(str, Enum):
    EQ = "eq"
    NE = "ne"

    IN = "in"
    NOT_IN = "not_in"

    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"

    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"

    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"

    EXISTS = "exists"
    NOT_EXISTS = "not_exists"

    BETWEEN = "between"

    ANY = "any"
    ALL = "all"


class FilterLogic(str, Enum):
    AND = "and"
    OR = "or"


# ============================================================
# Filter condition
# ============================================================


@dataclass(slots=True)
class MetadataFilterCondition:
    field: str
    operator: FilterOperator | str = FilterOperator.EQ
    value: Any = None

    def __post_init__(self) -> None:
        self.operator = FilterOperator(self.operator)

        if not self.field or not self.field.strip():
            raise MetadataFilterValidationError(
                "Filter field cannot be empty"
            )

        self.field = self.field.strip()


# ============================================================
# Filter group
# ============================================================


@dataclass(slots=True)
class MetadataFilterGroup:
    conditions: list[MetadataFilterCondition] = field(
        default_factory=list
    )

    logic: FilterLogic | str = FilterLogic.AND

    def __post_init__(self) -> None:
        self.logic = FilterLogic(self.logic)


# ============================================================
# Standard retrieval filter
# ============================================================


@dataclass(slots=True)
class RetrievalMetadataFilter:
    """
    Standard legal-document retrieval filters.

    organization_id is deliberately separated because it is
    a security boundary rather than a normal user-controlled
    metadata filter.
    """

    organization_id: str | UUID | None = None

    document_ids: Sequence[str | UUID] | None = None
    contract_ids: Sequence[str | UUID] | None = None
    clause_ids: Sequence[str | UUID] | None = None
    parent_ids: Sequence[str | UUID] | None = None

    document_types: Sequence[str] | None = None
    contract_types: Sequence[str] | None = None
    clause_types: Sequence[str] | None = None

    statuses: Sequence[str] | None = None
    risk_levels: Sequence[str] | None = None
    languages: Sequence[str] | None = None

    tags: Sequence[str] | None = None

    page_numbers: Sequence[int] | None = None
    page_from: int | None = None
    page_to: int | None = None

    created_from: date | datetime | None = None
    created_to: date | datetime | None = None

    effective_from: date | datetime | None = None
    effective_to: date | datetime | None = None

    expiry_from: date | datetime | None = None
    expiry_to: date | datetime | None = None

    minimum_risk_score: float | None = None
    maximum_risk_score: float | None = None

    minimum_confidence: float | None = None

    custom: Mapping[str, Any] | None = None


# ============================================================
# Metadata Filter
# ============================================================


class MetadataFilter:
    """
    Tenant-safe metadata filtering for Legal RAG.

    Supports:

    - organization / tenant isolation
    - document filtering
    - contract filtering
    - clause filtering
    - parent-child chunk filtering
    - document / contract / clause types
    - status
    - risk level and risk score
    - page filtering
    - date ranges
    - language
    - tags
    - extraction confidence
    - arbitrary metadata
    - AND / OR conditions
    - Qdrant filter generation

    Expected pipeline:

        query
          ↓
        query_router
          ↓
        MetadataFilter
          ↓
        BM25 + Vector Search
          ↓
        Hybrid Search
          ↓
        Permission Aware Search
          ↓
        Multi Query
          ↓
        Reranker
    """

    # Fields clients must not use to escape tenant isolation.
    PROTECTED_FIELDS = {
        "organization_id",
        "tenant_id",
        "workspace_id",
    }

    FIELD_ALIASES = {
        "org_id": "organization_id",
        "tenant": "organization_id",
        "doc_id": "document_id",
        "source_document_id": "document_id",
        "source_id": "document_id",
        "contract": "contract_id",
        "clause": "clause_id",
        "parent_chunk_id": "parent_id",
        "page": "page_number",
        "page_no": "page_number",
        "type": "document_type",
        "risk": "risk_level",
        "confidence": "extraction_confidence",
    }

    def __init__(
        self,
        *,
        strict_tenant_isolation: bool = True,
        case_sensitive: bool = False,
    ) -> None:
        self.strict_tenant_isolation = (
            strict_tenant_isolation
        )

        self.case_sensitive = case_sensitive

    # ========================================================
    # Public filtering
    # ========================================================

    def filter(
        self,
        items: Iterable[Any],
        *,
        organization_id: str | UUID | None = None,
        filters: (
            RetrievalMetadataFilter
            | MetadataFilterGroup
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> list[Any]:
        """
        Filter retrieval results/chunks in memory.

        organization_id should come from trusted auth/tenant
        context, never directly from an untrusted request body.
        """

        tenant = self._string_or_none(
            organization_id
        )

        if (
            self.strict_tenant_isolation
            and tenant is None
        ):
            raise MetadataFilterSecurityError(
                "organization_id is required when "
                "strict tenant isolation is enabled"
            )

        normalized = self.normalize_filter(
            filters,
            organization_id=tenant,
        )

        return [
            item
            for item in items
            if self.matches(
                item,
                organization_id=tenant,
                filters=normalized,
            )
        ]

    # ========================================================
    # Single item
    # ========================================================

    def matches(
        self,
        item: Any,
        *,
        organization_id: str | UUID | None = None,
        filters: (
            RetrievalMetadataFilter
            | MetadataFilterGroup
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> bool:
        tenant = self._string_or_none(
            organization_id
        )

        data = self._extract_data(item)

        if self.strict_tenant_isolation:
            if tenant is None:
                raise MetadataFilterSecurityError(
                    "organization_id is required"
                )

            item_tenant = self._string_or_none(
                self._lookup(
                    data,
                    "organization_id",
                )
            )

            if item_tenant != tenant:
                return False

        normalized = self.normalize_filter(
            filters,
            organization_id=tenant,
        )

        if normalized is None:
            return True

        if isinstance(
            normalized,
            RetrievalMetadataFilter,
        ):
            return self._matches_retrieval_filter(
                data,
                normalized,
            )

        if isinstance(
            normalized,
            MetadataFilterGroup,
        ):
            return self._matches_group(
                data,
                normalized,
            )

        return True

    # ========================================================
    # Normalize
    # ========================================================

    def normalize_filter(
        self,
        filters: (
            RetrievalMetadataFilter
            | MetadataFilterGroup
            | Mapping[str, Any]
            | None
        ),
        *,
        organization_id: str | None = None,
    ) -> (
        RetrievalMetadataFilter
        | MetadataFilterGroup
        | None
    ):
        if filters is None:
            if organization_id is None:
                return None

            return RetrievalMetadataFilter(
                organization_id=organization_id
            )

        if isinstance(
            filters,
            RetrievalMetadataFilter,
        ):
            supplied_tenant = self._string_or_none(
                filters.organization_id
            )

            self._validate_tenant(
                trusted=organization_id,
                supplied=supplied_tenant,
            )

            filters.organization_id = (
                organization_id
                or supplied_tenant
            )

            return filters

        if isinstance(
            filters,
            MetadataFilterGroup,
        ):
            self._validate_group_security(
                filters,
                organization_id,
            )

            return filters

        if not isinstance(filters, Mapping):
            raise MetadataFilterValidationError(
                "filters must be a mapping, "
                "RetrievalMetadataFilter or "
                "MetadataFilterGroup"
            )

        return self._from_mapping(
            filters,
            organization_id=organization_id,
        )

    # ========================================================
    # Mapping conversion
    # ========================================================

    def _from_mapping(
        self,
        filters: Mapping[str, Any],
        *,
        organization_id: str | None,
    ) -> RetrievalMetadataFilter:
        payload = dict(filters)

        supplied_tenant = (
            payload.get("organization_id")
            or payload.get("tenant_id")
        )

        supplied_tenant = self._string_or_none(
            supplied_tenant
        )

        self._validate_tenant(
            trusted=organization_id,
            supplied=supplied_tenant,
        )

        known_fields = {
            "document_ids",
            "contract_ids",
            "clause_ids",
            "parent_ids",
            "document_types",
            "contract_types",
            "clause_types",
            "statuses",
            "risk_levels",
            "languages",
            "tags",
            "page_numbers",
            "page_from",
            "page_to",
            "created_from",
            "created_to",
            "effective_from",
            "effective_to",
            "expiry_from",
            "expiry_to",
            "minimum_risk_score",
            "maximum_risk_score",
            "minimum_confidence",
        }

        custom = dict(
            payload.get("custom")
            or {}
        )

        for key, value in payload.items():
            if key in {
                "organization_id",
                "tenant_id",
                "custom",
            }:
                continue

            if key not in known_fields:
                custom[key] = value

        return RetrievalMetadataFilter(
            organization_id=(
                organization_id
                or supplied_tenant
            ),
            document_ids=self._sequence(
                payload.get("document_ids")
            ),
            contract_ids=self._sequence(
                payload.get("contract_ids")
            ),
            clause_ids=self._sequence(
                payload.get("clause_ids")
            ),
            parent_ids=self._sequence(
                payload.get("parent_ids")
            ),
            document_types=self._sequence(
                payload.get("document_types")
            ),
            contract_types=self._sequence(
                payload.get("contract_types")
            ),
            clause_types=self._sequence(
                payload.get("clause_types")
            ),
            statuses=self._sequence(
                payload.get("statuses")
            ),
            risk_levels=self._sequence(
                payload.get("risk_levels")
            ),
            languages=self._sequence(
                payload.get("languages")
            ),
            tags=self._sequence(
                payload.get("tags")
            ),
            page_numbers=self._sequence(
                payload.get("page_numbers")
            ),
            page_from=payload.get("page_from"),
            page_to=payload.get("page_to"),
            created_from=payload.get(
                "created_from"
            ),
            created_to=payload.get(
                "created_to"
            ),
            effective_from=payload.get(
                "effective_from"
            ),
            effective_to=payload.get(
                "effective_to"
            ),
            expiry_from=payload.get(
                "expiry_from"
            ),
            expiry_to=payload.get(
                "expiry_to"
            ),
            minimum_risk_score=payload.get(
                "minimum_risk_score"
            ),
            maximum_risk_score=payload.get(
                "maximum_risk_score"
            ),
            minimum_confidence=payload.get(
                "minimum_confidence"
            ),
            custom=custom or None,
        )

    # ========================================================
    # Standard retrieval filter
    # ========================================================

    def _matches_retrieval_filter(
        self,
        data: Mapping[str, Any],
        filters: RetrievalMetadataFilter,
    ) -> bool:
        checks = (
            (
                "organization_id",
                filters.organization_id,
            ),
            (
                "document_id",
                filters.document_ids,
            ),
            (
                "contract_id",
                filters.contract_ids,
            ),
            (
                "clause_id",
                filters.clause_ids,
            ),
            (
                "parent_id",
                filters.parent_ids,
            ),
            (
                "document_type",
                filters.document_types,
            ),
            (
                "contract_type",
                filters.contract_types,
            ),
            (
                "clause_type",
                filters.clause_types,
            ),
            (
                "status",
                filters.statuses,
            ),
            (
                "risk_level",
                filters.risk_levels,
            ),
            (
                "language",
                filters.languages,
            ),
        )

        for field_name, expected in checks:
            if expected is None:
                continue

            actual = self._lookup(
                data,
                field_name,
            )

            if isinstance(
                expected,
                (list, tuple, set, frozenset),
            ):
                if not self._in(
                    actual,
                    expected,
                ):
                    return False

            elif not self._equals(
                actual,
                expected,
            ):
                return False

        # ----------------------------------------------------
        # Tags
        # ----------------------------------------------------

        if filters.tags:
            actual_tags = self._lookup(
                data,
                "tags",
            )

            if not self._contains_all(
                actual_tags,
                filters.tags,
            ):
                return False

        # ----------------------------------------------------
        # Page
        # ----------------------------------------------------

        page_number = self._lookup(
            data,
            "page_number",
        )

        if filters.page_numbers:
            if not self._in(
                page_number,
                filters.page_numbers,
            ):
                return False

        if (
            filters.page_from is not None
            and not self._compare_number(
                page_number,
                filters.page_from,
                FilterOperator.GTE,
            )
        ):
            return False

        if (
            filters.page_to is not None
            and not self._compare_number(
                page_number,
                filters.page_to,
                FilterOperator.LTE,
            )
        ):
            return False

        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        risk_score = self._lookup(
            data,
            "risk_score",
        )

        if (
            filters.minimum_risk_score
            is not None
            and not self._compare_number(
                risk_score,
                filters.minimum_risk_score,
                FilterOperator.GTE,
            )
        ):
            return False

        if (
            filters.maximum_risk_score
            is not None
            and not self._compare_number(
                risk_score,
                filters.maximum_risk_score,
                FilterOperator.LTE,
            )
        ):
            return False

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        if filters.minimum_confidence is not None:
            confidence = (
                self._lookup(
                    data,
                    "extraction_confidence",
                )
            )

            if confidence is None:
                confidence = self._lookup(
                    data,
                    "confidence",
                )

            if not self._compare_number(
                confidence,
                filters.minimum_confidence,
                FilterOperator.GTE,
            ):
                return False

        # ----------------------------------------------------
        # Dates
        # ----------------------------------------------------

        date_ranges = (
            (
                "created_at",
                filters.created_from,
                filters.created_to,
            ),
            (
                "effective_date",
                filters.effective_from,
                filters.effective_to,
            ),
            (
                "expiry_date",
                filters.expiry_from,
                filters.expiry_to,
            ),
        )

        for (
            field_name,
            start,
            end,
        ) in date_ranges:
            if start is None and end is None:
                continue

            actual = self._lookup(
                data,
                field_name,
            )

            if not self._date_in_range(
                actual,
                start=start,
                end=end,
            ):
                return False

        # ----------------------------------------------------
        # Custom metadata
        # ----------------------------------------------------

        if filters.custom:
            for key, expected in (
                filters.custom.items()
            ):
                if key in self.PROTECTED_FIELDS:
                    continue

                actual = self._lookup(
                    data,
                    key,
                )

                if isinstance(expected, Mapping):
                    if not self._matches_operator_mapping(
                        actual,
                        expected,
                    ):
                        return False

                elif isinstance(
                    expected,
                    (list, tuple, set),
                ):
                    if not self._in(
                        actual,
                        expected,
                    ):
                        return False

                elif not self._equals(
                    actual,
                    expected,
                ):
                    return False

        return True

    # ========================================================
    # Advanced groups
    # ========================================================

    def _matches_group(
        self,
        data: Mapping[str, Any],
        group: MetadataFilterGroup,
    ) -> bool:
        if not group.conditions:
            return True

        results = [
            self._matches_condition(
                data,
                condition,
            )
            for condition in group.conditions
        ]

        if group.logic == FilterLogic.OR:
            return any(results)

        return all(results)

    def _matches_condition(
        self,
        data: Mapping[str, Any],
        condition: MetadataFilterCondition,
    ) -> bool:
        field_name = self.FIELD_ALIASES.get(
            condition.field,
            condition.field,
        )

        actual = self._lookup(
            data,
            field_name,
        )

        operator = condition.operator
        expected = condition.value

        if operator == FilterOperator.EQ:
            return self._equals(
                actual,
                expected,
            )

        if operator == FilterOperator.NE:
            return not self._equals(
                actual,
                expected,
            )

        if operator == FilterOperator.IN:
            return self._in(
                actual,
                self._sequence(expected),
            )

        if operator == FilterOperator.NOT_IN:
            return not self._in(
                actual,
                self._sequence(expected),
            )

        if operator in {
            FilterOperator.GT,
            FilterOperator.GTE,
            FilterOperator.LT,
            FilterOperator.LTE,
        }:
            return self._compare(
                actual,
                expected,
                operator,
            )

        if operator == FilterOperator.CONTAINS:
            return self._contains(
                actual,
                expected,
            )

        if operator == FilterOperator.NOT_CONTAINS:
            return not self._contains(
                actual,
                expected,
            )

        if operator == FilterOperator.STARTS_WITH:
            return self._string_compare(
                actual,
                expected,
                mode="starts_with",
            )

        if operator == FilterOperator.ENDS_WITH:
            return self._string_compare(
                actual,
                expected,
                mode="ends_with",
            )

        if operator == FilterOperator.EXISTS:
            return actual is not None

        if operator == FilterOperator.NOT_EXISTS:
            return actual is None

        if operator == FilterOperator.BETWEEN:
            values = self._sequence(expected)

            if len(values) != 2:
                raise MetadataFilterValidationError(
                    "between requires exactly "
                    "two values"
                )

            return (
                self._compare(
                    actual,
                    values[0],
                    FilterOperator.GTE,
                )
                and self._compare(
                    actual,
                    values[1],
                    FilterOperator.LTE,
                )
            )

        if operator == FilterOperator.ANY:
            return self._contains_any(
                actual,
                self._sequence(expected),
            )

        if operator == FilterOperator.ALL:
            return self._contains_all(
                actual,
                self._sequence(expected),
            )

        raise MetadataFilterValidationError(
            f"Unsupported operator: {operator}"
        )

    # ========================================================
    # Operator mapping
    # ========================================================

    def _matches_operator_mapping(
        self,
        actual: Any,
        config: Mapping[str, Any],
    ) -> bool:
        for operator_name, expected in config.items():
            try:
                operator = FilterOperator(
                    operator_name
                )
            except ValueError as exc:
                raise MetadataFilterValidationError(
                    f"Unknown filter operator: "
                    f"{operator_name}"
                ) from exc

            condition = MetadataFilterCondition(
                field="_",
                operator=operator,
                value=expected,
            )

            fake_data = {"_": actual}

            if not self._matches_condition(
                fake_data,
                condition,
            ):
                return False

        return True

    # ========================================================
    # Data extraction
    # ========================================================

    def _extract_data(
        self,
        item: Any,
    ) -> dict[str, Any]:
        if item is None:
            return {}

        if isinstance(item, Mapping):
            data = dict(item)

        elif hasattr(item, "to_dict"):
            value = item.to_dict()
            data = (
                dict(value)
                if isinstance(value, Mapping)
                else {}
            )

        elif hasattr(item, "model_dump"):
            value = item.model_dump()
            data = (
                dict(value)
                if isinstance(value, Mapping)
                else {}
            )

        else:
            data = {}

            for name in (
                "id",
                "chunk_id",
                "organization_id",
                "tenant_id",
                "document_id",
                "contract_id",
                "clause_id",
                "parent_id",
                "parent_chunk_id",
                "document_type",
                "contract_type",
                "clause_type",
                "status",
                "risk_level",
                "risk_score",
                "language",
                "tags",
                "page_number",
                "chunk_index",
                "created_at",
                "effective_date",
                "expiry_date",
                "confidence",
                "extraction_confidence",
                "metadata",
                "metadata_json",
                "payload",
            ):
                if hasattr(item, name):
                    data[name] = getattr(
                        item,
                        name,
                    )

        # Flatten retrieval metadata into the top-level lookup
        # namespace without overriding explicit top-level data.
        for metadata_key in (
            "metadata",
            "metadata_json",
            "payload",
        ):
            metadata = data.get(
                metadata_key
            )

            if isinstance(metadata, Mapping):
                for key, value in (
                    metadata.items()
                ):
                    data.setdefault(
                        key,
                        value,
                    )

        return data

    # ========================================================
    # Lookup
    # ========================================================

    def _lookup(
        self,
        data: Mapping[str, Any],
        field_name: str,
    ) -> Any:
        field_name = self.FIELD_ALIASES.get(
            field_name,
            field_name,
        )

        # Supports nested fields:
        #   source.jurisdiction
        #   access.department
        #   metadata.region

        if "." in field_name:
            current: Any = data

            for part in field_name.split("."):
                if isinstance(
                    current,
                    Mapping,
                ):
                    current = current.get(
                        part
                    )
                else:
                    current = getattr(
                        current,
                        part,
                        None,
                    )

                if current is None:
                    return None

            return current

        if field_name in data:
            return data[field_name]

        for metadata_name in (
            "metadata",
            "metadata_json",
            "payload",
        ):
            metadata = data.get(
                metadata_name
            )

            if (
                isinstance(metadata, Mapping)
                and field_name in metadata
            ):
                return metadata[field_name]

        return None

    # ========================================================
    # Comparison
    # ========================================================

    def _equals(
        self,
        actual: Any,
        expected: Any,
    ) -> bool:
        actual = self._enum_value(actual)
        expected = self._enum_value(expected)

        if isinstance(actual, UUID):
            actual = str(actual)

        if isinstance(expected, UUID):
            expected = str(expected)

        if (
            isinstance(actual, str)
            and isinstance(expected, str)
        ):
            if not self.case_sensitive:
                return (
                    actual.casefold()
                    == expected.casefold()
                )

        return actual == expected

    def _in(
        self,
        actual: Any,
        expected: Sequence[Any],
    ) -> bool:
        return any(
            self._equals(
                actual,
                item,
            )
            for item in expected
        )

    def _compare(
        self,
        actual: Any,
        expected: Any,
        operator: FilterOperator,
    ) -> bool:
        if actual is None or expected is None:
            return False

        # Date/time comparison
        actual_date = self._to_datetime(
            actual
        )

        expected_date = self._to_datetime(
            expected
        )

        if (
            actual_date is not None
            and expected_date is not None
        ):
            left = actual_date
            right = expected_date

        else:
            # Numeric comparison
            try:
                left = float(actual)
                right = float(expected)

            except (
                TypeError,
                ValueError,
            ):
                left = str(actual)
                right = str(expected)

                if not self.case_sensitive:
                    left = left.casefold()
                    right = right.casefold()

        if operator == FilterOperator.GT:
            return left > right

        if operator == FilterOperator.GTE:
            return left >= right

        if operator == FilterOperator.LT:
            return left < right

        if operator == FilterOperator.LTE:
            return left <= right

        return False

    def _compare_number(
        self,
        actual: Any,
        expected: Any,
        operator: FilterOperator,
    ) -> bool:
        if actual is None:
            return False

        try:
            left = float(actual)
            right = float(expected)

        except (
            TypeError,
            ValueError,
        ):
            return False

        if operator == FilterOperator.GT:
            return left > right

        if operator == FilterOperator.GTE:
            return left >= right

        if operator == FilterOperator.LT:
            return left < right

        if operator == FilterOperator.LTE:
            return left <= right

        return False

    # ========================================================
    # Contains
    # ========================================================

    def _contains(
        self,
        actual: Any,
        expected: Any,
    ) -> bool:
        if actual is None:
            return False

        if isinstance(
            actual,
            (list, tuple, set, frozenset),
        ):
            return any(
                self._equals(
                    item,
                    expected,
                )
                for item in actual
            )

        if isinstance(actual, Mapping):
            return (
                expected in actual
                or str(expected) in actual
            )

        actual_text = str(actual)
        expected_text = str(expected)

        if not self.case_sensitive:
            actual_text = (
                actual_text.casefold()
            )
            expected_text = (
                expected_text.casefold()
            )

        return expected_text in actual_text

    def _contains_any(
        self,
        actual: Any,
        expected: Sequence[Any],
    ) -> bool:
        return any(
            self._contains(
                actual,
                item,
            )
            for item in expected
        )

    def _contains_all(
        self,
        actual: Any,
        expected: Sequence[Any],
    ) -> bool:
        return all(
            self._contains(
                actual,
                item,
            )
            for item in expected
        )

    def _string_compare(
        self,
        actual: Any,
        expected: Any,
        *,
        mode: str,
    ) -> bool:
        if actual is None:
            return False

        actual_text = str(actual)
        expected_text = str(expected)

        if not self.case_sensitive:
            actual_text = actual_text.casefold()
            expected_text = expected_text.casefold()

        if mode == "starts_with":
            return actual_text.startswith(
                expected_text
            )

        if mode == "ends_with":
            return actual_text.endswith(
                expected_text
            )

        return False

    # ========================================================
    # Date helpers
    # ========================================================

    def _date_in_range(
        self,
        value: Any,
        *,
        start: date | datetime | None,
        end: date | datetime | None,
    ) -> bool:
        actual = self._to_datetime(value)

        if actual is None:
            return False

        if start is not None:
            start_dt = self._to_datetime(
                start
            )

            if (
                start_dt is not None
                and actual < start_dt
            ):
                return False

        if end is not None:
            end_dt = self._to_datetime(
                end,
                end_of_day=True,
            )

            if (
                end_dt is not None
                and actual > end_dt
            ):
                return False

        return True

    @staticmethod
    def _to_datetime(
        value: Any,
        *,
        end_of_day: bool = False,
    ) -> datetime | None:
        if value is None:
            return None

        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(
                    tzinfo=timezone.utc
                )

            return value.astimezone(
                timezone.utc
            )

        if isinstance(value, date):
            return datetime(
                value.year,
                value.month,
                value.day,
                23 if end_of_day else 0,
                59 if end_of_day else 0,
                59 if end_of_day else 0,
                999999 if end_of_day else 0,
                tzinfo=timezone.utc,
            )

        if isinstance(value, str):
            text = value.strip()

            if not text:
                return None

            if text.endswith("Z"):
                text = (
                    text[:-1]
                    + "+00:00"
                )

            try:
                parsed = datetime.fromisoformat(
                    text
                )

                if parsed.tzinfo is None:
                    parsed = parsed.replace(
                        tzinfo=timezone.utc
                    )

                return parsed.astimezone(
                    timezone.utc
                )

            except ValueError:
                try:
                    parsed_date = (
                        date.fromisoformat(
                            text
                        )
                    )

                    return MetadataFilter._to_datetime(
                        parsed_date,
                        end_of_day=end_of_day,
                    )

                except ValueError:
                    return None

        return None

    # ========================================================
    # Security
    # ========================================================

    def _validate_tenant(
        self,
        *,
        trusted: str | None,
        supplied: str | None,
    ) -> None:
        if (
            trusted is not None
            and supplied is not None
            and trusted != supplied
        ):
            raise MetadataFilterSecurityError(
                "Filter organization_id does not "
                "match authenticated organization"
            )

        if (
            self.strict_tenant_isolation
            and trusted is None
            and supplied is None
        ):
            raise MetadataFilterSecurityError(
                "organization_id is required"
            )

    def _validate_group_security(
        self,
        group: MetadataFilterGroup,
        trusted_organization_id: str | None,
    ) -> None:
        for condition in group.conditions:
            field_name = self.FIELD_ALIASES.get(
                condition.field,
                condition.field,
            )

            if field_name not in self.PROTECTED_FIELDS:
                continue

            supplied = self._string_or_none(
                condition.value
            )

            self._validate_tenant(
                trusted=trusted_organization_id,
                supplied=supplied,
            )

    # ========================================================
    # Qdrant conversion
    # ========================================================

    def to_qdrant_filter(
        self,
        *,
        organization_id: str | UUID,
        filters: (
            RetrievalMetadataFilter
            | Mapping[str, Any]
            | None
        ) = None,
    ) -> dict[str, Any]:
        """
        Produce a Qdrant-compatible JSON filter.

        The returned dictionary can either be passed through
        your qdrant_client adapter or converted into Qdrant's
        Filter/FieldCondition models there.

        Example:

        {
            "must": [
                {
                    "key": "organization_id",
                    "match": {"value": "..."}
                },
                {
                    "key": "document_id",
                    "match": {"any": ["...", "..."]}
                }
            ]
        }
        """

        tenant = str(organization_id)

        normalized = self.normalize_filter(
            filters,
            organization_id=tenant,
        )

        if not isinstance(
            normalized,
            RetrievalMetadataFilter,
        ):
            raise MetadataFilterValidationError(
                "Qdrant conversion currently requires "
                "RetrievalMetadataFilter or mapping"
            )

        must: list[dict[str, Any]] = []

        must.append(
            {
                "key": "organization_id",
                "match": {
                    "value": tenant,
                },
            }
        )

        self._qdrant_match_any(
            must,
            "document_id",
            normalized.document_ids,
        )

        self._qdrant_match_any(
            must,
            "contract_id",
            normalized.contract_ids,
        )

        self._qdrant_match_any(
            must,
            "clause_id",
            normalized.clause_ids,
        )

        self._qdrant_match_any(
            must,
            "parent_id",
            normalized.parent_ids,
        )

        self._qdrant_match_any(
            must,
            "document_type",
            normalized.document_types,
        )

        self._qdrant_match_any(
            must,
            "contract_type",
            normalized.contract_types,
        )

        self._qdrant_match_any(
            must,
            "clause_type",
            normalized.clause_types,
        )

        self._qdrant_match_any(
            must,
            "status",
            normalized.statuses,
        )

        self._qdrant_match_any(
            must,
            "risk_level",
            normalized.risk_levels,
        )

        self._qdrant_match_any(
            must,
            "language",
            normalized.languages,
        )

        self._qdrant_match_any(
            must,
            "page_number",
            normalized.page_numbers,
        )

        if (
            normalized.page_from is not None
            or normalized.page_to is not None
        ):
            range_value: dict[str, Any] = {}

            if normalized.page_from is not None:
                range_value["gte"] = (
                    normalized.page_from
                )

            if normalized.page_to is not None:
                range_value["lte"] = (
                    normalized.page_to
                )

            must.append(
                {
                    "key": "page_number",
                    "range": range_value,
                }
            )

        if (
            normalized.minimum_risk_score
            is not None
            or normalized.maximum_risk_score
            is not None
        ):
            range_value = {}

            if (
                normalized.minimum_risk_score
                is not None
            ):
                range_value["gte"] = (
                    normalized.minimum_risk_score
                )

            if (
                normalized.maximum_risk_score
                is not None
            ):
                range_value["lte"] = (
                    normalized.maximum_risk_score
                )

            must.append(
                {
                    "key": "risk_score",
                    "range": range_value,
                }
            )

        if (
            normalized.minimum_confidence
            is not None
        ):
            must.append(
                {
                    "key":
                        "extraction_confidence",
                    "range": {
                        "gte":
                            normalized.minimum_confidence
                    },
                }
            )

        if normalized.tags:
            for tag in normalized.tags:
                must.append(
                    {
                        "key": "tags",
                        "match": {
                            "value": tag,
                        },
                    }
                )

        if normalized.custom:
            for key, value in (
                normalized.custom.items()
            ):
                if key in self.PROTECTED_FIELDS:
                    continue

                if isinstance(
                    value,
                    (list, tuple, set),
                ):
                    must.append(
                        {
                            "key": key,
                            "match": {
                                "any": [
                                    self._enum_value(
                                        item
                                    )
                                    for item in value
                                ]
                            },
                        }
                    )

                elif not isinstance(
                    value,
                    Mapping,
                ):
                    must.append(
                        {
                            "key": key,
                            "match": {
                                "value":
                                    self._enum_value(
                                        value
                                    )
                            },
                        }
                    )

        return {
            "must": must,
        }

    @classmethod
    def _qdrant_match_any(
        cls,
        output: list[dict[str, Any]],
        key: str,
        values: Sequence[Any] | None,
    ) -> None:
        if not values:
            return

        normalized = [
            cls._json_value(value)
            for value in values
        ]

        if len(normalized) == 1:
            output.append(
                {
                    "key": key,
                    "match": {
                        "value": normalized[0],
                    },
                }
            )

        else:
            output.append(
                {
                    "key": key,
                    "match": {
                        "any": normalized,
                    },
                }
            )

    # ========================================================
    # BM25 filter conversion
    # ========================================================

    def to_bm25_filters(
        self,
        filters: (
            RetrievalMetadataFilter
            | Mapping[str, Any]
            | None
        ),
        *,
        organization_id: str | UUID,
    ) -> dict[str, Any]:
        """
        Produce metadata filters suitable for bm25.py.

        document_ids and contract_ids should normally be passed
        through their dedicated BM25 search arguments. This
        method returns additional metadata restrictions.
        """

        normalized = self.normalize_filter(
            filters,
            organization_id=str(
                organization_id
            ),
        )

        if not isinstance(
            normalized,
            RetrievalMetadataFilter,
        ):
            return {}

        output: dict[str, Any] = {}

        mapping = {
            "document_type":
                normalized.document_types,
            "contract_type":
                normalized.contract_types,
            "clause_type":
                normalized.clause_types,
            "status":
                normalized.statuses,
            "risk_level":
                normalized.risk_levels,
            "language":
                normalized.languages,
            "page_number":
                normalized.page_numbers,
        }

        for key, value in mapping.items():
            if value:
                output[key] = list(value)

        if normalized.custom:
            output.update(
                normalized.custom
            )

        return output

    # ========================================================
    # Search argument conversion
    # ========================================================

    def retrieval_arguments(
        self,
        filters: (
            RetrievalMetadataFilter
            | Mapping[str, Any]
            | None
        ),
        *,
        organization_id: str | UUID,
    ) -> dict[str, Any]:
        """
        Convenience helper for hybrid_search.py.

        Example:

            args = metadata_filter.retrieval_arguments(
                request.filters,
                organization_id=current_user.organization_id,
            )

            results = await hybrid.search(
                query,
                **args,
            )
        """

        normalized = self.normalize_filter(
            filters,
            organization_id=str(
                organization_id
            ),
        )

        if not isinstance(
            normalized,
            RetrievalMetadataFilter,
        ):
            return {
                "organization_id":
                    organization_id,
            }

        return {
            "organization_id":
                organization_id,
            "document_ids":
                normalized.document_ids,
            "contract_ids":
                normalized.contract_ids,
            "filters":
                self.to_bm25_filters(
                    normalized,
                    organization_id=organization_id,
                ),
        }

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _sequence(
        value: Any,
    ) -> list[Any]:
        if value is None:
            return []

        if isinstance(
            value,
            (list, tuple, set, frozenset),
        ):
            return list(value)

        return [value]

    @staticmethod
    def _string_or_none(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        return str(
            MetadataFilter._enum_value(value)
        )

    @staticmethod
    def _enum_value(
        value: Any,
    ) -> Any:
        if isinstance(value, Enum):
            return value.value

        return value

    @staticmethod
    def _json_value(
        value: Any,
    ) -> Any:
        value = MetadataFilter._enum_value(
            value
        )

        if isinstance(value, UUID):
            return str(value)

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, date):
            return value.isoformat()

        return value

    # ========================================================
    # Health
    # ========================================================

    def health_check(
        self,
    ) -> dict[str, Any]:
        return {
            "service": "metadata_filter",
            "status": "healthy",
            "strict_tenant_isolation":
                self.strict_tenant_isolation,
            "case_sensitive":
                self.case_sensitive,
            "operators": [
                operator.value
                for operator in FilterOperator
            ],
        }


# ============================================================
# Compatibility aliases
# ============================================================

MetadataFilterEngine = MetadataFilter
RetrievalFilter = RetrievalMetadataFilter
FilterCondition = MetadataFilterCondition
FilterGroup = MetadataFilterGroup