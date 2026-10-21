# backend/app/rag/retrieval/permission_aware_search.py

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================
# Exceptions
# ============================================================


class PermissionSearchError(Exception):
    """Base permission-aware retrieval error."""


class PermissionSearchValidationError(PermissionSearchError):
    """Invalid permission-aware search request."""


class PermissionSearchAccessError(PermissionSearchError):
    """Search request violates an access boundary."""


class PermissionBackendError(PermissionSearchError):
    """Permission/access backend failed."""


# ============================================================
# Visibility
# ============================================================


class DocumentVisibility(str, Enum):
    PRIVATE = "private"
    RESTRICTED = "restricted"
    INTERNAL = "internal"
    ORGANIZATION = "organization"
    PUBLIC = "public"


# ============================================================
# Access context
# ============================================================


@dataclass(slots=True)
class SearchAccessContext:
    """
    Trusted authorization context.

    This should be built from authenticated server-side
    identity/RBAC data, not directly from the request body.
    """

    user_id: str | UUID
    organization_id: str | UUID

    roles: Sequence[str] = field(
        default_factory=tuple
    )

    permissions: Sequence[str] = field(
        default_factory=tuple
    )

    department_ids: Sequence[str | UUID] = field(
        default_factory=tuple
    )

    team_ids: Sequence[str | UUID] = field(
        default_factory=tuple
    )

    group_ids: Sequence[str | UUID] = field(
        default_factory=tuple
    )

    matter_ids: Sequence[str | UUID] = field(
        default_factory=tuple
    )

    is_admin: bool = False
    is_superuser: bool = False

    def normalized_user_id(self) -> str:
        return str(self.user_id)

    def normalized_organization_id(self) -> str:
        return str(self.organization_id)

    def normalized_roles(self) -> set[str]:
        return {
            str(value).casefold()
            for value in self.roles
        }

    def normalized_permissions(self) -> set[str]:
        return {
            str(value).casefold()
            for value in self.permissions
        }

    def normalized_departments(self) -> set[str]:
        return {
            str(value)
            for value in self.department_ids
        }

    def normalized_teams(self) -> set[str]:
        return {
            str(value)
            for value in self.team_ids
        }

    def normalized_groups(self) -> set[str]:
        return {
            str(value)
            for value in self.group_ids
        }

    def normalized_matters(self) -> set[str]:
        return {
            str(value)
            for value in self.matter_ids
        }


# ============================================================
# Permission decision
# ============================================================


@dataclass(slots=True)
class PermissionDecision:
    allowed: bool
    reason: str

    document_id: str | None = None
    resource_id: str | None = None

    rules_checked: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "document_id": self.document_id,
            "resource_id": self.resource_id,
            "rules_checked": self.rules_checked,
        }


# ============================================================
# Search result
# ============================================================


@dataclass(slots=True)
class PermissionAwareSearchResult:
    id: str
    text: str
    score: float
    rank: int

    organization_id: str | None = None
    document_id: str | None = None
    contract_id: str | None = None
    clause_id: str | None = None
    parent_id: str | None = None

    title: str | None = None
    page_number: int | None = None
    chunk_index: int | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    sources: list[str] = field(
        default_factory=list
    )

    matched_terms: list[str] = field(
        default_factory=list
    )

    permission_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "organization_id": self.organization_id,
            "document_id": self.document_id,
            "contract_id": self.contract_id,
            "clause_id": self.clause_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
            "sources": self.sources,
            "matched_terms": self.matched_terms,
            "permission_reason":
                self.permission_reason,
        }


# ============================================================
# Permission-aware retriever
# ============================================================


class PermissionAwareSearch:
    """
    Security layer for RAG retrieval.

    Recommended pipeline:

        authenticated user
              │
              ▼
        SearchAccessContext
              │
              ▼
        metadata_filter.py
              │
              ▼
        BM25 + vector_search.py
              │
              ▼
        hybrid_search.py
              │
              ▼
        multi_query.py
              │
              ▼
        permission_aware_search.py
              │
              ├── tenant validation
              ├── explicit deny
              ├── ownership
              ├── visibility
              ├── user ACL
              ├── role ACL
              ├── department ACL
              ├── team ACL
              ├── group ACL
              ├── matter ACL
              └── document_access service
              │
              ▼
        reranker
              │
              ▼
        Legal LLM
              │
              ▼
        citations

    Security principles:

    1. Tenant isolation is mandatory.
    2. Explicit deny overrides allow.
    3. Missing ACL metadata fails closed for restricted data.
    4. Permission context comes from trusted auth state.
    5. Retrieved unauthorized text never reaches the LLM.
    """

    READ_PERMISSIONS = {
        "document.read",
        "documents.read",
        "document:view",
        "documents:view",
        "legal_document.read",
        "rag.search",
        "search.documents",
    }

    ADMIN_ROLES = {
        "owner",
        "admin",
        "organization_admin",
        "legal_admin",
    }

    def __init__(
        self,
        *,
        retriever: Any,
        document_access: Any | None = None,
        audit_service: Any | None = None,
        fail_closed: bool = True,
        allow_public_documents: bool = False,
        admin_bypass_acl: bool = True,
    ) -> None:
        if retriever is None:
            raise PermissionSearchValidationError(
                "retriever is required"
            )

        self.retriever = retriever
        self.document_access = document_access
        self.audit_service = audit_service

        self.fail_closed = bool(fail_closed)
        self.allow_public_documents = bool(
            allow_public_documents
        )
        self.admin_bypass_acl = bool(
            admin_bypass_acl
        )

    # ========================================================
    # Search
    # ========================================================

    async def search(
        self,
        query: str,
        *,
        access_context: SearchAccessContext,
        top_k: int = 10,
        candidate_k: int | None = None,
        document_ids: Sequence[
            str | UUID
        ] | None = None,
        contract_ids: Sequence[
            str | UUID
        ] | None = None,
        filters: Mapping[str, Any] | None = None,
        min_score: float = 0.0,
        retriever_kwargs: Mapping[
            str,
            Any,
        ] | None = None,
    ) -> list[PermissionAwareSearchResult]:
        query = query.strip()

        if not query:
            return []

        if top_k <= 0:
            return []

        self._validate_context(
            access_context
        )

        organization_id = (
            access_context
            .normalized_organization_id()
        )

        candidate_k = (
            candidate_k
            or max(top_k * 4, top_k)
        )

        kwargs = dict(
            retriever_kwargs or {}
        )

        # Security-sensitive values always override
        # caller-provided kwargs.
        kwargs.update(
            {
                "organization_id":
                    organization_id,
                "top_k":
                    candidate_k,
                "document_ids":
                    document_ids,
                "contract_ids":
                    contract_ids,
                "filters":
                    filters,
            }
        )

        kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }

        raw_results = await self._retrieve(
            query,
            kwargs,
        )

        authorized: list[
            PermissionAwareSearchResult
        ] = []

        denied_count = 0

        for result in raw_results:
            data = self._result_mapping(
                result
            )

            decision = await self.check_access(
                data,
                access_context=access_context,
            )

            if not decision.allowed:
                denied_count += 1

                await self._audit_denied(
                    query=query,
                    data=data,
                    decision=decision,
                    context=access_context,
                )

                continue

            score = self._score(data)

            if score < min_score:
                continue

            authorized.append(
                self._to_result(
                    data,
                    rank=len(authorized) + 1,
                    permission_reason=(
                        decision.reason
                    ),
                )
            )

            if len(authorized) >= top_k:
                break

        await self._audit_search(
            query=query,
            context=access_context,
            retrieved=len(raw_results),
            returned=len(authorized),
            denied=denied_count,
        )

        return authorized

    async def retrieve(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[PermissionAwareSearchResult]:
        return await self.search(
            query,
            **kwargs,
        )

    async def search_dicts(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        results = await self.search(
            query,
            **kwargs,
        )

        return [
            result.to_dict()
            for result in results
        ]

    # ========================================================
    # Batch
    # ========================================================

    async def batch_search(
        self,
        queries: Sequence[str],
        *,
        access_context: SearchAccessContext,
        top_k: int = 10,
        **kwargs: Any,
    ) -> dict[
        str,
        list[PermissionAwareSearchResult],
    ]:
        tasks = [
            self.search(
                query,
                access_context=access_context,
                top_k=top_k,
                **kwargs,
            )
            for query in queries
        ]

        responses = await asyncio.gather(
            *tasks
        )

        return {
            query: response
            for query, response in zip(
                queries,
                responses,
                strict=True,
            )
        }

    # ========================================================
    # Permission check
    # ========================================================

    async def check_access(
        self,
        resource: Any,
        *,
        access_context: SearchAccessContext,
    ) -> PermissionDecision:
        data = self._result_mapping(
            resource
        )

        rules: list[str] = []

        user_id = (
            access_context
            .normalized_user_id()
        )

        organization_id = (
            access_context
            .normalized_organization_id()
        )

        document_id = self._string(
            self._value(
                data,
                "document_id",
            )
        )

        resource_id = self._resource_id(
            data
        )

        # ----------------------------------------------------
        # 1. Tenant isolation
        # ----------------------------------------------------

        rules.append("tenant")

        resource_org = self._string(
            self._value(
                data,
                "organization_id",
            )
        )

        if resource_org is None:
            if self.fail_closed:
                return PermissionDecision(
                    allowed=False,
                    reason=(
                        "missing_organization_id"
                    ),
                    document_id=document_id,
                    resource_id=resource_id,
                    rules_checked=rules,
                )

        elif resource_org != organization_id:
            return PermissionDecision(
                allowed=False,
                reason="tenant_mismatch",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 2. Explicit deny
        # ----------------------------------------------------

        rules.append("explicit_deny")

        denied_users = self._set(
            self._first(
                data,
                "denied_user_ids",
                "deny_user_ids",
                "blocked_user_ids",
            )
        )

        if user_id in denied_users:
            return PermissionDecision(
                allowed=False,
                reason="explicit_user_deny",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        denied_roles = self._casefold_set(
            self._first(
                data,
                "denied_roles",
                "deny_roles",
            )
        )

        if (
            denied_roles
            & access_context.normalized_roles()
        ):
            return PermissionDecision(
                allowed=False,
                reason="explicit_role_deny",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        denied_departments = self._set(
            self._first(
                data,
                "denied_department_ids",
                "deny_department_ids",
            )
        )

        if (
            denied_departments
            & access_context
            .normalized_departments()
        ):
            return PermissionDecision(
                allowed=False,
                reason="explicit_department_deny",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 3. External access backend
        # ----------------------------------------------------

        rules.append(
            "document_access_service"
        )

        backend_decision = (
            await self._check_access_backend(
                data,
                access_context,
            )
        )

        if backend_decision is False:
            return PermissionDecision(
                allowed=False,
                reason="document_access_denied",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        if backend_decision is True:
            return PermissionDecision(
                allowed=True,
                reason="document_access_allowed",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 4. Admin bypass
        # ----------------------------------------------------

        rules.append("admin")

        roles = (
            access_context
            .normalized_roles()
        )

        if (
            self.admin_bypass_acl
            and (
                access_context.is_superuser
                or access_context.is_admin
                or bool(
                    roles
                    & self.ADMIN_ROLES
                )
            )
        ):
            return PermissionDecision(
                allowed=True,
                reason="administrator",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 5. Owner
        # ----------------------------------------------------

        rules.append("owner")

        owner_id = self._string(
            self._first(
                data,
                "owner_id",
                "created_by",
                "uploaded_by",
                "user_id",
            )
        )

        if owner_id == user_id:
            return PermissionDecision(
                allowed=True,
                reason="resource_owner",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 6. Explicit users
        # ----------------------------------------------------

        rules.append("user_acl")

        allowed_users = self._set(
            self._first(
                data,
                "allowed_user_ids",
                "user_ids",
                "shared_with_user_ids",
                "permitted_user_ids",
            )
        )

        if (
            allowed_users
            and user_id in allowed_users
        ):
            return PermissionDecision(
                allowed=True,
                reason="explicit_user_allow",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 7. Role ACL
        # ----------------------------------------------------

        rules.append("role_acl")

        allowed_roles = self._casefold_set(
            self._first(
                data,
                "allowed_roles",
                "roles",
                "permitted_roles",
            )
        )

        if (
            allowed_roles
            and (
                allowed_roles
                & access_context
                .normalized_roles()
            )
        ):
            return PermissionDecision(
                allowed=True,
                reason="role_allowed",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 8. Department ACL
        # ----------------------------------------------------

        rules.append("department_acl")

        allowed_departments = self._set(
            self._first(
                data,
                "allowed_department_ids",
                "department_ids",
            )
        )

        if (
            allowed_departments
            and (
                allowed_departments
                & access_context
                .normalized_departments()
            )
        ):
            return PermissionDecision(
                allowed=True,
                reason="department_allowed",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 9. Team ACL
        # ----------------------------------------------------

        rules.append("team_acl")

        allowed_teams = self._set(
            self._first(
                data,
                "allowed_team_ids",
                "team_ids",
            )
        )

        if (
            allowed_teams
            and (
                allowed_teams
                & access_context
                .normalized_teams()
            )
        ):
            return PermissionDecision(
                allowed=True,
                reason="team_allowed",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 10. Group ACL
        # ----------------------------------------------------

        rules.append("group_acl")

        allowed_groups = self._set(
            self._first(
                data,
                "allowed_group_ids",
                "group_ids",
            )
        )

        if (
            allowed_groups
            and (
                allowed_groups
                & access_context
                .normalized_groups()
            )
        ):
            return PermissionDecision(
                allowed=True,
                reason="group_allowed",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 11. Matter / legal-case ACL
        # ----------------------------------------------------

        rules.append("matter_acl")

        matter_id = self._string(
            self._first(
                data,
                "matter_id",
                "case_id",
            )
        )

        allowed_matters = self._set(
            self._first(
                data,
                "allowed_matter_ids",
                "matter_ids",
            )
        )

        user_matters = (
            access_context
            .normalized_matters()
        )

        if (
            matter_id
            and matter_id in user_matters
        ):
            return PermissionDecision(
                allowed=True,
                reason="matter_member",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        if (
            allowed_matters
            and (
                allowed_matters
                & user_matters
            )
        ):
            return PermissionDecision(
                allowed=True,
                reason="matter_acl_allowed",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 12. Visibility
        # ----------------------------------------------------

        rules.append("visibility")

        visibility = self._visibility(
            self._first(
                data,
                "visibility",
                "access_level",
            )
        )

        if (
            visibility
            == DocumentVisibility.PUBLIC
            and self.allow_public_documents
        ):
            return PermissionDecision(
                allowed=True,
                reason="public_document",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        if visibility in {
            DocumentVisibility.ORGANIZATION,
            DocumentVisibility.INTERNAL,
        }:
            if self._has_read_permission(
                access_context
            ):
                return PermissionDecision(
                    allowed=True,
                    reason=(
                        "organization_document"
                    ),
                    document_id=document_id,
                    resource_id=resource_id,
                    rules_checked=rules,
                )

        if visibility in {
            DocumentVisibility.PRIVATE,
            DocumentVisibility.RESTRICTED,
        }:
            return PermissionDecision(
                allowed=False,
                reason=(
                    "restricted_document"
                ),
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 13. Generic readable flag
        # ----------------------------------------------------

        rules.append("readable")

        readable = self._first(
            data,
            "is_readable",
            "can_read",
        )

        if readable is True:
            return PermissionDecision(
                allowed=True,
                reason="readable_flag",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        # ----------------------------------------------------
        # 14. Default
        # ----------------------------------------------------

        if self.fail_closed:
            return PermissionDecision(
                allowed=False,
                reason="no_allow_rule_matched",
                document_id=document_id,
                resource_id=resource_id,
                rules_checked=rules,
            )

        return PermissionDecision(
            allowed=True,
            reason="fail_open",
            document_id=document_id,
            resource_id=resource_id,
            rules_checked=rules,
        )

    # ========================================================
    # Filter already-retrieved results
    # ========================================================

    async def filter_results(
        self,
        results: Sequence[Any],
        *,
        access_context: SearchAccessContext,
        top_k: int | None = None,
    ) -> list[PermissionAwareSearchResult]:
        self._validate_context(
            access_context
        )

        output: list[
            PermissionAwareSearchResult
        ] = []

        for item in results:
            data = self._result_mapping(item)

            decision = await self.check_access(
                data,
                access_context=access_context,
            )

            if not decision.allowed:
                continue

            output.append(
                self._to_result(
                    data,
                    rank=len(output) + 1,
                    permission_reason=(
                        decision.reason
                    ),
                )
            )

            if (
                top_k is not None
                and len(output) >= top_k
            ):
                break

        return output

    # ========================================================
    # Access backend
    # ========================================================

    async def _check_access_backend(
        self,
        data: Mapping[str, Any],
        context: SearchAccessContext,
    ) -> bool | None:
        """
        Returns:
            True  -> backend explicitly allowed
            False -> backend explicitly denied
            None  -> backend unavailable/no decision
        """

        if self.document_access is None:
            return None

        document_id = self._string(
            self._value(
                data,
                "document_id",
            )
        )

        if not document_id:
            return None

        methods = (
            "can_read",
            "can_access",
            "check_access",
            "has_access",
            "check_document_access",
        )

        for method_name in methods:
            method = getattr(
                self.document_access,
                method_name,
                None,
            )

            if method is None:
                continue

            attempts = (
                {
                    "document_id":
                        document_id,
                    "user_id":
                        context.normalized_user_id(),
                    "organization_id":
                        context
                        .normalized_organization_id(),
                    "permission":
                        "read",
                },
                {
                    "document_id":
                        document_id,
                    "user_id":
                        context.normalized_user_id(),
                    "organization_id":
                        context
                        .normalized_organization_id(),
                },
                {
                    "document_id":
                        document_id,
                    "user_id":
                        context.normalized_user_id(),
                },
            )

            for kwargs in attempts:
                try:
                    result = method(
                        **kwargs
                    )

                    result = (
                        await self
                        ._await_if_needed(
                            result
                        )
                    )

                    return self._access_result(
                        result
                    )

                except TypeError:
                    continue

                except Exception as exc:
                    logger.exception(
                        "Document access backend "
                        "failed for document %s",
                        document_id,
                    )

                    if self.fail_closed:
                        raise PermissionBackendError(
                            "Document access backend "
                            "failed"
                        ) from exc

                    return None

        return None

    @staticmethod
    def _access_result(
        value: Any,
    ) -> bool | None:
        if isinstance(value, bool):
            return value

        if value is None:
            return None

        if isinstance(value, Mapping):
            for key in (
                "allowed",
                "can_read",
                "has_access",
                "permitted",
            ):
                if key in value:
                    return bool(value[key])

        for attr in (
            "allowed",
            "can_read",
            "has_access",
            "permitted",
        ):
            if hasattr(value, attr):
                return bool(
                    getattr(value, attr)
                )

        return None

    # ========================================================
    # Underlying retrieval
    # ========================================================

    async def _retrieve(
        self,
        query: str,
        kwargs: Mapping[str, Any],
    ) -> list[Any]:
        method = None

        for method_name in (
            "search",
            "retrieve",
            "query",
            "similarity_search",
        ):
            method = getattr(
                self.retriever,
                method_name,
                None,
            )

            if method is not None:
                break

        if method is None:
            raise PermissionSearchValidationError(
                "Retriever exposes no supported "
                "search method"
            )

        try:
            result = method(
                query,
                **dict(kwargs),
            )

            result = (
                await self._await_if_needed(
                    result
                )
            )

            return list(result or [])

        except TypeError:
            try:
                result = method(
                    query=query,
                    **dict(kwargs),
                )

                result = (
                    await self
                    ._await_if_needed(
                        result
                    )
                )

                return list(
                    result or []
                )

            except Exception as exc:
                raise PermissionSearchError(
                    "Permission-aware retrieval "
                    "failed"
                ) from exc

        except Exception as exc:
            raise PermissionSearchError(
                "Permission-aware retrieval failed"
            ) from exc

    # ========================================================
    # Result mapping
    # ========================================================

    @staticmethod
    def _result_mapping(
        result: Any,
    ) -> dict[str, Any]:
        if result is None:
            return {}

        if isinstance(
            result,
            Mapping,
        ):
            data = dict(result)

        elif hasattr(
            result,
            "to_dict",
        ):
            value = result.to_dict()

            data = (
                dict(value)
                if isinstance(
                    value,
                    Mapping,
                )
                else {}
            )

        elif hasattr(
            result,
            "model_dump",
        ):
            value = result.model_dump()

            data = (
                dict(value)
                if isinstance(
                    value,
                    Mapping,
                )
                else {}
            )

        else:
            data = {}

            fields = (
                "id",
                "chunk_id",
                "point_id",
                "text",
                "content",
                "chunk_text",
                "score",
                "rank",
                "organization_id",
                "document_id",
                "contract_id",
                "clause_id",
                "parent_id",
                "parent_chunk_id",
                "title",
                "page_number",
                "chunk_index",
                "metadata",
                "metadata_json",
                "payload",
                "sources",
                "matched_terms",
                "visibility",
                "access_level",
                "owner_id",
                "created_by",
                "uploaded_by",
                "allowed_user_ids",
                "denied_user_ids",
                "allowed_roles",
                "denied_roles",
                "department_ids",
                "team_ids",
                "group_ids",
                "matter_id",
                "matter_ids",
            )

            for name in fields:
                if hasattr(
                    result,
                    name,
                ):
                    data[name] = getattr(
                        result,
                        name,
                    )

        # Flatten payload/metadata while preserving
        # explicit top-level values.
        for key in (
            "metadata",
            "metadata_json",
            "payload",
        ):
            metadata = data.get(key)

            if isinstance(
                metadata,
                Mapping,
            ):
                for (
                    metadata_key,
                    metadata_value,
                ) in metadata.items():
                    data.setdefault(
                        metadata_key,
                        metadata_value,
                    )

        return data

    # ========================================================
    # Field lookup
    # ========================================================

    @staticmethod
    def _value(
        data: Mapping[str, Any],
        key: str,
    ) -> Any:
        if key in data:
            return data[key]

        for container_name in (
            "metadata",
            "metadata_json",
            "payload",
            "access",
            "acl",
            "permissions",
        ):
            container = data.get(
                container_name
            )

            if (
                isinstance(
                    container,
                    Mapping,
                )
                and key in container
            ):
                return container[key]

        return None

    @classmethod
    def _first(
        cls,
        data: Mapping[str, Any],
        *keys: str,
    ) -> Any:
        for key in keys:
            value = cls._value(
                data,
                key,
            )

            if value is not None:
                return value

        return None

    # ========================================================
    # Result conversion
    # ========================================================

    @classmethod
    def _to_result(
        cls,
        data: Mapping[str, Any],
        *,
        rank: int,
        permission_reason: str,
    ) -> PermissionAwareSearchResult:
        metadata = cls._metadata(data)

        sources = data.get(
            "sources",
            [],
        )

        if isinstance(sources, str):
            sources = [sources]

        matched_terms = data.get(
            "matched_terms",
            [],
        )

        if isinstance(
            matched_terms,
            str,
        ):
            matched_terms = [
                matched_terms
            ]

        return PermissionAwareSearchResult(
            id=(
                cls._resource_id(data)
                or f"result-{rank}"
            ),
            text=str(
                data.get("text")
                or data.get("content")
                or data.get(
                    "chunk_text"
                )
                or metadata.get("text")
                or ""
            ),
            score=cls._score(data),
            rank=rank,
            organization_id=cls._string(
                cls._value(
                    data,
                    "organization_id",
                )
            ),
            document_id=cls._string(
                cls._value(
                    data,
                    "document_id",
                )
            ),
            contract_id=cls._string(
                cls._value(
                    data,
                    "contract_id",
                )
            ),
            clause_id=cls._string(
                cls._value(
                    data,
                    "clause_id",
                )
            ),
            parent_id=cls._string(
                cls._first(
                    data,
                    "parent_id",
                    "parent_chunk_id",
                )
            ),
            title=cls._first(
                data,
                "title",
                "document_title",
            ),
            page_number=cls._int(
                cls._first(
                    data,
                    "page_number",
                    "page",
                )
            ),
            chunk_index=cls._int(
                cls._value(
                    data,
                    "chunk_index",
                )
            ),
            metadata=metadata,
            sources=[
                str(value)
                for value in (
                    sources or []
                )
            ],
            matched_terms=[
                str(value)
                for value in (
                    matched_terms or []
                )
            ],
            permission_reason=(
                permission_reason
            ),
        )

    @staticmethod
    def _metadata(
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}

        for name in (
            "metadata",
            "metadata_json",
            "payload",
        ):
            value = data.get(name)

            if isinstance(
                value,
                Mapping,
            ):
                output.update(value)

        return output

    # ========================================================
    # Permissions
    # ========================================================

    @classmethod
    def _has_read_permission(
        cls,
        context: SearchAccessContext,
    ) -> bool:
        permissions = (
            context.normalized_permissions()
        )

        return bool(
            permissions
            & cls.READ_PERMISSIONS
        )

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_context(
        context: SearchAccessContext,
    ) -> None:
        if context is None:
            raise PermissionSearchValidationError(
                "access_context is required"
            )

        if not context.user_id:
            raise PermissionSearchValidationError(
                "user_id is required"
            )

        if not context.organization_id:
            raise PermissionSearchValidationError(
                "organization_id is required"
            )

    # ========================================================
    # Utility
    # ========================================================

    @staticmethod
    def _resource_id(
        data: Mapping[str, Any],
    ) -> str | None:
        value = (
            data.get("id")
            or data.get("chunk_id")
            or data.get("point_id")
        )

        return (
            str(value)
            if value is not None
            else None
        )

    @staticmethod
    def _score(
        data: Mapping[str, Any],
    ) -> float:
        for key in (
            "score",
            "rerank_score",
            "hybrid_score",
            "relevance_score",
            "similarity",
        ):
            value = data.get(key)

            if value is None:
                continue

            try:
                return float(value)

            except (
                TypeError,
                ValueError,
            ):
                continue

        return 0.0

    @staticmethod
    def _string(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if isinstance(value, Enum):
            value = value.value

        return str(value)

    @staticmethod
    def _int(
        value: Any,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(value)

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _set(
        value: Any,
    ) -> set[str]:
        if value is None:
            return set()

        if isinstance(
            value,
            (list, tuple, set, frozenset),
        ):
            return {
                str(item)
                for item in value
                if item is not None
            }

        return {str(value)}

    @classmethod
    def _casefold_set(
        cls,
        value: Any,
    ) -> set[str]:
        return {
            item.casefold()
            for item in cls._set(
                value
            )
        }

    @staticmethod
    def _visibility(
        value: Any,
    ) -> DocumentVisibility | None:
        if value is None:
            return None

        if isinstance(
            value,
            DocumentVisibility,
        ):
            return value

        if isinstance(value, Enum):
            value = value.value

        normalized = (
            str(value)
            .strip()
            .casefold()
        )

        aliases = {
            "org":
                DocumentVisibility.ORGANIZATION,
            "tenant":
                DocumentVisibility.ORGANIZATION,
            "company":
                DocumentVisibility.ORGANIZATION,
            "company_wide":
                DocumentVisibility.ORGANIZATION,
            "org_wide":
                DocumentVisibility.ORGANIZATION,
            "restricted":
                DocumentVisibility.RESTRICTED,
            "private":
                DocumentVisibility.PRIVATE,
            "internal":
                DocumentVisibility.INTERNAL,
            "public":
                DocumentVisibility.PUBLIC,
        }

        return aliases.get(normalized)

    @staticmethod
    async def _await_if_needed(
        value: Any,
    ) -> Any:
        if inspect.isawaitable(value):
            return await value

        return value

    # ========================================================
    # Audit
    # ========================================================

    async def _audit_search(
        self,
        *,
        query: str,
        context: SearchAccessContext,
        retrieved: int,
        returned: int,
        denied: int,
    ) -> None:
        if self.audit_service is None:
            return

        try:
            method = getattr(
                self.audit_service,
                "log_search",
                None,
            )

            if method is not None:
                value = method(
                    organization_id=(
                        context
                        .normalized_organization_id()
                    ),
                    user_id=(
                        context
                        .normalized_user_id()
                    ),
                    query=query,
                    result_count=returned,
                    metadata={
                        "retrieved": retrieved,
                        "returned": returned,
                        "denied": denied,
                        "permission_aware": True,
                    },
                )

                await self._await_if_needed(
                    value
                )

                return

            method = getattr(
                self.audit_service,
                "log",
                None,
            )

            if method is not None:
                value = method(
                    organization_id=(
                        context
                        .normalized_organization_id()
                    ),
                    user_id=(
                        context
                        .normalized_user_id()
                    ),
                    action="rag_search",
                    entity_type="search",
                    metadata={
                        "query": query,
                        "retrieved": retrieved,
                        "returned": returned,
                        "denied": denied,
                        "permission_aware": True,
                    },
                )

                await self._await_if_needed(
                    value
                )

        except Exception:
            logger.exception(
                "Search audit logging failed"
            )

    async def _audit_denied(
        self,
        *,
        query: str,
        data: Mapping[str, Any],
        decision: PermissionDecision,
        context: SearchAccessContext,
    ) -> None:
        if self.audit_service is None:
            return

        try:
            method = getattr(
                self.audit_service,
                "log_security_event",
                None,
            )

            if method is None:
                return

            value = method(
                organization_id=(
                    context
                    .normalized_organization_id()
                ),
                user_id=(
                    context
                    .normalized_user_id()
                ),
                event_type=(
                    "rag_access_denied"
                ),
                severity="medium",
                metadata={
                    "query": query,
                    "document_id":
                        decision.document_id,
                    "resource_id":
                        decision.resource_id,
                    "reason":
                        decision.reason,
                    "rules_checked":
                        decision.rules_checked,
                },
            )

            await self._await_if_needed(
                value
            )

        except Exception:
            logger.exception(
                "Denied retrieval audit "
                "logging failed"
            )

    # ========================================================
    # Health
    # ========================================================

    async def health_check(
        self,
    ) -> dict[str, Any]:
        retriever_health = None
        access_health = None

        method = getattr(
            self.retriever,
            "health_check",
            None,
        )

        if method is not None:
            try:
                retriever_health = method()

                retriever_health = (
                    await self
                    ._await_if_needed(
                        retriever_health
                    )
                )

            except Exception as exc:
                retriever_health = {
                    "status": "unhealthy",
                    "error": str(exc),
                }

        if self.document_access is not None:
            method = getattr(
                self.document_access,
                "health_check",
                None,
            )

            if method is not None:
                try:
                    access_health = method()

                    access_health = (
                        await self
                        ._await_if_needed(
                            access_health
                        )
                    )

                except Exception as exc:
                    access_health = {
                        "status": "unhealthy",
                        "error": str(exc),
                    }

        return {
            "service":
                "permission_aware_search",
            "status":
                "healthy",
            "fail_closed":
                self.fail_closed,
            "allow_public_documents":
                self.allow_public_documents,
            "admin_bypass_acl":
                self.admin_bypass_acl,
            "document_access_backend":
                self.document_access
                is not None,
            "retriever":
                retriever_health,
            "document_access":
                access_health,
        }


# ============================================================
# Compatibility aliases
# ============================================================

PermissionAwareRetriever = PermissionAwareSearch
SecureSearch = PermissionAwareSearch
AccessContext = SearchAccessContext