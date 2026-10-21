"""
Enterprise Legal Document Intelligence AI
------------------------------------------

Distributed API Rate Limiting Middleware

Features
--------
- Redis-backed distributed rate limiting.
- Fixed-window and sliding-window algorithms.
- Per-IP limits for unauthenticated traffic.
- Per-user limits for authenticated traffic.
- Per-tenant limits for multi-tenant APIs.
- Endpoint-specific limits.
- Stricter authentication/security limits.
- Retry-After and rate-limit headers.
- Fail-open / fail-closed configuration.
- Security-safe logging and telemetry.
- No passwords, tokens, query strings, or document content in keys.
- Works across multiple API replicas/workers.

Recommended production topology
--------------------------------
Client
  |
Load Balancer
  |
FastAPI
  |
RateLimitMiddleware
  |
Redis
  |
Application routes

Environment variables
----------------------
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REDIS_URL=redis://redis:6379/2
RATE_LIMIT_ALGORITHM=sliding_window
RATE_LIMIT_FAIL_OPEN=true

RATE_LIMIT_DEFAULT_REQUESTS=120
RATE_LIMIT_DEFAULT_WINDOW_SECONDS=60

RATE_LIMIT_AUTH_REQUESTS=10
RATE_LIMIT_AUTH_WINDOW_SECONDS=60

RATE_LIMIT_SEARCH_REQUESTS=60
RATE_LIMIT_SEARCH_WINDOW_SECONDS=60

RATE_LIMIT_CHAT_REQUESTS=30
RATE_LIMIT_CHAT_WINDOW_SECONDS=60

RATE_LIMIT_UPLOAD_REQUESTS=20
RATE_LIMIT_UPLOAD_WINDOW_SECONDS=60

RATE_LIMIT_AI_REQUESTS=30
RATE_LIMIT_AI_WINDOW_SECONDS=60
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.app.core.logging import get_logger
from backend.app.core.telemetry import telemetry


logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class RateLimitAlgorithm(str, Enum):
    """
    Supported rate-limiting algorithms.
    """

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"


class RateLimitIdentity(str, Enum):
    """
    Identity used to construct the rate-limit key.
    """

    IP = "ip"
    USER = "user"
    TENANT = "tenant"
    API_KEY = "api_key"


class RateLimitDecision(str, Enum):
    """
    Result of a rate-limit evaluation.
    """

    ALLOW = "allow"
    DENY = "deny"
    BYPASS = "bypass"
    ERROR = "error"


# ============================================================================
# EXCEPTIONS
# ============================================================================


class RateLimiterError(Exception):
    """
    Base rate limiter error.
    """


class RateLimiterConfigurationError(
    RateLimiterError
):
    """
    Invalid rate limiter configuration.
    """


class RateLimiterBackendError(
    RateLimiterError
):
    """
    Redis/backend operation failed.
    """


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True)
class RateLimitRule:
    """
    Rate-limit rule for an endpoint or logical API category.
    """

    name: str

    requests: int

    window_seconds: int

    identity: RateLimitIdentity = (
        RateLimitIdentity.IP
    )

    enabled: bool = True

    methods: frozenset[str] = frozenset()

    paths: tuple[str, ...] = ()

    priority: int = 100

    description: str = ""

    def __post_init__(self) -> None:
        if self.requests <= 0:
            raise RateLimiterConfigurationError(
                "Rate-limit requests must be greater than zero."
            )

        if self.window_seconds <= 0:
            raise RateLimiterConfigurationError(
                "Rate-limit window must be greater than zero."
            )

    def matches(
        self,
        request: Request,
    ) -> bool:
        if not self.enabled:
            return False

        if self.methods:
            if request.method.upper() not in {
                method.upper()
                for method in self.methods
            }:
                return False

        if not self.paths:
            return True

        path = request.url.path

        return any(
            path == configured
            or path.startswith(
                configured.rstrip("/")
                + "/"
            )
            for configured in self.paths
        )


@dataclass(frozen=True)
class RateLimitResult:
    """
    Result of a rate-limit check.
    """

    allowed: bool

    decision: RateLimitDecision

    limit: int

    remaining: int

    reset_after: int

    retry_after: Optional[int] = None

    rule_name: Optional[str] = None

    identity_type: Optional[
        RateLimitIdentity
    ] = None

    backend_error: bool = False

    def headers(self) -> dict[str, str]:
        """
        Generate standards-friendly rate-limit headers.
        """

        headers = {
            "X-RateLimit-Limit": str(
                self.limit
            ),
            "X-RateLimit-Remaining": str(
                max(
                    0,
                    self.remaining,
                )
            ),
            "X-RateLimit-Reset": str(
                max(
                    0,
                    self.reset_after,
                )
            ),
        }

        if self.retry_after is not None:
            headers["Retry-After"] = str(
                max(
                    1,
                    self.retry_after,
                )
            )

        return headers


@dataclass
class RateLimiterConfig:
    """
    Runtime rate limiter configuration.
    """

    enabled: bool = True

    redis_url: str = (
        "redis://localhost:6379/2"
    )

    algorithm: RateLimitAlgorithm = (
        RateLimitAlgorithm.SLIDING_WINDOW
    )

    fail_open: bool = True

    key_prefix: str = (
        "legal-ai:rate-limit"
    )

    default_requests: int = 120

    default_window_seconds: int = 60

    auth_requests: int = 10

    auth_window_seconds: int = 60

    search_requests: int = 60

    search_window_seconds: int = 60

    chat_requests: int = 30

    chat_window_seconds: int = 60

    upload_requests: int = 20

    upload_window_seconds: int = 60

    ai_requests: int = 30

    ai_window_seconds: int = 60

    trusted_proxies: tuple[str, ...] = ()

    excluded_paths: set[str] = field(
        default_factory=lambda: {
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        }
    )

    excluded_ips: set[str] = field(
        default_factory=set
    )

    excluded_roles: set[str] = field(
        default_factory=lambda: {
            "super_admin",
        }
    )

    max_key_length: int = 200


# ============================================================================
# REDIS BACKEND
# ============================================================================


class RedisRateLimitBackend:
    """
    Async Redis backend.

    Redis is imported lazily so local unit tests can run without Redis
    installed when rate limiting is disabled or mocked.
    """

    def __init__(
        self,
        redis_url: str,
    ) -> None:
        self.redis_url = redis_url
        self._redis: Any = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        if self._redis is not None:
            return

        async with self._lock:
            if self._redis is not None:
                return

            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                )

                await self._redis.ping()

            except Exception as exc:
                self._redis = None

                raise RateLimiterBackendError(
                    "Unable to connect to rate-limit backend."
                ) from exc

    async def close(self) -> None:
        if self._redis is None:
            return

        try:
            await self._redis.close()
        except Exception:
            logger.debug(
                "rate_limit_redis_close_failed",
                exc_info=True,
            )
        finally:
            self._redis = None

    async def fixed_window(
        self,
        *,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """
        Atomic-ish fixed-window counter.

        INCR + EXPIRE is sufficient for practical API limiting,
        while the key remains isolated by the application prefix.
        """

        await self.connect()

        try:
            count = await self._redis.incr(
                key
            )

            if count == 1:
                await self._redis.expire(
                    key,
                    window_seconds,
                )

            ttl = await self._redis.ttl(
                key
            )

            remaining = max(
                0,
                limit - int(count),
            )

            allowed = (
                int(count) <= limit
            )

            return RateLimitResult(
                allowed=allowed,
                decision=(
                    RateLimitDecision.ALLOW
                    if allowed
                    else RateLimitDecision.DENY
                ),
                limit=limit,
                remaining=remaining,
                reset_after=max(
                    0,
                    int(ttl),
                ),
                retry_after=(
                    max(
                        1,
                        int(ttl),
                    )
                    if not allowed
                    else None
                ),
            )

        except Exception as exc:
            raise RateLimiterBackendError(
                "Rate-limit backend operation failed."
            ) from exc

    async def sliding_window(
        self,
        *,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> RateLimitResult:
        """
        Sliding-window rate limiter using Redis sorted sets.

        Each request is represented by a unique timestamp/member.
        Expired entries are removed before counting.
        """

        await self.connect()

        now = time.time()

        window_start = (
            now - window_seconds
        )

        member = (
            f"{now:.6f}:"
            f"{id(asyncio.current_task())}"
        )

        try:
            pipeline = self._redis.pipeline(
                transaction=True
            )

            pipeline.zremrangebyscore(
                key,
                0,
                window_start,
            )

            pipeline.zcard(
                key
            )

            results = await pipeline.execute()

            current_count = int(
                results[-1]
            )

            if current_count >= limit:
                oldest = await self._redis.zrange(
                    key,
                    0,
                    0,
                    withscores=True,
                )

                retry_after = 1

                if oldest:
                    oldest_timestamp = float(
                        oldest[0][1]
                    )

                    retry_after = max(
                        1,
                        int(
                            (
                                oldest_timestamp
                                + window_seconds
                            )
                            - now
                        ),
                    )

                return RateLimitResult(
                    allowed=False,
                    decision=RateLimitDecision.DENY,
                    limit=limit,
                    remaining=0,
                    reset_after=retry_after,
                    retry_after=retry_after,
                )

            await self._redis.zadd(
                key,
                {
                    member: now,
                },
            )

            await self._redis.expire(
                key,
                window_seconds + 1,
            )

            new_count = current_count + 1

            return RateLimitResult(
                allowed=True,
                decision=RateLimitDecision.ALLOW,
                limit=limit,
                remaining=max(
                    0,
                    limit - new_count,
                ),
                reset_after=window_seconds,
            )

        except Exception as exc:
            raise RateLimiterBackendError(
                "Sliding-window rate-limit operation failed."
            ) from exc


# ============================================================================
# RATE LIMITER SERVICE
# ============================================================================


class RateLimiter:
    """
    Application-level distributed rate limiter.
    """

    def __init__(
        self,
        config: Optional[
            RateLimiterConfig
        ] = None,
        backend: Optional[
            RedisRateLimitBackend
        ] = None,
    ) -> None:
        self.config = (
            config
            or RateLimiterConfig()
        )

        self.backend = (
            backend
            or RedisRateLimitBackend(
                self.config.redis_url
            )
        )

        self.rules: list[
            RateLimitRule
        ] = []

        self._install_default_rules()

    # ------------------------------------------------------------------
    # RULES
    # ------------------------------------------------------------------

    def _install_default_rules(
        self,
    ) -> None:
        """
        Install application-specific defaults.

        More specific rules have higher priority.
        """

        self.rules.extend(
            [
                RateLimitRule(
                    name="authentication",
                    requests=self.config.auth_requests,
                    window_seconds=self.config.auth_window_seconds,
                    identity=RateLimitIdentity.IP,
                    methods=frozenset(
                        {"POST"}
                    ),
                    paths=(
                        "/api/v1/auth/login",
                        "/api/v1/auth/refresh",
                        "/api/v1/auth/change-password",
                    ),
                    priority=10,
                    description=(
                        "Protect authentication endpoints "
                        "against brute-force abuse."
                    ),
                ),
                RateLimitRule(
                    name="chat",
                    requests=self.config.chat_requests,
                    window_seconds=self.config.chat_window_seconds,
                    identity=RateLimitIdentity.USER,
                    methods=frozenset(
                        {"POST"}
                    ),
                    paths=(
                        "/api/v1/chat",
                        "/api/v1/chat/",
                    ),
                    priority=20,
                    description=(
                        "Limit expensive legal AI chat requests."
                    ),
                ),
                RateLimitRule(
                    name="search",
                    requests=self.config.search_requests,
                    window_seconds=self.config.search_window_seconds,
                    identity=RateLimitIdentity.USER,
                    methods=frozenset(
                        {"GET", "POST"}
                    ),
                    paths=(
                        "/api/v1/search",
                        "/api/v1/search/",
                    ),
                    priority=30,
                    description=(
                        "Limit search and retrieval traffic."
                    ),
                ),
                RateLimitRule(
                    name="document_upload",
                    requests=self.config.upload_requests,
                    window_seconds=self.config.upload_window_seconds,
                    identity=RateLimitIdentity.USER,
                    methods=frozenset(
                        {"POST"}
                    ),
                    paths=(
                        "/api/v1/documents/upload",
                        "/api/v1/documents/",
                    ),
                    priority=40,
                    description=(
                        "Protect document ingestion resources."
                    ),
                ),
                RateLimitRule(
                    name="ai_operations",
                    requests=self.config.ai_requests,
                    window_seconds=self.config.ai_window_seconds,
                    identity=RateLimitIdentity.USER,
                    methods=frozenset(
                        {"POST"}
                    ),
                    paths=(
                        "/api/v1/risk",
                        "/api/v1/expiry",
                        "/api/v1/renewals",
                        "/api/v1/reviews",
                        "/api/v1/contracts",
                    ),
                    priority=50,
                    description=(
                        "Protect expensive AI/legal analysis operations."
                    ),
                ),
                RateLimitRule(
                    name="default",
                    requests=self.config.default_requests,
                    window_seconds=self.config.default_window_seconds,
                    identity=RateLimitIdentity.IP,
                    priority=1000,
                    description=(
                        "Default API protection."
                    ),
                ),
            ]
        )

    def add_rule(
        self,
        rule: RateLimitRule,
    ) -> None:
        self.rules.append(
            rule
        )

        self.rules.sort(
            key=lambda item: item.priority
        )

    def get_rule(
        self,
        request: Request,
    ) -> Optional[
        RateLimitRule
    ]:
        matching = [
            rule
            for rule in self.rules
            if rule.matches(request)
        ]

        if not matching:
            return None

        matching.sort(
            key=lambda rule: rule.priority
        )

        return matching[0]

    # ------------------------------------------------------------------
    # IDENTITY
    # ------------------------------------------------------------------

    def _extract_user(
        self,
        request: Request,
    ) -> Any:
        """
        Resolve the authenticated user from request.state.

        Authentication middleware/dependencies should populate this.
        """

        return getattr(
            request.state,
            "user",
            None,
        )

    def _extract_tenant_id(
        self,
        request: Request,
    ) -> Optional[str]:
        tenant_id = getattr(
            request.state,
            "tenant_id",
            None,
        )

        if tenant_id:
            return str(
                tenant_id
            )

        user = self._extract_user(
            request
        )

        if user is not None:
            for attribute in (
                "tenant_id",
                "organization_id",
            ):
                value = getattr(
                    user,
                    attribute,
                    None,
                )

                if value:
                    return str(
                        value
                    )

        return None

    def _extract_user_id(
        self,
        request: Request,
    ) -> Optional[str]:
        user = self._extract_user(
            request
        )

        if user is None:
            return None

        for attribute in (
            "id",
            "user_id",
            "sub",
        ):
            value = getattr(
                user,
                attribute,
                None,
            )

            if value:
                return str(
                    value
                )

        if isinstance(
            user,
            Mapping,
        ):
            for key in (
                "id",
                "user_id",
                "sub",
            ):
                value = user.get(
                    key
                )

                if value:
                    return str(
                        value
                    )

        return None

    def _extract_role(
        self,
        request: Request,
    ) -> Optional[str]:
        user = self._extract_user(
            request
        )

        if user is None:
            return None

        role = getattr(
            user,
            "role",
            None,
        )

        if role is None and isinstance(
            user,
            Mapping,
        ):
            role = user.get(
                "role"
            )

        if role is None:
            return None

        return str(
            getattr(
                role,
                "value",
                role,
            )
        ).lower()

    def _get_client_ip(
        self,
        request: Request,
    ) -> str:
        """
        Resolve client IP.

        X-Forwarded-For is only trusted when the application is configured
        behind a known trusted proxy. Otherwise request.client.host is used.
        """

        client_host = (
            request.client.host
            if request.client
            else "unknown"
        )

        if not self.config.trusted_proxies:
            return client_host

        forwarded_for = request.headers.get(
            "X-Forwarded-For"
        )

        if forwarded_for:
            return (
                forwarded_for
                .split(",")[0]
                .strip()
            )

        real_ip = request.headers.get(
            "X-Real-IP"
        )

        if real_ip:
            return real_ip.strip()

        return client_host

    def _hash_identity(
        self,
        identity: str,
    ) -> str:
        """
        Hash identifiers before storing them in Redis keys.

        Raw IP/user/tenant identifiers are therefore not persisted
        in the rate-limit key namespace.
        """

        digest = hashlib.sha256(
            identity.encode(
                "utf-8"
            )
        ).hexdigest()

        return digest[:32]

    def _identity_value(
        self,
        request: Request,
        identity_type: RateLimitIdentity,
    ) -> str:
        if identity_type == RateLimitIdentity.IP:
            return self._get_client_ip(
                request
            )

        if identity_type == RateLimitIdentity.USER:
            user_id = self._extract_user_id(
                request
            )

            if user_id:
                return user_id

            return self._get_client_ip(
                request
            )

        if identity_type == RateLimitIdentity.TENANT:
            tenant_id = self._extract_tenant_id(
                request
            )

            if tenant_id:
                return tenant_id

            return self._get_client_ip(
                request
            )

        if identity_type == RateLimitIdentity.API_KEY:
            # Never place the raw API key into Redis.
            api_key = request.headers.get(
                "X-API-Key"
            )

            if api_key:
                return self._hash_identity(
                    api_key
                )

            return self._get_client_ip(
                request
            )

        return self._get_client_ip(
            request
        )

    # ------------------------------------------------------------------
    # BYPASS
    # ------------------------------------------------------------------

    def should_bypass(
        self,
        request: Request,
    ) -> bool:
        if not self.config.enabled:
            return True

        path = request.url.path

        if path in self.config.excluded_paths:
            return True

        ip = self._get_client_ip(
            request
        )

        if ip in self.config.excluded_ips:
            return True

        role = self._extract_role(
            request
        )

        if role and role in self.config.excluded_roles:
            return True

        return False

    # ------------------------------------------------------------------
    # KEY
    # ------------------------------------------------------------------

    def build_key(
        self,
        request: Request,
        rule: RateLimitRule,
    ) -> str:
        identity = self._identity_value(
            request,
            rule.identity,
        )

        identity_hash = self._hash_identity(
            identity
        )

        path = request.url.path

        path_hash = hashlib.sha256(
            path.encode(
                "utf-8"
            )
        ).hexdigest()[:16]

        raw_key = (
            f"{self.config.key_prefix}:"
            f"{rule.name}:"
            f"{rule.identity.value}:"
            f"{identity_hash}:"
            f"{path_hash}"
        )

        return raw_key[
            : self.config.max_key_length
        ]

    # ------------------------------------------------------------------
    # CHECK
    # ------------------------------------------------------------------

    async def check(
        self,
        request: Request,
    ) -> RateLimitResult:
        if self.should_bypass(
            request
        ):
            return RateLimitResult(
                allowed=True,
                decision=RateLimitDecision.BYPASS,
                limit=0,
                remaining=0,
                reset_after=0,
            )

        rule = self.get_rule(
            request
        )

        if rule is None:
            return RateLimitResult(
                allowed=True,
                decision=RateLimitDecision.BYPASS,
                limit=0,
                remaining=0,
                reset_after=0,
            )

        key = self.build_key(
            request,
            rule,
        )

        try:
            if (
                self.config.algorithm
                == RateLimitAlgorithm.FIXED_WINDOW
            ):
                result = (
                    await self.backend.fixed_window(
                        key=key,
                        limit=rule.requests,
                        window_seconds=rule.window_seconds,
                    )
                )
            else:
                result = (
                    await self.backend.sliding_window(
                        key=key,
                        limit=rule.requests,
                        window_seconds=rule.window_seconds,
                    )
                )

            return RateLimitResult(
                allowed=result.allowed,
                decision=result.decision,
                limit=result.limit,
                remaining=result.remaining,
                reset_after=result.reset_after,
                retry_after=result.retry_after,
                rule_name=rule.name,
                identity_type=rule.identity,
            )

        except RateLimiterBackendError:
            logger.exception(
                "rate_limit_backend_error",
                extra={
                    "rule": rule.name,
                    "path": request.url.path,
                },
            )

            try:
                telemetry.increment(
                    "rate_limit.backend_errors",
                    1,
                )
            except Exception:
                pass

            if self.config.fail_open:
                return RateLimitResult(
                    allowed=True,
                    decision=RateLimitDecision.ERROR,
                    limit=rule.requests,
                    remaining=rule.requests,
                    reset_after=rule.window_seconds,
                    rule_name=rule.name,
                    identity_type=rule.identity,
                    backend_error=True,
                )

            return RateLimitResult(
                allowed=False,
                decision=RateLimitDecision.ERROR,
                limit=rule.requests,
                remaining=0,
                reset_after=rule.window_seconds,
                retry_after=1,
                rule_name=rule.name,
                identity_type=rule.identity,
                backend_error=True,
            )

    # ------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        if not self.config.enabled:
            logger.info(
                "rate_limiter_disabled"
            )
            return

        try:
            await self.backend.connect()

            logger.info(
                "rate_limiter_started",
                extra={
                    "algorithm": self.config.algorithm.value,
                },
            )

        except RateLimiterBackendError:
            if self.config.fail_open:
                logger.warning(
                    "rate_limiter_backend_unavailable_fail_open",
                    exc_info=True,
                )
            else:
                raise

    async def shutdown(self) -> None:
        await self.backend.close()


# ============================================================================
# MIDDLEWARE
# ============================================================================


class RateLimitMiddleware(
    BaseHTTPMiddleware
):
    """
    FastAPI/Starlette middleware enforcing distributed API rate limits.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        rate_limiter: Optional[
            RateLimiter
        ] = None,
        config: Optional[
            RateLimiterConfig
        ] = None,
    ) -> None:
        super().__init__(
            app
        )

        self.rate_limiter = (
            rate_limiter
            or RateLimiter(
                config=config
            )
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[
            [Request],
            Any,
        ],
    ) -> Any:
        started_at = time.perf_counter()

        result = await self.rate_limiter.check(
            request
        )

        try:
            telemetry.increment(
                "rate_limit.requests",
                1,
            )

            if not result.allowed:
                telemetry.increment(
                    "rate_limit.denied",
                    1,
                )

        except Exception:
            logger.debug(
                "rate_limit_telemetry_failed",
                exc_info=True,
            )

        if not result.allowed:
            duration_ms = (
                time.perf_counter()
                - started_at
            ) * 1000

            logger.warning(
                "rate_limit_exceeded",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "rule": result.rule_name,
                    "identity_type": (
                        result.identity_type.value
                        if result.identity_type
                        else None
                    ),
                    "duration_ms": round(
                        duration_ms,
                        2,
                    ),
                },
            )

            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "success": False,
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": (
                            "Too many requests. "
                            "Please try again later."
                        ),
                    },
                },
                headers=result.headers(),
            )

            return response

        response = await call_next(
            request
        )

        for name, value in result.headers().items():
            response.headers.setdefault(
                name,
                value,
            )

        return response


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================


_default_rate_limiter: Optional[
    RateLimiter
] = None


def get_rate_limiter(
    config: Optional[
        RateLimiterConfig
    ] = None,
) -> RateLimiter:
    """
    Return the process-level rate limiter.
    """

    global _default_rate_limiter

    if _default_rate_limiter is None:
        _default_rate_limiter = RateLimiter(
            config=config
        )

    return _default_rate_limiter


async def initialize_rate_limiter() -> None:
    """
    Initialize the global rate limiter.
    """

    limiter = get_rate_limiter()

    await limiter.startup()


async def shutdown_rate_limiter() -> None:
    """
    Shut down the global rate limiter.
    """

    global _default_rate_limiter

    if _default_rate_limiter is None:
        return

    await _default_rate_limiter.shutdown()

    _default_rate_limiter = None


# ============================================================================
# FASTAPI HELPER
# ============================================================================


def add_rate_limit_middleware(
    app: Any,
    *,
    config: Optional[
        RateLimiterConfig
    ] = None,
) -> Any:
    """
    Add rate limiting to a FastAPI application.

    Example
    -------
        add_rate_limit_middleware(
            app,
            config=RateLimiterConfig(
                redis_url=settings.REDIS_URL,
            ),
        )
    """

    limiter = get_rate_limiter(
        config=config
    )

    app.add_middleware(
        RateLimitMiddleware,
        rate_limiter=limiter,
    )

    return app


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "RateLimitAlgorithm",
    "RateLimitIdentity",
    "RateLimitDecision",
    "RateLimiterError",
    "RateLimiterConfigurationError",
    "RateLimiterBackendError",
    "RateLimitRule",
    "RateLimitResult",
    "RateLimiterConfig",
    "RedisRateLimitBackend",
    "RateLimiter",
    "RateLimitMiddleware",
    "get_rate_limiter",
    "initialize_rate_limiter",
    "shutdown_rate_limiter",
    "add_rate_limit_middleware",
]