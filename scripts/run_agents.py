"""
Run and orchestrate legal AI agents.

Supports:
    - Supervisor agent
    - Document analysis
    - Clause analysis
    - Legal Q&A
    - Risk analysis
    - Expiry/deadline analysis
    - Renewal analysis
    - Obligation extraction
    - Contract comparison
    - Human review routing

Design goals:
    - Deterministic CLI behavior
    - Async-compatible execution
    - Safe agent loading
    - Structured JSON output
    - Run tracing and timing
    - Optional persistence
    - Graceful degradation when optional agent
      implementations are unavailable
    - Production-oriented error handling

Usage:
    python scripts/run_agents.py \
        --agent supervisor \
        --input data/sample/contracts/sample.pdf

    python scripts/run_agents.py \
        --agent legal_qa \
        --input data/sample/contracts/sample.json \
        --query "What are the termination rights?"

    python scripts/run_agents.py \
        --agent risk \
        --input data/sample/contracts/sample.json

    python scripts/run_agents.py \
        --agent all \
        --input data/sample/contracts/sample.json

    python scripts/run_agents.py \
        --agent supervisor \
        --input data/sample/contracts/sample.json \
        --output data/evaluation/agent_run.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    ),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "legal-agent-runner"
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "agent_runs"
    / "latest.json"
)

DEFAULT_TENANT_ID = os.getenv(
    "DEFAULT_TENANT_ID",
    "default-tenant",
)

DEFAULT_ORGANIZATION_ID = os.getenv(
    "DEFAULT_ORGANIZATION_ID",
    "default-organization",
)

DEFAULT_USER_ID = os.getenv(
    "DEFAULT_USER_ID",
    "system-agent-runner",
)

# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_IMPORTS: dict[str, tuple[str, ...]] = {
    "supervisor": (
        "backend.agents.supervisor",
        "backend.agents.supervisor_agent",
    ),
    "document": (
        "backend.agents.document",
        "backend.agents.document_agent",
    ),
    "clause": (
        "backend.agents.clause",
        "backend.agents.clause_agent",
    ),
    "legal_qa": (
        "backend.agents.legal_qa",
        "backend.agents.legal_qa_agent",
    ),
    "risk": (
        "backend.agents.risk",
        "backend.agents.risk_agent",
    ),
    "expiry": (
        "backend.agents.expiry",
        "backend.agents.expiry_agent",
    ),
    "renewal": (
        "backend.agents.renewal",
        "backend.agents.renewal_agent",
    ),
    "obligation": (
        "backend.agents.obligation",
        "backend.agents.obligation_agent",
    ),
    "comparison": (
        "backend.agents.comparison",
        "backend.agents.comparison_agent",
    ),
    "review": (
        "backend.agents.review",
        "backend.agents.review_agent",
    ),
}

AGENT_ALIASES = {
    "qa": "legal_qa",
    "legal-q-a": "legal_qa",
    "legalqa": "legal_qa",
    "doc": "document",
    "contracts": "document",
    "risk-analysis": "risk",
    "deadline": "expiry",
    "deadlines": "expiry",
    "obligations": "obligation",
    "compare": "comparison",
    "human-review": "review",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentContext:
    """Execution context shared with agents."""

    run_id: str
    tenant_id: str
    organization_id: str
    user_id: str

    input_path: str | None = None
    query: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class AgentResult:
    """Normalized result returned by an agent."""

    agent: str
    run_id: str

    status: str

    started_at: str
    completed_at: str

    latency_ms: float

    result: Any = None
    error: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class AgentRun:
    """Complete multi-agent execution record."""

    run_id: str

    status: str

    started_at: str
    completed_at: str

    total_latency_ms: float

    requested_agent: str

    input_path: str | None

    query: str | None

    results: list[AgentResult] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def stable_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temporary,
        path,
    )


def load_input(
    input_path: Path | None,
) -> Any:
    """
    Load JSON/JSONL/text input.

    PDFs and DOCX files are passed as metadata for
    the actual ingestion/document-processing layer.
    """

    if input_path is None:
        return None

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input does not exist: {input_path}"
        )

    suffix = input_path.suffix.lower()

    if suffix == ".json":
        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    if suffix == ".jsonl":
        records: list[Any] = []

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                records.append(
                    json.loads(line)
                )

        return records

    if suffix in {
        ".txt",
        ".md",
        ".html",
        ".htm",
        ".rtf",
    }:
        return input_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    return {
        "file_path": str(
            input_path.resolve()
        ),
        "filename": input_path.name,
        "extension": suffix,
        "mime_type": (
            "application/pdf"
            if suffix == ".pdf"
            else None
        ),
    }


def normalize_agent_name(
    name: str,
) -> str:
    normalized = (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return AGENT_ALIASES.get(
        normalized,
        normalized,
    )


# ---------------------------------------------------------------------------
# Dynamic agent loading
# ---------------------------------------------------------------------------


def import_agent_module(
    agent_name: str,
) -> Any | None:
    """
    Try known import locations without making the
    runner dependent on one exact implementation.
    """

    candidates = AGENT_IMPORTS.get(
        agent_name,
        (),
    )

    for module_name in candidates:
        try:
            return importlib.import_module(
                module_name
            )
        except ImportError as exc:
            logger.debug(
                "Could not import %s: %s",
                module_name,
                exc,
            )

    return None


def find_callable(
    module: Any,
    agent_name: str,
) -> Callable[..., Any] | None:
    """
    Find the most likely callable implementation.
    """

    if module is None:
        return None

    candidates = [
        "run",
        "execute",
        "run_agent",
        "execute_agent",
        "invoke",
        "process",
        f"run_{agent_name}",
        f"execute_{agent_name}",
    ]

    for name in candidates:
        candidate = getattr(
            module,
            name,
            None,
        )

        if callable(candidate):
            return candidate

    # Search for conventional Agent classes.
    class_candidates = [
        "Agent",
        "BaseAgent",
        "SupervisorAgent",
        f"{agent_name.title().replace('_', '')}Agent",
    ]

    for class_name in class_candidates:
        cls = getattr(
            module,
            class_name,
            None,
        )

        if cls is None or not inspect.isclass(cls):
            continue

        try:
            instance = cls()
        except Exception:
            logger.debug(
                "Unable to instantiate %s",
                class_name,
                exc_info=True,
            )
            continue

        for method_name in (
            "run",
            "execute",
            "invoke",
            "process",
        ):
            method = getattr(
                instance,
                method_name,
                None,
            )

            if callable(method):
                return method

    return None


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------


async def invoke_callable(
    callable_obj: Callable[..., Any],
    context: AgentContext,
    payload: Any,
) -> Any:
    """
    Invoke a sync or async agent while supporting common
    argument conventions.
    """

    signature = inspect.signature(
        callable_obj
    )

    parameters = signature.parameters

    kwargs: dict[str, Any] = {}

    if "context" in parameters:
        kwargs["context"] = context

    if "payload" in parameters:
        kwargs["payload"] = payload

    elif "input_data" in parameters:
        kwargs["input_data"] = payload

    elif "data" in parameters:
        kwargs["data"] = payload

    elif "document" in parameters:
        kwargs["document"] = payload

    elif "input" in parameters:
        kwargs["input"] = payload

    if "query" in parameters:
        kwargs["query"] = context.query

    if "run_id" in parameters:
        kwargs["run_id"] = context.run_id

    if "tenant_id" in parameters:
        kwargs["tenant_id"] = (
            context.tenant_id
        )

    if "organization_id" in parameters:
        kwargs["organization_id"] = (
            context.organization_id
        )

    if kwargs:
        result = callable_obj(
            **kwargs
        )
    else:
        # Last-resort conventional single argument.
        result = callable_obj(
            payload
        )

    if inspect.isawaitable(result):
        return await result

    return result


def fallback_result(
    agent_name: str,
    context: AgentContext,
    payload: Any,
) -> dict[str, Any]:
    """
    Safe fallback when the actual agent module is not
    installed or is intentionally being developed.

    This does not pretend that an AI analysis occurred.
    """

    return {
        "status": "not_implemented",
        "agent": agent_name,
        "message": (
            "Agent implementation was not found. "
            "The runner completed safely without "
            "fabricating an analysis."
        ),
        "input": {
            "path": context.input_path,
            "query": context.query,
        },
        "payload_type": type(payload).__name__,
    }


# ---------------------------------------------------------------------------
# Single agent execution
# ---------------------------------------------------------------------------


async def run_agent(
    agent_name: str,
    context: AgentContext,
    payload: Any,
    allow_fallback: bool = True,
) -> AgentResult:
    started_at = utc_now()
    start_time = time.perf_counter()

    logger.info(
        "Starting agent=%s run_id=%s",
        agent_name,
        context.run_id,
    )

    try:
        module = import_agent_module(
            agent_name
        )

        callable_obj = find_callable(
            module,
            agent_name,
        )

        if callable_obj is None:
            if not allow_fallback:
                raise RuntimeError(
                    "Agent implementation not found: "
                    f"{agent_name}"
                )

            result = fallback_result(
                agent_name,
                context,
                payload,
            )

            status = "not_implemented"

        else:
            result = await invoke_callable(
                callable_obj,
                context,
                payload,
            )

            status = "success"

        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        completed_at = utc_now()

        logger.info(
            "Completed agent=%s status=%s "
            "latency=%.2fms",
            agent_name,
            status,
            latency_ms,
        )

        return AgentResult(
            agent=agent_name,
            run_id=context.run_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            latency_ms=round(
                latency_ms,
                3,
            ),
            result=result,
        )

    except Exception as exc:
        latency_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.exception(
            "Agent failed: %s",
            agent_name,
        )

        return AgentResult(
            agent=agent_name,
            run_id=context.run_id,
            status="error",
            started_at=started_at,
            completed_at=utc_now(),
            latency_ms=round(
                latency_ms,
                3,
            ),
            error=str(exc),
            metadata={
                "exception_type": type(
                    exc
                ).__name__,
                "traceback": traceback.format_exc(),
            },
        )


# ---------------------------------------------------------------------------
# Agent orchestration
# ---------------------------------------------------------------------------


DEFAULT_AGENT_ORDER = (
    "document",
    "clause",
    "obligation",
    "risk",
    "expiry",
    "renewal",
    "legal_qa",
    "comparison",
    "review",
)


async def run_all_agents(
    context: AgentContext,
    payload: Any,
    allow_fallback: bool,
) -> list[AgentResult]:
    """
    Run independent analysis agents sequentially.

    Sequential execution makes the demo deterministic and
    easier to observe. The production service can replace
    this with dependency-aware parallel execution.
    """

    results: list[AgentResult] = []

    for agent_name in DEFAULT_AGENT_ORDER:
        result = await run_agent(
            agent_name=agent_name,
            context=context,
            payload=payload,
            allow_fallback=allow_fallback,
        )

        results.append(result)

    return results


async def run_supervisor(
    context: AgentContext,
    payload: Any,
    allow_fallback: bool,
) -> AgentResult:
    """
    Execute the supervisor agent.

    If a supervisor implementation exists, it controls
    orchestration. Otherwise the runner executes the known
    specialist agents and returns their results.
    """

    supervisor_module = import_agent_module(
        "supervisor"
    )

    supervisor_callable = find_callable(
        supervisor_module,
        "supervisor",
    )

    if supervisor_callable is not None:
        return await run_agent(
            agent_name="supervisor",
            context=context,
            payload=payload,
            allow_fallback=allow_fallback,
        )

    logger.warning(
        "Supervisor implementation unavailable; "
        "using deterministic specialist orchestration."
    )

    started_at = utc_now()
    start_time = time.perf_counter()

    specialist_results = (
        await run_all_agents(
            context=context,
            payload=payload,
            allow_fallback=allow_fallback,
        )
    )

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    return AgentResult(
        agent="supervisor",
        run_id=context.run_id,
        status="success",
        started_at=started_at,
        completed_at=utc_now(),
        latency_ms=round(
            latency_ms,
            3,
        ),
        result={
            "orchestration_mode": (
                "deterministic_fallback"
            ),
            "agents": [
                asdict(result)
                for result in specialist_results
            ],
        },
    )


# ---------------------------------------------------------------------------
# Run orchestration
# ---------------------------------------------------------------------------


async def execute_run(
    requested_agent: str,
    input_path: Path | None,
    query: str | None,
    tenant_id: str,
    organization_id: str,
    user_id: str,
    allow_fallback: bool,
) -> AgentRun:
    run_id = str(
        uuid.uuid4()
    )

    started_at = utc_now()
    start_time = time.perf_counter()

    context = AgentContext(
        run_id=run_id,
        tenant_id=tenant_id,
        organization_id=organization_id,
        user_id=user_id,
        input_path=(
            str(input_path.resolve())
            if input_path
            else None
        ),
        query=query,
        metadata={
            "runner": "scripts/run_agents.py",
            "schema_version": "1.0",
        },
    )

    payload = load_input(
        input_path
    )

    normalized_agent = normalize_agent_name(
        requested_agent
    )

    if normalized_agent == "all":
        results = await run_all_agents(
            context=context,
            payload=payload,
            allow_fallback=allow_fallback,
        )

    elif normalized_agent == "supervisor":
        supervisor_result = (
            await run_supervisor(
                context=context,
                payload=payload,
                allow_fallback=allow_fallback,
            )
        )

        results = [
            supervisor_result
        ]

    elif normalized_agent in AGENT_IMPORTS:
        results = [
            await run_agent(
                agent_name=normalized_agent,
                context=context,
                payload=payload,
                allow_fallback=allow_fallback,
            )
        ]

    else:
        raise ValueError(
            "Unknown agent: "
            f"{requested_agent}. "
            f"Available agents: "
            f"{', '.join(sorted(AGENT_IMPORTS))}, all"
        )

    total_latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    if any(
        result.status == "error"
        for result in results
    ):
        status = "partial_failure"

    elif all(
        result.status == "not_implemented"
        for result in results
    ):
        status = "not_implemented"

    else:
        status = "success"

    return AgentRun(
        run_id=run_id,
        status=status,
        started_at=started_at,
        completed_at=utc_now(),
        total_latency_ms=round(
            total_latency_ms,
            3,
        ),
        requested_agent=normalized_agent,
        input_path=(
            str(input_path.resolve())
            if input_path
            else None
        ),
        query=query,
        results=results,
        metadata={
            "tenant_id": tenant_id,
            "organization_id": organization_id,
            "user_id": user_id,
            "schema_version": "1.0",
        },
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def serialize_run(
    run: AgentRun,
) -> dict[str, Any]:
    return asdict(run)


def print_run_summary(
    run: AgentRun,
) -> None:
    print()
    print("=" * 72)
    print("LEGAL AI AGENT RUN")
    print("=" * 72)
    print(f"Run ID:       {run.run_id}")
    print(f"Status:       {run.status}")
    print(f"Requested:    {run.requested_agent}")
    print(
        f"Latency:      {run.total_latency_ms:.2f} ms"
    )
    print("-" * 72)

    for result in run.results:
        print(
            f"{result.agent:<18} "
            f"{result.status:<16} "
            f"{result.latency_ms:>10.2f} ms"
        )

    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run enterprise legal document "
            "AI agents."
        )
    )

    parser.add_argument(
        "--agent",
        default="supervisor",
        help=(
            "Agent to run. "
            "Use 'all' for the full specialist pipeline."
        ),
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Input document, JSON, JSONL, "
            "or text file."
        ),
    )

    parser.add_argument(
        "--query",
        default=None,
        help="Legal question for the legal-QA agent.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON output path.",
    )

    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="Tenant identifier.",
    )

    parser.add_argument(
        "--organization-id",
        default=DEFAULT_ORGANIZATION_ID,
        help="Organization identifier.",
    )

    parser.add_argument(
        "--user-id",
        default=DEFAULT_USER_ID,
        help="Executing user identifier.",
    )

    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help=(
            "Fail if an actual agent implementation "
            "cannot be loaded."
        ),
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Print the complete JSON result.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    try:
        run = asyncio.run(
            execute_run(
                requested_agent=args.agent,
                input_path=args.input,
                query=args.query,
                tenant_id=args.tenant_id,
                organization_id=(
                    args.organization_id
                ),
                user_id=args.user_id,
                allow_fallback=(
                    not args.no_fallback
                ),
            )
        )

        output_payload = serialize_run(
            run
        )

        atomic_write_json(
            args.output,
            output_payload,
        )

        print_run_summary(
            run
        )

        if args.pretty:
            print(
                json.dumps(
                    output_payload,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        logger.info(
            "Agent run written to %s",
            args.output,
        )

        if run.status == "partial_failure":
            return 2

        if run.status == "not_implemented":
            return 3

        return 0

    except KeyboardInterrupt:
        logger.warning(
            "Agent execution interrupted."
        )
        return 130

    except Exception:
        logger.exception(
            "Agent runner failed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )