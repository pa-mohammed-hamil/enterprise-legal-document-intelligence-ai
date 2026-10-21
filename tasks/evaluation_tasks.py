"""
Background tasks for model, RAG, agent, citation, hallucination,
security, and regression evaluation.

Designed for:
    - Celery workers
    - local synchronous execution
    - CI/CD evaluation gates
    - scheduled benchmark runs
    - model promotion workflows
    - regression detection

The module is intentionally defensive and integrates with the existing
evaluation stack when available without making hard assumptions about
the exact implementation of individual evaluators.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(
        PROJECT_ROOT / ".env",
        override=False,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(
    "legal-evaluation-tasks"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EVALUATION_ROOT = Path(
    os.getenv(
        "EVALUATION_OUTPUT_DIR",
        str(
            PROJECT_ROOT
            / "data"
            / "evaluation"
            / "runs"
        ),
    )
)

DEFAULT_DATASET = Path(
    os.getenv(
        "EVALUATION_DATASET",
        str(
            PROJECT_ROOT
            / "data"
            / "evaluation"
            / "evaluation.jsonl"
        ),
    )
)

DEFAULT_BASELINE = Path(
    os.getenv(
        "EVALUATION_BASELINE",
        str(
            PROJECT_ROOT
            / "data"
            / "evaluation"
            / "model_baseline.json"
        ),
    )
)

EVAL_TOP_K = int(
    os.getenv(
        "EVALUATION_TOP_K",
        "5",
    )
)

MIN_ACCURACY = float(
    os.getenv(
        "EVAL_MIN_ACCURACY",
        "0.80",
    )
)

MIN_F1 = float(
    os.getenv(
        "EVAL_MIN_F1",
        "0.75",
    )
)

MIN_RECALL = float(
    os.getenv(
        "EVAL_MIN_RECALL",
        "0.75",
    )
)

MAX_LATENCY_MS = float(
    os.getenv(
        "EVAL_MAX_LATENCY_MS",
        "5000",
    )
)

MAX_REGRESSION = float(
    os.getenv(
        "EVAL_MAX_REGRESSION",
        "0.05",
    )
)

TASK_MAX_RETRIES = int(
    os.getenv(
        "EVALUATION_TASK_MAX_RETRIES",
        "3",
    )
)

TASK_RETRY_BACKOFF = int(
    os.getenv(
        "EVALUATION_TASK_RETRY_BACKOFF_SECONDS",
        "15",
    )
)

# ---------------------------------------------------------------------------
# Optional Celery
# ---------------------------------------------------------------------------

try:
    from celery import Task

    CELERY_AVAILABLE = True
except ImportError:
    Task = object  # type: ignore
    CELERY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluationTaskResult:
    task: str
    run_id: str
    status: str
    started_at: str
    completed_at: str
    duration_ms: float
    output_path: str | None = None
    model: str | None = None
    dataset: str | None = None
    metrics: dict[str, Any] | None = None
    quality_gate_passed: bool | None = None
    regression_detected: bool | None = None
    message: str | None = None
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_iso() -> str:
    return utc_now().isoformat()


def stable_id(
    *parts: Any,
) -> str:
    value = ":".join(
        str(part)
        for part in parts
    )

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"legal-ai-evaluation:{value}",
        )
    )


def sha256_text(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def ensure_directory(
    path: Path,
) -> Path:
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def safe_name(
    value: str,
) -> str:
    return "".join(
        char
        if char.isalnum()
        or char in "-_."
        else "_"
        for char in str(value)
    )[:200] or "evaluation"


def write_json(
    path: Path,
    payload: Any,
) -> None:
    ensure_directory(
        path.parent
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def read_json(
    path: Path,
) -> Any | None:
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        logger.warning(
            "Unable to read JSON file: %s",
            path,
            exc_info=True,
        )
        return None


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    rows: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}: "
                    f"{exc}"
                ) from exc

            if isinstance(
                value,
                dict,
            ):
                rows.append(value)

    return rows


# ---------------------------------------------------------------------------
# Dynamic component resolution
# ---------------------------------------------------------------------------


def resolve_callable(
    candidates: Iterable[
        tuple[str, str]
    ],
) -> Callable[..., Any] | None:
    for module_name, attribute_name in candidates:
        try:
            module = importlib.import_module(
                module_name
            )

            function = getattr(
                module,
                attribute_name,
                None,
            )

            if callable(function):
                return function

        except ImportError:
            continue

        except Exception:
            logger.debug(
                "Could not resolve %s.%s",
                module_name,
                attribute_name,
                exc_info=True,
            )

    return None


def resolve_evaluator(
    evaluator_type: str,
) -> Callable[..., Any] | None:
    """
    Locate an evaluator from the project's evaluation layer.

    Supports both the recommended:
        ml.evaluation.*

    and older:
        scripts.evaluate_models

    layouts.
    """

    candidates: dict[
        str,
        tuple[tuple[str, str], ...],
    ] = {
        "model": (
            (
                "ml.evaluation.benchmark_runner",
                "evaluate_model",
            ),
            (
                "ml.evaluation.llm_evaluation",
                "evaluate_model",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_model",
            ),
        ),
        "classification": (
            (
                "ml.evaluation.benchmark_runner",
                "evaluate_classification",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_classification",
            ),
        ),
        "regression": (
            (
                "ml.evaluation.benchmark_runner",
                "evaluate_regression",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_regression",
            ),
        ),
        "retrieval": (
            (
                "ml.evaluation.retrieval_evaluation",
                "evaluate_retrieval",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_retrieval",
            ),
        ),
        "rag": (
            (
                "ml.evaluation.llm_evaluation",
                "evaluate_rag",
            ),
            (
                "ml.evaluation.benchmark_runner",
                "evaluate_rag",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_rag",
            ),
        ),
        "citation": (
            (
                "ml.evaluation.citation_evaluation",
                "evaluate_citations",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_citations",
            ),
        ),
        "hallucination": (
            (
                "ml.evaluation.hallucination_evaluation",
                "evaluate_hallucination",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_hallucination",
            ),
        ),
        "agent": (
            (
                "ml.evaluation.agent_evaluation",
                "evaluate_agents",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_agents",
            ),
        ),
        "safety": (
            (
                "ml.evaluation.safety_evaluation",
                "evaluate_safety",
            ),
            (
                "scripts.evaluate_models",
                "evaluate_safety",
            ),
        ),
        "regression_test": (
            (
                "ml.evaluation.regression_evaluation",
                "evaluate_regression",
            ),
            (
                "scripts.evaluate_models",
                "compare_with_baseline",
            ),
        ),
    }

    return resolve_callable(
        candidates.get(
            evaluator_type,
            (),
        )
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def flatten_metrics(
    value: Any,
    prefix: str = "",
) -> dict[str, float]:
    """
    Extract numeric metrics from nested evaluator output.
    """

    output: dict[str, float] = {}

    if isinstance(
        value,
        dict,
    ):
        for key, item in value.items():
            child_prefix = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )

            output.update(
                flatten_metrics(
                    item,
                    child_prefix,
                )
            )

    elif isinstance(
        value,
        (int, float),
    ) and not isinstance(
        value,
        bool,
    ):
        output[prefix] = float(value)

    return output


def extract_metric(
    payload: Any,
    names: tuple[str, ...],
) -> float | None:
    if not isinstance(
        payload,
        dict,
    ):
        return None

    metrics = flatten_metrics(
        payload
    )

    for wanted in names:
        wanted_lower = wanted.lower()

        for key, value in metrics.items():
            if key.lower().endswith(
                wanted_lower
            ):
                return value

    return None


def quality_gate(
    metrics: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Apply deterministic quality gates.

    Missing metrics are not treated as failures.
    """

    failures: list[str] = []

    accuracy = extract_metric(
        metrics,
        (
            "accuracy",
            "metrics.accuracy",
        ),
    )

    if (
        accuracy is not None
        and accuracy < MIN_ACCURACY
    ):
        failures.append(
            f"accuracy={accuracy:.4f} "
            f"< {MIN_ACCURACY:.4f}"
        )

    f1 = extract_metric(
        metrics,
        (
            "f1",
            "f1_macro",
            "metrics.f1_macro",
        ),
    )

    if (
        f1 is not None
        and f1 < MIN_F1
    ):
        failures.append(
            f"f1={f1:.4f} "
            f"< {MIN_F1:.4f}"
        )

    recall = extract_metric(
        metrics,
        (
            "recall",
            "recall_macro",
            "recall_at_k",
        ),
    )

    if (
        recall is not None
        and recall < MIN_RECALL
    ):
        failures.append(
            f"recall={recall:.4f} "
            f"< {MIN_RECALL:.4f}"
        )

    latency = extract_metric(
        metrics,
        (
            "p95_latency_ms",
            "latency_p95_ms",
            "p95_ms",
        ),
    )

    if (
        latency is not None
        and latency > MAX_LATENCY_MS
    ):
        failures.append(
            f"p95_latency_ms={latency:.2f} "
            f"> {MAX_LATENCY_MS:.2f}"
        )

    return (
        len(failures) == 0,
        failures,
    )


# ---------------------------------------------------------------------------
# Baseline regression
# ---------------------------------------------------------------------------


def compare_metrics_to_baseline(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare current numeric metrics against a baseline.

    Higher-is-better metrics:
        accuracy
        precision
        recall
        f1
        mrr
        groundedness
        citation_recall
        relevance

    Lower-is-better metrics:
        latency
        cost
        mae
        rmse
        error
        hallucination_rate
        unsafe_output_rate
        pii_leak_rate
        prompt_injection_rate
    """

    current = flatten_metrics(
        metrics
    )

    previous = flatten_metrics(
        baseline
    )

    higher_is_better = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "mrr",
        "grounded",
        "citation",
        "relevance",
        "faithfulness",
        "quality",
    )

    lower_is_better = (
        "latency",
        "cost",
        "mae",
        "rmse",
        "error",
        "hallucination",
        "unsafe",
        "pii",
        "injection",
    )

    comparisons: dict[str, Any] = {}
    regressions: list[str] = []

    for metric_name, current_value in current.items():
        previous_value = None

        for baseline_name, baseline_value in previous.items():
            if (
                baseline_name == metric_name
                or baseline_name.endswith(
                    metric_name
                )
                or metric_name.endswith(
                    baseline_name
                )
            ):
                previous_value = baseline_value
                break

        if previous_value is None:
            continue

        if previous_value == 0:
            delta = (
                current_value
                - previous_value
            )
            relative_change = None
        else:
            delta = (
                current_value
                - previous_value
            )
            relative_change = (
                delta
                / abs(previous_value)
            )

        metric_lower = metric_name.lower()

        is_higher_better = any(
            token in metric_lower
            for token in higher_is_better
        )

        is_lower_better = any(
            token in metric_lower
            for token in lower_is_better
        )

        regression = False

        if relative_change is not None:
            if is_higher_better:
                regression = (
                    relative_change
                    < -MAX_REGRESSION
                )

            elif is_lower_better:
                regression = (
                    relative_change
                    > MAX_REGRESSION
                )

        comparisons[metric_name] = {
            "current": current_value,
            "baseline": previous_value,
            "delta": delta,
            "relative_change": relative_change,
            "regression": regression,
        }

        if regression:
            regressions.append(
                metric_name
            )

    return {
        "comparisons": comparisons,
        "regression_detected": bool(
            regressions
        ),
        "regressed_metrics": regressions,
        "max_allowed_regression": MAX_REGRESSION,
    }


# ---------------------------------------------------------------------------
# Generic evaluator invocation
# ---------------------------------------------------------------------------


def invoke_evaluator(
    evaluator: Callable[..., Any],
    *,
    dataset: list[dict[str, Any]],
    model: str | None = None,
    top_k: int = EVAL_TOP_K,
    **kwargs: Any,
) -> Any:
    """
    Invoke evaluator implementations with progressively smaller
    argument sets to remain compatible with different evaluator APIs.
    """

    attempts = (
        {
            "dataset": dataset,
            "model": model,
            "top_k": top_k,
            **kwargs,
        },
        {
            "records": dataset,
            "model": model,
            "top_k": top_k,
            **kwargs,
        },
        {
            "dataset": dataset,
            "model": model,
            **kwargs,
        },
        {
            "records": dataset,
            "model": model,
            **kwargs,
        },
        {
            "dataset": dataset,
        },
        {
            "records": dataset,
        },
    )

    last_error: Exception | None = None

    for parameters in attempts:
        cleaned = {
            key: value
            for key, value in parameters.items()
            if value is not None
        }

        try:
            return evaluator(
                **cleaned
            )

        except TypeError as exc:
            last_error = exc

    if last_error:
        raise last_error

    raise RuntimeError(
        "Unable to invoke evaluator."
    )


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------


def create_run_id(
    task_name: str,
    model: str | None,
    dataset: Path,
) -> str:
    timestamp = (
        utc_now()
        .strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    suffix = stable_id(
        task_name,
        model or "default",
        str(dataset),
        timestamp,
    )[:12]

    return (
        f"{timestamp}-"
        f"{safe_name(task_name)}-"
        f"{suffix}"
    )


def run_directory(
    run_id: str,
) -> Path:
    return ensure_directory(
        EVALUATION_ROOT
        / safe_name(run_id)
    )


def persist_evaluation_result(
    result: dict[str, Any],
) -> Path:
    run_id = result["run_id"]

    directory = run_directory(
        run_id
    )

    output_path = (
        directory
        / "evaluation_result.json"
    )

    write_json(
        output_path,
        result,
    )

    return output_path


def make_task_result(
    *,
    task_name: str,
    run_id: str,
    started: float,
    status: str,
    model: str | None = None,
    dataset: str | None = None,
    output_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    quality_gate_passed: bool | None = None,
    regression_detected: bool | None = None,
    message: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    result = EvaluationTaskResult(
        task=task_name,
        run_id=run_id,
        status=status,
        started_at=utc_iso(),
        completed_at=utc_iso(),
        duration_ms=round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        ),
        output_path=output_path,
        model=model,
        dataset=dataset,
        metrics=metrics,
        quality_gate_passed=quality_gate_passed,
        regression_detected=regression_detected,
        message=message,
        error_type=error_type,
    )

    return result.to_dict()


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    *,
    task_name: str,
    evaluator_type: str,
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
    top_k: int = EVAL_TOP_K,
    baseline_path: str | Path | None = DEFAULT_BASELINE,
    evaluator_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()

    dataset = Path(
        dataset_path
    ).expanduser()

    if not dataset.is_absolute():
        dataset = (
            PROJECT_ROOT
            / dataset
        )

    if run_id is None:
        run_id = create_run_id(
            task_name,
            model,
            dataset,
        )

    try:
        records = read_jsonl(
            dataset
        )

        if not records:
            raise ValueError(
                f"Evaluation dataset is empty: {dataset}"
            )

        evaluator = resolve_evaluator(
            evaluator_type
        )

        if evaluator is None:
            raise RuntimeError(
                f"No evaluator found for "
                f"type '{evaluator_type}'."
            )

        evaluator_result = invoke_evaluator(
            evaluator,
            dataset=records,
            model=model,
            top_k=top_k,
            **(
                evaluator_kwargs
                or {}
            ),
        )

        serialized = (
            evaluator_result
            if isinstance(
                evaluator_result,
                dict,
            )
            else {
                "result": evaluator_result
            }
        )

        metrics = (
            serialized.get(
                "metrics",
                serialized,
            )
        )

        if not isinstance(
            metrics,
            dict,
        ):
            metrics = {}

        gate_passed, gate_failures = (
            quality_gate(
                metrics
            )
        )

        regression = None
        regression_detected = False

        if baseline_path:
            baseline = Path(
                baseline_path
            ).expanduser()

            if not baseline.is_absolute():
                baseline = (
                    PROJECT_ROOT
                    / baseline
                )

            baseline_payload = read_json(
                baseline
            )

            if isinstance(
                baseline_payload,
                dict,
            ):
                regression = (
                    compare_metrics_to_baseline(
                        metrics,
                        baseline_payload,
                    )
                )

                regression_detected = bool(
                    regression.get(
                        "regression_detected",
                        False,
                    )
                )

        result = {
            "task": task_name,
            "run_id": run_id,
            "status": (
                "completed"
                if gate_passed
                and not regression_detected
                else "quality_gate_failed"
                if not gate_passed
                else "regression_detected"
            ),
            "created_at": utc_iso(),
            "model": model,
            "dataset": str(dataset),
            "dataset_sha256": sha256_text(
                dataset.read_text(
                    encoding="utf-8"
                )
            ),
            "dataset_records": len(
                records
            ),
            "evaluator_type": evaluator_type,
            "top_k": top_k,
            "metrics": metrics,
            "evaluator_result": serialized,
            "quality_gate": {
                "passed": gate_passed,
                "failures": gate_failures,
            },
            "regression": regression,
        }

        output_path = persist_evaluation_result(
            {
                **result,
                "duration_ms": round(
                    (
                        time.perf_counter()
                        - started
                    )
                    * 1000,
                    2,
                ),
            }
        )

        return make_task_result(
            task_name=task_name,
            run_id=run_id,
            started=started,
            status=result["status"],
            model=model,
            dataset=str(dataset),
            output_path=str(
                output_path
            ),
            metrics=metrics,
            quality_gate_passed=gate_passed,
            regression_detected=regression_detected,
            message=(
                "Evaluation completed."
                if result["status"]
                == "completed"
                else (
                    "Evaluation completed "
                    "with quality-gate or "
                    "regression failures."
                )
            ),
        )

    except Exception as exc:
        logger.exception(
            "Evaluation task failed: %s",
            task_name,
        )

        return make_task_result(
            task_name=task_name,
            run_id=run_id,
            started=started,
            status="failed",
            model=model,
            dataset=str(dataset),
            message=str(exc)[:4000],
            error_type=type(
                exc
            ).__name__,
        )


# ---------------------------------------------------------------------------
# Public evaluation tasks
# ---------------------------------------------------------------------------


def evaluate_model(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
    top_k: int = EVAL_TOP_K,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_model",
        evaluator_type="model",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
        top_k=top_k,
    )


def evaluate_classification(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_classification",
        evaluator_type="classification",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
    )


def evaluate_regression(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_regression",
        evaluator_type="regression",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
    )


def evaluate_retrieval(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
    top_k: int = EVAL_TOP_K,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_retrieval",
        evaluator_type="retrieval",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
        top_k=top_k,
    )


def evaluate_rag(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
    top_k: int = EVAL_TOP_K,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_rag",
        evaluator_type="rag",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
        top_k=top_k,
    )


def evaluate_citations(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_citations",
        evaluator_type="citation",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
    )


def evaluate_hallucinations(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_hallucinations",
        evaluator_type="hallucination",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
    )


def evaluate_agents(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_agents",
        evaluator_type="agent",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
    )


def evaluate_safety(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_safety",
        evaluator_type="safety",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
    )


def evaluate_regression_against_baseline(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
    baseline_path: str | Path = DEFAULT_BASELINE,
) -> dict[str, Any]:
    return run_evaluation(
        task_name="evaluate_regression_against_baseline",
        evaluator_type="regression_test",
        dataset_path=dataset_path,
        model=model,
        run_id=run_id,
        baseline_path=baseline_path,
    )


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------


def run_full_evaluation_suite(
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
    top_k: int = EVAL_TOP_K,
) -> dict[str, Any]:
    """
    Run the complete legal-AI evaluation suite.

    The suite evaluates:
        1. classification
        2. regression
        3. retrieval
        4. RAG
        5. citations
        6. hallucination/groundedness
        7. agents
        8. security/safety
        9. baseline regression
    """

    started = time.perf_counter()

    if run_id is None:
        run_id = create_run_id(
            "full_evaluation_suite",
            model,
            Path(dataset_path),
        )

    evaluations: dict[
        str,
        dict[str, Any],
    ] = {}

    suite = (
        (
            "classification",
            evaluate_classification,
        ),
        (
            "regression",
            evaluate_regression,
        ),
        (
            "retrieval",
            lambda **kwargs: evaluate_retrieval(
                top_k=top_k,
                **kwargs,
            ),
        ),
        (
            "rag",
            lambda **kwargs: evaluate_rag(
                top_k=top_k,
                **kwargs,
            ),
        ),
        (
            "citations",
            evaluate_citations,
        ),
        (
            "hallucinations",
            evaluate_hallucinations,
        ),
        (
            "agents",
            evaluate_agents,
        ),
        (
            "safety",
            evaluate_safety,
        ),
    )

    for name, evaluator in suite:
        try:
            result = evaluator(
                dataset_path=dataset_path,
                model=model,
                run_id=(
                    f"{run_id}-{name}"
                ),
            )

            evaluations[name] = result

        except Exception as exc:
            logger.exception(
                "Evaluation suite stage failed: %s",
                name,
            )

            evaluations[name] = {
                "task": name,
                "status": "failed",
                "message": str(exc),
                "error_type": type(
                    exc
                ).__name__,
            }

    failed = [
        name
        for name, result in evaluations.items()
        if result.get("status")
        == "failed"
    ]

    quality_failures = [
        name
        for name, result in evaluations.items()
        if result.get(
            "quality_gate_passed"
        )
        is False
    ]

    regressions = [
        name
        for name, result in evaluations.items()
        if result.get(
            "regression_detected"
        )
    ]

    status = (
        "failed"
        if failed
        else "quality_gate_failed"
        if quality_failures
        else "regression_detected"
        if regressions
        else "completed"
    )

    suite_result = {
        "task": "run_full_evaluation_suite",
        "run_id": run_id,
        "status": status,
        "model": model,
        "dataset": str(
            dataset_path
        ),
        "created_at": utc_iso(),
        "duration_ms": round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            2,
        ),
        "evaluations": evaluations,
        "summary": {
            "total": len(
                evaluations
            ),
            "failed": failed,
            "quality_gate_failures": (
                quality_failures
            ),
            "regressions": regressions,
        },
    }

    output_path = (
        run_directory(run_id)
        / "full_evaluation_suite.json"
    )

    write_json(
        output_path,
        suite_result,
    )

    suite_result[
        "output_path"
    ] = str(output_path)

    return suite_result


# ---------------------------------------------------------------------------
# Celery base task
# ---------------------------------------------------------------------------


class EvaluationTaskBase(
    Task
):
    """
    Shared Celery configuration for evaluation jobs.
    """

    max_retries = TASK_MAX_RETRIES

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        logger.error(
            "Evaluation task failed: "
            "task_id=%s error=%s",
            task_id,
            exc,
        )


# ---------------------------------------------------------------------------
# Celery app discovery
# ---------------------------------------------------------------------------


def get_celery_app() -> Any | None:
    candidates = (
        "backend.workers.celery_app",
        "backend.tasks.celery_app",
        "backend.celery_app",
    )

    for module_name in candidates:
        try:
            module = importlib.import_module(
                module_name
            )

            for attribute in (
                "celery_app",
                "app",
                "celery",
            ):
                application = getattr(
                    module,
                    attribute,
                    None,
                )

                if application is not None:
                    return application

        except ImportError:
            continue

        except Exception:
            logger.debug(
                "Unable to load Celery app %s.",
                module_name,
                exc_info=True,
            )

    return None


celery_app = get_celery_app()


# ---------------------------------------------------------------------------
# Celery wrappers
# ---------------------------------------------------------------------------


def _celery_full_suite(
    self: Any,
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
    top_k: int = EVAL_TOP_K,
) -> dict[str, Any]:
    try:
        result = run_full_evaluation_suite(
            dataset_path=dataset_path,
            model=model,
            run_id=run_id,
            top_k=top_k,
        )

        if result.get(
            "status"
        ) == "failed":
            raise RuntimeError(
                "Full evaluation suite failed."
            )

        return result

    except Exception as exc:
        countdown = (
            TASK_RETRY_BACKOFF
            * (
                2
                ** self.request.retries
            )
        )

        raise self.retry(
            exc=exc,
            countdown=countdown,
        )


def _celery_model_eval(
    self: Any,
    dataset_path: str | Path = DEFAULT_DATASET,
    model: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    try:
        result = evaluate_model(
            dataset_path=dataset_path,
            model=model,
            run_id=run_id,
        )

        if result.get(
            "status"
        ) == "failed":
            raise RuntimeError(
                result.get(
                    "message",
                    "Model evaluation failed.",
                )
            )

        return result

    except Exception as exc:
        countdown = (
            TASK_RETRY_BACKOFF
            * (
                2
                ** self.request.retries
            )
        )

        raise self.retry(
            exc=exc,
            countdown=countdown,
        )


if celery_app is not None:

    @celery_app.task(
        bind=True,
        base=EvaluationTaskBase,
        name="legal.evaluation.full_suite",
        max_retries=TASK_MAX_RETRIES,
    )
    def full_evaluation_suite_task(
        self: Any,
        dataset_path: str | Path = DEFAULT_DATASET,
        model: str | None = None,
        run_id: str | None = None,
        top_k: int = EVAL_TOP_K,
    ) -> dict[str, Any]:
        return _celery_full_suite(
            self,
            dataset_path,
            model,
            run_id,
            top_k,
        )

    @celery_app.task(
        bind=True,
        base=EvaluationTaskBase,
        name="legal.evaluation.model",
        max_retries=TASK_MAX_RETRIES,
    )
    def model_evaluation_task(
        self: Any,
        dataset_path: str | Path = DEFAULT_DATASET,
        model: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return _celery_model_eval(
            self,
            dataset_path,
            model,
            run_id,
        )

else:

    def full_evaluation_suite_task(
        dataset_path: str | Path = DEFAULT_DATASET,
        model: str | None = None,
        run_id: str | None = None,
        top_k: int = EVAL_TOP_K,
    ) -> dict[str, Any]:
        return run_full_evaluation_suite(
            dataset_path=dataset_path,
            model=model,
            run_id=run_id,
            top_k=top_k,
        )

    def model_evaluation_task(
        dataset_path: str | Path = DEFAULT_DATASET,
        model: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        return evaluate_model(
            dataset_path=dataset_path,
            model=model,
            run_id=run_id,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "EvaluationTaskResult",
    "evaluate_model",
    "evaluate_classification",
    "evaluate_regression",
    "evaluate_retrieval",
    "evaluate_rag",
    "evaluate_citations",
    "evaluate_hallucinations",
    "evaluate_agents",
    "evaluate_safety",
    "evaluate_regression_against_baseline",
    "run_evaluation",
    "run_full_evaluation_suite",
    "full_evaluation_suite_task",
    "model_evaluation_task",
]