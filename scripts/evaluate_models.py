"""
Evaluate legal-document AI / NLP / RAG models.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --dataset data/evaluation/evaluation.jsonl
    python scripts/evaluate_models.py --output data/evaluation/model_results.json
    python scripts/evaluate_models.py --model <model-name>

The evaluator supports:
    - Classification metrics
    - Regression metrics
    - Embedding / retrieval metrics
    - RAG retrieval evaluation
    - Citation correctness
    - Answer relevance
    - Faithfulness / groundedness
    - Hallucination detection
    - Latency and throughput
    - Model comparison
    - JSON + CSV reports
    - Regression detection against a baseline

The script is designed to be safe to run without external LLM APIs.
When an LLM evaluator is unavailable, deterministic metrics are still
calculated and unavailable metrics are explicitly marked rather than
invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

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
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
        r2_score,
    )
except ImportError:  # pragma: no cover
    accuracy_score = None
    confusion_matrix = None
    f1_score = None
    mean_absolute_error = None
    mean_squared_error = None
    precision_score = None
    recall_score = None
    r2_score = None

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
    "legal-model-evaluator"
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "model_results.json"
)

DEFAULT_CSV_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "model_results.csv"
)

DEFAULT_BASELINE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "model_baseline.json"
)

DEFAULT_TOP_K = int(
    os.getenv(
        "EVALUATION_TOP_K",
        "5",
    )
)

# Thresholds used for CI/regression checks.
DEFAULT_MIN_ACCURACY = float(
    os.getenv(
        "EVAL_MIN_ACCURACY",
        "0.80",
    )
)

DEFAULT_MIN_F1 = float(
    os.getenv(
        "EVAL_MIN_F1",
        "0.75",
    )
)

DEFAULT_MIN_RECALL = float(
    os.getenv(
        "EVAL_MIN_RECALL",
        "0.75",
    )
)

DEFAULT_MAX_LATENCY_MS = float(
    os.getenv(
        "EVAL_MAX_LATENCY_MS",
        "5000",
    )
)

DEFAULT_MAX_REGRESSION = float(
    os.getenv(
        "EVAL_MAX_REGRESSION",
        "0.05",
    )
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MetricResult:
    name: str
    value: float | None
    unit: str = ""
    higher_is_better: bool = True
    status: str = "available"
    details: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ModelEvaluation:
    model_name: str

    classification: list[MetricResult] = field(
        default_factory=list
    )

    regression: list[MetricResult] = field(
        default_factory=list
    )

    retrieval: list[MetricResult] = field(
        default_factory=list
    )

    rag: list[MetricResult] = field(
        default_factory=list
    )

    performance: list[MetricResult] = field(
        default_factory=list
    )

    safety: list[MetricResult] = field(
        default_factory=list
    )

    overall_score: float | None = None

    status: str = "not_evaluated"


@dataclass(slots=True)
class EvaluationReport:
    schema_version: str
    generated_at: str

    dataset: str
    dataset_hash: str

    sample_count: int

    evaluations: list[ModelEvaluation]

    baseline_model: str | None = None

    regressions: list[dict[str, Any]] = field(
        default_factory=list
    )

    passed: bool = True

    errors: list[str] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def safe_float(
    value: Any,
) -> float | None:
    try:
        result = float(value)

        if math.isnan(result):
            return None

        return result

    except (
        TypeError,
        ValueError,
    ):
        return None


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def mean(
    values: Sequence[float],
) -> float | None:
    if not values:
        return None

    return statistics.fmean(values)


def percentile(
    values: Sequence[float],
    percentile_value: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        (len(ordered) - 1)
        * percentile_value
        / 100.0
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower]
        * (1.0 - weight)
        + ordered[upper]
        * weight
    )


def metric(
    name: str,
    value: float | None,
    unit: str = "",
    higher_is_better: bool = True,
    status: str = "available",
    details: dict[str, Any] | None = None,
) -> MetricResult:
    return MetricResult(
        name=name,
        value=value,
        unit=unit,
        higher_is_better=higher_is_better,
        status=status,
        details=details or {},
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping invalid JSONL line %d: %s",
                    line_number,
                    exc,
                )
                continue

            if isinstance(record, dict):
                records.append(record)

    return records


def load_json(
    path: Path,
) -> list[dict[str, Any]]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return [
            item
            for item in payload
            if isinstance(item, dict)
        ]

    if isinstance(payload, dict):
        for key in (
            "samples",
            "records",
            "data",
            "evaluations",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                return [
                    item
                    for item in value
                    if isinstance(item, dict)
                ]

        return [payload]

    return []


def load_dataset(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    if path.suffix.lower() == ".jsonl":
        records = load_jsonl(path)

    elif path.suffix.lower() == ".json":
        records = load_json(path)

    else:
        raise ValueError(
            "Dataset must be .json or .jsonl"
        )

    logger.info(
        "Loaded %d evaluation samples.",
        len(records),
    )

    return records


# ---------------------------------------------------------------------------
# Generic field extraction
# ---------------------------------------------------------------------------


def first_present(
    record: dict[str, Any],
    fields: Sequence[str],
) -> Any:
    for field_name in fields:
        if field_name in record:
            value = record[field_name]

            if value is not None:
                return value

    return None


def get_predictions(
    records: Sequence[dict[str, Any]],
    model_name: str,
) -> tuple[
    list[Any],
    list[Any],
]:
    """Extract classification predictions/labels."""

    labels: list[Any] = []
    predictions: list[Any] = []

    for record in records:
        expected = first_present(
            record,
            [
                "label",
                "true_label",
                "ground_truth",
                "expected_label",
                "target",
            ],
        )

        model_predictions = record.get(
            "predictions"
        )

        prediction = None

        if isinstance(
            model_predictions,
            dict,
        ):
            prediction = model_predictions.get(
                model_name
            )

        if prediction is None:
            prediction = first_present(
                record,
                [
                    "prediction",
                    "predicted_label",
                    "model_prediction",
                ],
            )

        if (
            expected is not None
            and prediction is not None
        ):
            labels.append(expected)
            predictions.append(prediction)

    return labels, predictions


def get_regression_values(
    records: Sequence[dict[str, Any]],
    model_name: str,
) -> tuple[
    list[float],
    list[float],
]:
    actual: list[float] = []
    predicted: list[float] = []

    for record in records:
        true_value = safe_float(
            first_present(
                record,
                [
                    "target_value",
                    "true_value",
                    "ground_truth_value",
                    "actual",
                ],
            )
        )

        prediction = None

        predictions = record.get(
            "predictions"
        )

        if isinstance(
            predictions,
            dict,
        ):
            prediction = safe_float(
                predictions.get(
                    model_name
                )
            )

        if prediction is None:
            prediction = safe_float(
                first_present(
                    record,
                    [
                        "prediction_value",
                        "predicted_value",
                        "prediction",
                    ],
                )
            )

        if (
            true_value is not None
            and prediction is not None
        ):
            actual.append(true_value)
            predicted.append(prediction)

    return actual, predicted


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------


def evaluate_classification(
    labels: Sequence[Any],
    predictions: Sequence[Any],
) -> list[MetricResult]:
    if not labels or not predictions:
        return [
            metric(
                "accuracy",
                None,
                status="unavailable",
            ),
            metric(
                "precision_macro",
                None,
                status="unavailable",
            ),
            metric(
                "recall_macro",
                None,
                status="unavailable",
            ),
            metric(
                "f1_macro",
                None,
                status="unavailable",
            ),
        ]

    if accuracy_score is None:
        return [
            metric(
                "classification_metrics",
                None,
                status=(
                    "unavailable:scikit-learn"
                ),
            )
        ]

    labels_list = list(labels)
    predictions_list = list(predictions)

    accuracy = accuracy_score(
        labels_list,
        predictions_list,
    )

    precision = precision_score(
        labels_list,
        predictions_list,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        labels_list,
        predictions_list,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        labels_list,
        predictions_list,
        average="macro",
        zero_division=0,
    )

    classes = sorted(
        set(labels_list)
        | set(predictions_list),
        key=str,
    )

    matrix = confusion_matrix(
        labels_list,
        predictions_list,
        labels=classes,
    )

    return [
        metric(
            "accuracy",
            float(accuracy),
        ),
        metric(
            "precision_macro",
            float(precision),
        ),
        metric(
            "recall_macro",
            float(recall),
        ),
        metric(
            "f1_macro",
            float(f1),
        ),
        metric(
            "class_count",
            float(len(classes)),
            higher_is_better=False,
            details={
                "classes": [
                    str(item)
                    for item in classes
                ]
            },
        ),
        metric(
            "confusion_matrix",
            None,
            status="available",
            details={
                "labels": [
                    str(item)
                    for item in classes
                ],
                "matrix": matrix.tolist(),
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------


def evaluate_regression(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> list[MetricResult]:
    if not actual or not predicted:
        return [
            metric(
                "mae",
                None,
                status="unavailable",
            ),
            metric(
                "rmse",
                None,
                status="unavailable",
            ),
            metric(
                "r2",
                None,
                status="unavailable",
            ),
        ]

    if mean_absolute_error is None:
        return [
            metric(
                "regression_metrics",
                None,
                status=(
                    "unavailable:scikit-learn"
                ),
            )
        ]

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = math.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    r2 = r2_score(
        actual,
        predicted,
    )

    return [
        metric(
            "mae",
            float(mae),
            higher_is_better=False,
        ),
        metric(
            "rmse",
            float(rmse),
            higher_is_better=False,
        ),
        metric(
            "r2",
            float(r2),
        ),
    ]


# ---------------------------------------------------------------------------
# Retrieval evaluation
# ---------------------------------------------------------------------------


def normalize_ids(
    values: Any,
) -> list[str]:
    if values is None:
        return []

    if isinstance(values, str):
        return [values]

    if isinstance(values, dict):
        values = values.get(
            "ids",
            values.get(
                "documents",
                [],
            ),
        )

    if not isinstance(
        values,
        (list, tuple, set),
    ):
        return []

    result: list[str] = []

    for value in values:
        if isinstance(value, dict):
            value = (
                value.get("id")
                or value.get("chunk_id")
                or value.get("document_id")
            )

        if value is not None:
            result.append(
                str(value)
            )

    return result


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    for index, item in enumerate(
        retrieved,
        start=1,
    ):
        if item in relevant:
            return 1.0 / index

    return 0.0


def evaluate_retrieval(
    records: Sequence[dict[str, Any]],
    top_k: int,
) -> list[MetricResult]:
    recall_values: list[float] = []
    precision_values: list[float] = []
    reciprocal_ranks: list[float] = []
    hit_values: list[float] = []

    for record in records:
        retrieved = normalize_ids(
            first_present(
                record,
                [
                    "retrieved_ids",
                    "retrieved_chunks",
                    "retrieved_documents",
                    "results",
                ],
            )
        )

        relevant = set(
            normalize_ids(
                first_present(
                    record,
                    [
                        "relevant_ids",
                        "relevant_chunks",
                        "relevant_documents",
                        "gold_ids",
                    ],
                )
            )
        )

        if not retrieved or not relevant:
            continue

        retrieved = retrieved[:top_k]

        hits = len(
            set(retrieved)
            & relevant
        )

        recall_values.append(
            hits / len(relevant)
        )

        precision_values.append(
            hits / max(
                len(retrieved),
                1,
            )
        )

        reciprocal_ranks.append(
            reciprocal_rank(
                retrieved,
                relevant,
            )
        )

        hit_values.append(
            1.0 if hits > 0 else 0.0
        )

    return [
        metric(
            "recall_at_k",
            mean(recall_values),
        ),
        metric(
            "precision_at_k",
            mean(precision_values),
        ),
        metric(
            "mrr",
            mean(reciprocal_ranks),
        ),
        metric(
            "hit_rate",
            mean(hit_values),
        ),
    ]


# ---------------------------------------------------------------------------
# RAG evaluation
# ---------------------------------------------------------------------------


def tokenize(
    text: str,
) -> set[str]:
    return {
        token.lower().strip(
            ".,;:!?()[]{}\"'"
        )
        for token in text.split()
        if token.strip()
    }


def lexical_similarity(
    left: str,
    right: str,
) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)

    if not left_tokens or not right_tokens:
        return 0.0

    intersection = len(
        left_tokens & right_tokens
    )

    union = len(
        left_tokens | right_tokens
    )

    if union == 0:
        return 0.0

    return intersection / union


def citation_ids(
    record: dict[str, Any],
) -> list[str]:
    citations = first_present(
        record,
        [
            "citations",
            "citation_ids",
            "sources",
            "source_ids",
        ],
    )

    return normalize_ids(citations)


def evaluate_rag(
    records: Sequence[dict[str, Any]],
) -> list[MetricResult]:
    relevance_scores: list[float] = []
    grounded_scores: list[float] = []
    citation_scores: list[float] = []
    answer_scores: list[float] = []

    for record in records:
        answer = first_present(
            record,
            [
                "answer",
                "generated_answer",
                "prediction",
            ],
        )

        reference = first_present(
            record,
            [
                "reference_answer",
                "expected_answer",
                "ground_truth_answer",
            ],
        )

        context = first_present(
            record,
            [
                "context",
                "retrieved_context",
                "evidence",
            ],
        )

        expected_citations = normalize_ids(
            first_present(
                record,
                [
                    "required_citations",
                    "gold_citations",
                ],
            )
        )

        actual_citations = citation_ids(
            record
        )

        if (
            isinstance(answer, str)
            and isinstance(reference, str)
        ):
            answer_scores.append(
                lexical_similarity(
                    answer,
                    reference,
                )
            )

        if (
            isinstance(answer, str)
            and isinstance(context, str)
        ):
            grounded_scores.append(
                lexical_similarity(
                    answer,
                    context,
                )
            )

        if (
            expected_citations
            and actual_citations
        ):
            expected_set = set(
                expected_citations
            )

            actual_set = set(
                actual_citations
            )

            citation_scores.append(
                len(
                    expected_set
                    & actual_set
                )
                / len(expected_set)
            )

        question = first_present(
            record,
            [
                "question",
                "query",
                "prompt",
            ],
        )

        if (
            isinstance(question, str)
            and isinstance(answer, str)
        ):
            relevance_scores.append(
                lexical_similarity(
                    question,
                    answer,
                )
            )

    return [
        metric(
            "answer_reference_similarity",
            mean(answer_scores),
        ),
        metric(
            "answer_context_groundedness",
            mean(grounded_scores),
        ),
        metric(
            "citation_recall",
            mean(citation_scores),
        ),
        metric(
            "question_answer_relevance",
            mean(relevance_scores),
        ),
    ]


# ---------------------------------------------------------------------------
# Hallucination / safety evaluation
# ---------------------------------------------------------------------------


def evaluate_hallucination(
    records: Sequence[dict[str, Any]],
) -> list[MetricResult]:
    """
    Deterministic approximation.

    A production deployment should additionally connect this
    to the project's LLM-as-judge / entailment evaluator.
    """

    grounded: list[float] = []

    for record in records:
        explicit_score = safe_float(
            first_present(
                record,
                [
                    "groundedness",
                    "faithfulness",
                    "hallucination_score",
                ],
            )
        )

        if explicit_score is not None:
            # Support either groundedness (higher is better)
            # or hallucination probability (lower is better).
            if (
                "hallucination_score"
                in record
            ):
                grounded.append(
                    1.0 - clamp(
                        explicit_score
                    )
                )
            else:
                grounded.append(
                    clamp(explicit_score)
                )

    return [
        metric(
            "groundedness",
            mean(grounded),
        ),
        metric(
            "hallucination_rate",
            (
                1.0 - mean(grounded)
                if grounded
                else None
            ),
            higher_is_better=False,
        ),
    ]


def evaluate_safety(
    records: Sequence[dict[str, Any]],
) -> list[MetricResult]:
    injection_detected: list[float] = []
    pii_leaks: list[float] = []
    unsafe_outputs: list[float] = []

    for record in records:
        injection = record.get(
            "prompt_injection_detected"
        )

        pii = record.get(
            "pii_leak"
        )

        unsafe = record.get(
            "unsafe_output"
        )

        if injection is not None:
            injection_detected.append(
                1.0
                if bool(injection)
                else 0.0
            )

        if pii is not None:
            pii_leaks.append(
                1.0
                if bool(pii)
                else 0.0
            )

        if unsafe is not None:
            unsafe_outputs.append(
                1.0
                if bool(unsafe)
                else 0.0
            )

    return [
        metric(
            "prompt_injection_rate",
            mean(injection_detected),
            higher_is_better=False,
        ),
        metric(
            "pii_leak_rate",
            mean(pii_leaks),
            higher_is_better=False,
        ),
        metric(
            "unsafe_output_rate",
            mean(unsafe_outputs),
            higher_is_better=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Performance evaluation
# ---------------------------------------------------------------------------


def evaluate_performance(
    records: Sequence[dict[str, Any]],
) -> list[MetricResult]:
    latency_values: list[float] = []
    token_values: list[float] = []
    cost_values: list[float] = []

    for record in records:
        latency = safe_float(
            first_present(
                record,
                [
                    "latency_ms",
                    "latency",
                    "response_time_ms",
                ],
            )
        )

        tokens = safe_float(
            first_present(
                record,
                [
                    "total_tokens",
                    "tokens",
                    "token_usage",
                ],
            )
        )

        cost = safe_float(
            first_present(
                record,
                [
                    "cost_usd",
                    "cost",
                    "estimated_cost_usd",
                ],
            )
        )

        if latency is not None:
            latency_values.append(
                latency
            )

        if tokens is not None:
            token_values.append(
                tokens
            )

        if cost is not None:
            cost_values.append(
                cost
            )

    return [
        metric(
            "latency_mean",
            mean(latency_values),
            unit="ms",
            higher_is_better=False,
        ),
        metric(
            "latency_p50",
            percentile(
                latency_values,
                50,
            ),
            unit="ms",
            higher_is_better=False,
        ),
        metric(
            "latency_p95",
            percentile(
                latency_values,
                95,
            ),
            unit="ms",
            higher_is_better=False,
        ),
        metric(
            "latency_p99",
            percentile(
                latency_values,
                99,
            ),
            unit="ms",
            higher_is_better=False,
        ),
        metric(
            "average_tokens",
            mean(token_values),
            unit="tokens",
            higher_is_better=False,
        ),
        metric(
            "average_cost",
            mean(cost_values),
            unit="USD",
            higher_is_better=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def calculate_overall_score(
    evaluation: ModelEvaluation,
) -> float | None:
    """
    Calculate a balanced score.

    The score deliberately excludes unavailable metrics.
    This prevents missing evaluation dimensions from
    silently becoming zeros.
    """

    values: list[float] = []

    for collection in (
        evaluation.classification,
        evaluation.regression,
        evaluation.retrieval,
        evaluation.rag,
        evaluation.safety,
    ):
        for item in collection:
            if (
                item.value is None
                or item.status != "available"
            ):
                continue

            value = item.value

            if not item.higher_is_better:
                # Normalize common error/rate metrics.
                if item.name in {
                    "mae",
                    "rmse",
                }:
                    value = 1.0 / (
                        1.0 + max(value, 0.0)
                    )

                elif item.name.endswith(
                    "_rate"
                ):
                    value = 1.0 - clamp(
                        value
                    )

                elif item.name.startswith(
                    "latency"
                ):
                    value = 1.0 / (
                        1.0 + max(value, 0.0)
                    )

            values.append(
                clamp(value)
            )

    if not values:
        return None

    return mean(values)


def evaluate_model(
    model_name: str,
    records: Sequence[dict[str, Any]],
    top_k: int,
) -> ModelEvaluation:
    """Evaluate one model against the dataset."""

    evaluation = ModelEvaluation(
        model_name=model_name
    )

    labels, predictions = (
        get_predictions(
            records,
            model_name,
        )
    )

    evaluation.classification = (
        evaluate_classification(
            labels,
            predictions,
        )
    )

    actual, predicted = (
        get_regression_values(
            records,
            model_name,
        )
    )

    evaluation.regression = (
        evaluate_regression(
            actual,
            predicted,
        )
    )

    evaluation.retrieval = (
        evaluate_retrieval(
            records,
            top_k,
        )
    )

    evaluation.rag = evaluate_rag(
        records
    )

    evaluation.safety = (
        evaluate_safety(
            records
        )
    )

    evaluation.performance = (
        evaluate_performance(
            records
        )
    )

    # Hallucination metrics are represented
    # under safety because they are safety-critical
    # for legal RAG.
    evaluation.safety.extend(
        evaluate_hallucination(
            records
        )
    )

    evaluation.overall_score = (
        calculate_overall_score(
            evaluation
        )
    )

    evaluation.status = (
        "evaluated"
        if evaluation.overall_score is not None
        else "insufficient_data"
    )

    return evaluation


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def discover_models(
    records: Sequence[dict[str, Any]],
) -> list[str]:
    models: set[str] = set()

    for record in records:
        predictions = record.get(
            "predictions"
        )

        if isinstance(
            predictions,
            dict,
        ):
            models.update(
                str(key)
                for key in predictions
            )

        model_name = record.get(
            "model_name"
        )

        if model_name:
            models.add(
                str(model_name)
            )

    if not models:
        models.add("default")

    return sorted(models)


# ---------------------------------------------------------------------------
# Baseline / regression checking
# ---------------------------------------------------------------------------


def flatten_metrics(
    evaluation: ModelEvaluation,
) -> dict[str, float]:
    flattened: dict[str, float] = {}

    for collection in (
        evaluation.classification,
        evaluation.regression,
        evaluation.retrieval,
        evaluation.rag,
        evaluation.performance,
        evaluation.safety,
    ):
        for item in collection:
            if item.value is not None:
                flattened[item.name] = (
                    item.value
                )

    if evaluation.overall_score is not None:
        flattened["overall_score"] = (
            evaluation.overall_score
        )

    return flattened


def load_baseline(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except Exception:
        logger.warning(
            "Could not load baseline.",
            exc_info=True,
        )
        return None


def extract_baseline_metrics(
    baseline: dict[str, Any],
) -> dict[str, float]:
    evaluations = baseline.get(
        "evaluations",
        [],
    )

    if not evaluations:
        return {}

    # Prefer explicit baseline model.
    baseline_model = baseline.get(
        "baseline_model"
    )

    selected = None

    for evaluation in evaluations:
        if (
            baseline_model
            and evaluation.get(
                "model_name"
            )
            == baseline_model
        ):
            selected = evaluation
            break

    if selected is None:
        selected = evaluations[0]

    metrics: dict[str, float] = {}

    for collection_name in (
        "classification",
        "regression",
        "retrieval",
        "rag",
        "performance",
        "safety",
    ):
        for item in selected.get(
            collection_name,
            [],
        ):
            value = item.get(
                "value"
            )

            if value is not None:
                metrics[
                    item["name"]
                ] = float(value)

    if selected.get(
        "overall_score"
    ) is not None:
        metrics[
            "overall_score"
        ] = float(
            selected[
                "overall_score"
            ]
        )

    return metrics


def check_regressions(
    evaluation: ModelEvaluation,
    baseline_metrics: dict[str, float],
    max_regression: float,
) -> list[dict[str, Any]]:
    current = flatten_metrics(
        evaluation
    )

    regressions: list[
        dict[str, Any]
    ] = []

    lower_is_better = {
        "mae",
        "rmse",
        "latency_mean",
        "latency_p50",
        "latency_p95",
        "latency_p99",
        "average_tokens",
        "average_cost",
        "hallucination_rate",
        "prompt_injection_rate",
        "pii_leak_rate",
        "unsafe_output_rate",
    }

    for name, baseline_value in (
        baseline_metrics.items()
    ):
        current_value = current.get(
            name
        )

        if current_value is None:
            continue

        if baseline_value == 0:
            continue

        if name in lower_is_better:
            relative_change = (
                baseline_value
                - current_value
            ) / abs(
                baseline_value
            )

            regressed = (
                relative_change
                < -max_regression
            )

        else:
            relative_change = (
                current_value
                - baseline_value
            ) / abs(
                baseline_value
            )

            regressed = (
                relative_change
                < -max_regression
            )

        if regressed:
            regressions.append(
                {
                    "model": (
                        evaluation.model_name
                    ),
                    "metric": name,
                    "baseline": (
                        baseline_value
                    ),
                    "current": (
                        current_value
                    ),
                    "relative_change": (
                        relative_change
                    ),
                    "threshold": (
                        max_regression
                    ),
                }
            )

    return regressions


# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------


def run_quality_gates(
    evaluations: Sequence[ModelEvaluation],
    min_accuracy: float,
    min_f1: float,
    min_recall: float,
    max_latency_ms: float,
) -> tuple[
    bool,
    list[str],
]:
    failures: list[str] = []

    for evaluation in evaluations:
        metrics = flatten_metrics(
            evaluation
        )

        accuracy = metrics.get(
            "accuracy"
        )

        if (
            accuracy is not None
            and accuracy < min_accuracy
        ):
            failures.append(
                f"{evaluation.model_name}: "
                f"accuracy {accuracy:.4f} "
                f"< {min_accuracy:.4f}"
            )

        f1 = metrics.get(
            "f1_macro"
        )

        if (
            f1 is not None
            and f1 < min_f1
        ):
            failures.append(
                f"{evaluation.model_name}: "
                f"f1_macro {f1:.4f} "
                f"< {min_f1:.4f}"
            )

        recall = metrics.get(
            "recall_macro"
        )

        if (
            recall is not None
            and recall < min_recall
        ):
            failures.append(
                f"{evaluation.model_name}: "
                f"recall_macro {recall:.4f} "
                f"< {min_recall:.4f}"
            )

        latency = metrics.get(
            "latency_p95"
        )

        if (
            latency is not None
            and latency > max_latency_ms
        ):
            failures.append(
                f"{evaluation.model_name}: "
                f"p95 latency {latency:.2f}ms "
                f"> {max_latency_ms:.2f}ms"
            )

    return (
        not failures,
        failures,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_json_report(
    report: EvaluationReport,
    output: Path,
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            asdict(report),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_csv_report(
    report: EvaluationReport,
    output: Path,
) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict[str, Any]] = []

    for evaluation in report.evaluations:
        for collection_name in (
            "classification",
            "regression",
            "retrieval",
            "rag",
            "performance",
            "safety",
        ):
            collection = getattr(
                evaluation,
                collection_name,
            )

            for item in collection:
                rows.append(
                    {
                        "model": (
                            evaluation.model_name
                        ),
                        "category": (
                            collection_name
                        ),
                        "metric": item.name,
                        "value": item.value,
                        "unit": item.unit,
                        "higher_is_better": (
                            item.higher_is_better
                        ),
                        "status": item.status,
                    }
                )

        rows.append(
            {
                "model": (
                    evaluation.model_name
                ),
                "category": "overall",
                "metric": "overall_score",
                "value": (
                    evaluation.overall_score
                ),
                "unit": "",
                "higher_is_better": True,
                "status": evaluation.status,
            }
        )

    with output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "category",
                "metric",
                "value",
                "unit",
                "higher_is_better",
                "status",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate legal-document AI models, "
            "RAG quality, safety and performance."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Evaluation JSON/JSONL dataset.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON evaluation report.",
    )

    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help="CSV evaluation report.",
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Previous evaluation report.",
    )

    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "Model name to evaluate. "
            "May be specified multiple times."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Retrieval top-k.",
    )

    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=DEFAULT_MIN_ACCURACY,
    )

    parser.add_argument(
        "--min-f1",
        type=float,
        default=DEFAULT_MIN_F1,
    )

    parser.add_argument(
        "--min-recall",
        type=float,
        default=DEFAULT_MIN_RECALL,
    )

    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=DEFAULT_MAX_LATENCY_MS,
    )

    parser.add_argument(
        "--max-regression",
        type=float,
        default=DEFAULT_MAX_REGRESSION,
        help=(
            "Maximum allowed relative metric regression."
        ),
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Fail if baseline regression is detected.",
    )

    parser.add_argument(
        "--fail-on-quality-gate",
        action="store_true",
        help="Fail when configured quality gates fail.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError(
            "--top-k must be greater than zero."
        )

    if not 0 <= args.max_regression:
        raise ValueError(
            "--max-regression must be >= 0."
        )

    started = time.perf_counter()

    try:
        records = load_dataset(
            args.dataset
        )

        if not records:
            logger.error(
                "Evaluation dataset is empty."
            )
            return 1

        discovered_models = (
            discover_models(
                records
            )
        )

        model_names = (
            args.models
            if args.models
            else discovered_models
        )

        logger.info(
            "Models to evaluate: %s",
            ", ".join(model_names),
        )

        evaluations: list[
            ModelEvaluation
        ] = []

        for model_name in model_names:
            logger.info(
                "Evaluating model: %s",
                model_name,
            )

            evaluation = evaluate_model(
                model_name=model_name,
                records=records,
                top_k=args.top_k,
            )

            evaluations.append(
                evaluation
            )

            logger.info(
                "%s overall score: %s",
                model_name,
                (
                    f"{evaluation.overall_score:.4f}"
                    if evaluation.overall_score
                    is not None
                    else "N/A"
                ),
            )

        baseline = load_baseline(
            args.baseline
        )

        baseline_metrics: dict[
            str,
            float,
        ] = {}

        baseline_model = None

        if baseline:
            baseline_metrics = (
                extract_baseline_metrics(
                    baseline
                )
            )

            baseline_model = baseline.get(
                "baseline_model"
            )

        regressions: list[
            dict[str, Any]
        ] = []

        if baseline_metrics:
            for evaluation in evaluations:
                regressions.extend(
                    check_regressions(
                        evaluation=evaluation,
                        baseline_metrics=(
                            baseline_metrics
                        ),
                        max_regression=(
                            args.max_regression
                        ),
                    )
                )

        quality_passed, quality_failures = (
            run_quality_gates(
                evaluations=evaluations,
                min_accuracy=args.min_accuracy,
                min_f1=args.min_f1,
                min_recall=args.min_recall,
                max_latency_ms=(
                    args.max_latency_ms
                ),
            )
        )

        for failure in quality_failures:
            logger.error(
                "Quality gate failed: %s",
                failure,
            )

        report = EvaluationReport(
            schema_version="1.0",
            generated_at=utc_now(),
            dataset=str(
                args.dataset
            ),
            dataset_hash=sha256_file(
                args.dataset
            ),
            sample_count=len(records),
            evaluations=evaluations,
            baseline_model=baseline_model,
            regressions=regressions,
            passed=(
                quality_passed
                and not (
                    args.fail_on_regression
                    and regressions
                )
            ),
        )

        report.errors.extend(
            quality_failures
        )

        if args.fail_on_regression:
            report.errors.extend(
                [
                    (
                        f"Regression: "
                        f"{item['model']} / "
                        f"{item['metric']}"
                    )
                    for item in regressions
                ]
            )

        write_json_report(
            report,
            args.output,
        )

        write_csv_report(
            report,
            args.csv_output,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        logger.info(
            "Model evaluation completed "
            "in %.2f seconds.",
            elapsed,
        )

        logger.info(
            "JSON report: %s",
            args.output,
        )

        logger.info(
            "CSV report: %s",
            args.csv_output,
        )

        logger.info(
            "Overall status: %s",
            "PASS"
            if report.passed
            else "FAIL",
        )

        if (
            args.fail_on_quality_gate
            and not quality_passed
        ):
            return 2

        if (
            args.fail_on_regression
            and regressions
        ):
            return 3

        return 0

    except Exception:
        logger.exception(
            "Model evaluation failed."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())