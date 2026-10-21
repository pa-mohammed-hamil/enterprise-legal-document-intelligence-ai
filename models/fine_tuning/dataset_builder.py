"""
Dataset builder for legal AI model training.

Supported tasks:
- document classification
- clause classification
- legal NER
- risk classification

Responsibilities:
- Load JSON / JSONL / CSV datasets.
- Normalize records into a common schema.
- Validate required fields.
- Remove duplicates.
- Normalize labels.
- Perform deterministic train/validation/test splits.
- Export Hugging Face Datasets.
- Preserve metadata for reproducibility.

Expected examples
-----------------

Document classification:
{
    "text": "...",
    "label": "nda"
}

Clause classification:
{
    "text": "...",
    "label": "termination"
}

Risk classification:
{
    "text": "...",
    "label": "high"
}

Legal NER:
{
    "tokens": ["The", "agreement", "expires", "on", "31", "March", "2027"],
    "ner_tags": ["O", "O", "O", "O", "B-DATE", "I-DATE", "I-DATE"]
}
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from datasets import Dataset, DatasetDict
except ImportError:  # pragma: no cover
    Dataset = None
    DatasetDict = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


SUPPORTED_EXTENSIONS = {
    ".json",
    ".jsonl",
    ".csv",
}


@dataclass(frozen=True)
class SplitConfig:
    """
    Dataset split configuration.

    The default is appropriate for supervised ML experiments.
    """

    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42

    def __post_init__(self) -> None:
        total = (
            self.train_ratio
            + self.validation_ratio
            + self.test_ratio
        )

        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "train_ratio + validation_ratio + test_ratio must equal 1.0."
            )

        for value in (
            self.train_ratio,
            self.validation_ratio,
            self.test_ratio,
        ):
            if value < 0:
                raise ValueError(
                    "Split ratios cannot be negative."
                )


@dataclass
class DatasetRecord:
    """
    Canonical supervised-learning record.
    """

    text: str | None = None
    label: str | int | None = None

    tokens: list[str] | None = None
    ner_tags: list[str | int] | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetBuildResult:
    """
    Result of a dataset-building operation.
    """

    total_records: int
    train_records: int
    validation_records: int
    test_records: int
    labels: list[str]
    duplicate_records_removed: int
    invalid_records_removed: int


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


class DatasetBuilder:
    """
    Reusable dataset preparation pipeline.

    Pipeline:

        Raw data
           ↓
        Load
           ↓
        Normalize
           ↓
        Validate
           ↓
        Deduplicate
           ↓
        Normalize labels
           ↓
        Deterministic split
           ↓
        Hugging Face Dataset
           ↓
        Training
    """

    def __init__(
        self,
        *,
        task: str,
        split_config: SplitConfig | None = None,
        seed: int = 42,
        lowercase_labels: bool = True,
    ) -> None:

        self.task = task.strip().lower()

        self.split_config = split_config or SplitConfig(
            seed=seed
        )

        self.seed = seed
        self.lowercase_labels = lowercase_labels

        self._records: list[DatasetRecord] = []
        self._duplicate_count = 0
        self._invalid_count = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        path: str | Path,
    ) -> "DatasetBuilder":

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported dataset format: {path.suffix}"
            )

        if path.suffix.lower() == ".json":
            records = self._load_json(path)

        elif path.suffix.lower() == ".jsonl":
            records = self._load_jsonl(path)

        else:
            records = self._load_csv(path)

        for record in records:
            normalized = self._normalize_record(record)

            if normalized is None:
                self._invalid_count += 1
                continue

            self._records.append(normalized)

        return self

    def load_records(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> "DatasetBuilder":

        for record in records:
            normalized = self._normalize_record(record)

            if normalized is None:
                self._invalid_count += 1
                continue

            self._records.append(normalized)

        return self

    # ------------------------------------------------------------------
    # File readers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(
        path: Path,
    ) -> list[Mapping[str, Any]]:

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        if isinstance(data, dict):
            if "records" in data:
                data = data["records"]
            else:
                data = [data]

        if not isinstance(data, list):
            raise ValueError(
                "JSON dataset must contain an object or list of objects."
            )

        return data

    @staticmethod
    def _load_jsonl(
        path: Path,
    ) -> list[Mapping[str, Any]]:

        records: list[Mapping[str, Any]] = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number}."
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL line {line_number} must be an object."
                    )

                records.append(record)

        return records

    @staticmethod
    def _load_csv(
        path: Path,
    ) -> list[Mapping[str, Any]]:

        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:

            return list(csv.DictReader(file))

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_record(
        self,
        record: Mapping[str, Any],
    ) -> DatasetRecord | None:

        if self.task == "legal_ner":
            return self._normalize_ner(record)

        text = record.get("text")

        if text is None:
            text = record.get("content")

        if text is None:
            return None

        text = str(text).strip()

        if not text:
            return None

        label = record.get("label")

        if label is None:
            label = record.get("category")

        if label is None:
            return None

        label = self._normalize_label(label)

        metadata = dict(
            record.get("metadata") or {}
        )

        record_id = record.get("record_id")

        if not record_id:
            record_id = self._make_record_id(
                text=text,
                label=str(label),
            )

        return DatasetRecord(
            text=text,
            label=label,
            metadata=metadata,
            record_id=str(record_id),
        )

    def _normalize_ner(
        self,
        record: Mapping[str, Any],
    ) -> DatasetRecord | None:

        tokens = record.get("tokens")
        ner_tags = record.get("ner_tags")

        if not tokens or not ner_tags:
            return None

        if len(tokens) != len(ner_tags):
            return None

        normalized_tokens = [
            str(token)
            for token in tokens
        ]

        normalized_tags = [
            self._normalize_label(tag)
            if isinstance(tag, str)
            else tag
            for tag in ner_tags
        ]

        record_id = record.get("record_id")

        if not record_id:
            record_id = self._make_record_id(
                text=" ".join(normalized_tokens),
                label=json.dumps(
                    normalized_tags,
                    sort_keys=True,
                ),
            )

        return DatasetRecord(
            tokens=normalized_tokens,
            ner_tags=normalized_tags,
            metadata=dict(
                record.get("metadata") or {}
            ),
            record_id=str(record_id),
        )

    # ------------------------------------------------------------------
    # Label normalization
    # ------------------------------------------------------------------

    def _normalize_label(
        self,
        label: Any,
    ) -> str | int:

        if isinstance(label, int):
            return label

        value = str(label).strip()

        if self.lowercase_labels:
            value = value.lower()

        # Normalize common formatting differences.
        value = value.replace("-", "_")
        value = value.replace(" ", "_")

        return value

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(
        self,
    ) -> "DatasetBuilder":

        valid: list[DatasetRecord] = []

        for record in self._records:

            if self.task == "legal_ner":

                if not record.tokens:
                    self._invalid_count += 1
                    continue

                if not record.ner_tags:
                    self._invalid_count += 1
                    continue

                if len(record.tokens) != len(record.ner_tags):
                    self._invalid_count += 1
                    continue

            else:

                if not record.text:
                    self._invalid_count += 1
                    continue

                if record.label is None:
                    self._invalid_count += 1
                    continue

            valid.append(record)

        self._records = valid

        return self

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def deduplicate(self) -> "DatasetBuilder":

        unique: dict[str, DatasetRecord] = {}

        for record in self._records:

            if record.record_id is None:
                key = self._make_record_id(
                    text=record.text or "",
                    label=str(record.label),
                )
            else:
                key = record.record_id

            if key in unique:
                self._duplicate_count += 1
                continue

            unique[key] = record

        self._records = list(unique.values())

        return self

    # ------------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------------

    def labels(self) -> list[str]:

        if self.task == "legal_ner":

            labels = {
                str(tag)
                for record in self._records
                for tag in (record.ner_tags or [])
            }

        else:

            labels = {
                str(record.label)
                for record in self._records
                if record.label is not None
        }

        return sorted(labels)

    def class_distribution(self) -> dict[str, int]:

        distribution: dict[str, int] = {}

        if self.task == "legal_ner":
            for record in self._records:
                for tag in record.ner_tags or []:
                    key = str(tag)
                    distribution[key] = (
                        distribution.get(key, 0) + 1
                    )

        else:
            for record in self._records:
                if record.label is None:
                    continue

                key = str(record.label)

                distribution[key] = (
                    distribution.get(key, 0) + 1
                )

        return dict(
            sorted(distribution.items())
        )

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split(
        self,
    ) -> dict[str, list[DatasetRecord]]:

        records = list(self._records)

        rng = random.Random(
            self.split_config.seed
        )

        rng.shuffle(records)

        total = len(records)

        train_end = int(
            total * self.split_config.train_ratio
        )

        validation_end = train_end + int(
            total * self.split_config.validation_ratio
        )

        return {
            "train": records[:train_end],
            "validation": records[
                train_end:validation_end
            ],
            "test": records[
                validation_end:
            ],
        }

    # ------------------------------------------------------------------
    # Hugging Face conversion
    # ------------------------------------------------------------------

    def to_huggingface(
        self,
    ) -> "Dataset":

        if Dataset is None:
            raise ImportError(
                "The 'datasets' package is required. "
                "Install it with: pip install datasets"
            )

        rows = [
            record.to_dict()
            for record in self._records
        ]

        return Dataset.from_list(rows)

    def to_dataset_dict(
        self,
    ) -> "DatasetDict":

        if DatasetDict is None:
            raise ImportError(
                "The 'datasets' package is required. "
                "Install it with: pip install datasets"
            )

        splits = self.split()

        return DatasetDict(
            {
                split_name: Dataset.from_list(
                    [
                        record.to_dict()
                        for record in split_records
                    ]
                )
                for split_name, split_records in splits.items()
            }
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_jsonl(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for record in self._records:
                file.write(
                    json.dumps(
                        record.to_dict(),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def save_splits(
        self,
        output_dir: str | Path,
    ) -> None:

        output_dir = Path(output_dir)

        splits = self.split()

        for split_name, records in splits.items():

            split_dir = output_dir / split_name

            split_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                split_dir / "data.jsonl"
            )

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:

                for record in records:
                    file.write(
                        json.dumps(
                            record.to_dict(),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
    ) -> tuple["DatasetDict", DatasetBuildResult]:

        self.validate()
        self.deduplicate()

        splits = self.split()

        dataset_dict = None

        if DatasetDict is not None:
            dataset_dict = DatasetDict(
                {
                    split_name: Dataset.from_list(
                        [
                            record.to_dict()
                            for record in records
                        ]
                    )
                    for split_name, records in splits.items()
                }
            )

        result = DatasetBuildResult(
            total_records=len(self._records),
            train_records=len(splits["train"]),
            validation_records=len(splits["validation"]),
            test_records=len(splits["test"]),
            labels=self.labels(),
            duplicate_records_removed=self._duplicate_count,
            invalid_records_removed=self._invalid_count,
        )

        return dataset_dict, result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_record_id(
        *,
        text: str,
        label: str,
    ) -> str:

        payload = (
            text.strip()
            + "\n"
            + label.strip()
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def build_classification_dataset(
    path: str | Path,
    *,
    seed: int = 42,
) -> tuple["DatasetDict", DatasetBuildResult]:

    builder = DatasetBuilder(
        task="classification",
        seed=seed,
    )

    return (
        builder
        .load(path)
        .validate()
        .deduplicate()
        .build()
    )


def build_document_classification_dataset(
    path: str | Path,
    *,
    seed: int = 42,
) -> tuple["DatasetDict", DatasetBuildResult]:

    builder = DatasetBuilder(
        task="document_classification",
        seed=seed,
    )

    return (
        builder
        .load(path)
        .validate()
        .deduplicate()
        .build()
    )


def build_clause_classification_dataset(
    path: str | Path,
    *,
    seed: int = 42,
) -> tuple["DatasetDict", DatasetBuildResult]:

    builder = DatasetBuilder(
        task="clause_classification",
        seed=seed,
    )

    return (
        builder
        .load(path)
        .validate()
        .deduplicate()
        .build()
    )


def build_risk_classification_dataset(
    path: str | Path,
    *,
    seed: int = 42,
) -> tuple["DatasetDict", DatasetBuildResult]:

    builder = DatasetBuilder(
        task="risk_classification",
        seed=seed,
    )

    return (
        builder
        .load(path)
        .validate()
        .deduplicate()
        .build()
    )


def build_legal_ner_dataset(
    path: str | Path,
    *,
    seed: int = 42,
) -> tuple["DatasetDict", DatasetBuildResult]:

    builder = DatasetBuilder(
        task="legal_ner",
        seed=seed,
    )

    return (
        builder
        .load(path)
        .validate()
        .deduplicate()
        .build()
    )


__all__ = [
    "SplitConfig",
    "DatasetRecord",
    "DatasetBuildResult",
    "DatasetBuilder",
    "build_classification_dataset",
    "build_document_classification_dataset",
    "build_clause_classification_dataset",
    "build_risk_classification_dataset",
    "build_legal_ner_dataset",
]