"""
Legal Named Entity Recognition (NER).

Path:
    backend/models/huggingface/legal_ner.py

Responsibilities:
    - Extract legal entities from contract/document text.
    - Support Hugging Face token-classification models.
    - Normalize model-specific labels.
    - Merge subword/token spans.
    - Provide confidence scores and character offsets.
    - Support batch inference.
    - Remain independent from legal risk/decision engines.

Non-responsibilities:
    - Legal advice.
    - Contract interpretation.
    - Risk scoring.
    - Clause classification.
    - Enforceability/breach determination.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence
import re


class LegalEntityType(str, Enum):
    """Canonical legal entity categories."""

    PERSON = "person"
    ORGANIZATION = "organization"
    PARTY = "party"
    COURT = "court"
    JURISDICTION = "jurisdiction"
    LAW = "law"
    REGULATION = "regulation"
    CASE = "case"
    CONTRACT = "contract"
    CLAUSE = "clause"
    DATE = "date"
    DURATION = "duration"
    MONEY = "money"
    PERCENTAGE = "percentage"
    ADDRESS = "address"
    LOCATION = "location"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    IDENTIFIER = "identifier"
    PRODUCT = "product"
    SERVICE = "service"
    EFFECTIVE_DATE = "effective_date"
    EXPIRY_DATE = "expiry_date"
    SIGNATURE = "signature"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class LegalEntity:
    """Single extracted legal entity."""

    text: str
    entity_type: LegalEntityType
    start: int
    end: int
    confidence: float = 0.0

    label: Optional[str] = None
    sentence: Optional[str] = None
    normalized_value: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class LegalNERResult:
    """NER inference result for one document/text."""

    text: str
    entities: list[LegalEntity]

    model_name: str
    model_version: str

    processing_time_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    def by_type(
        self,
        entity_type: LegalEntityType,
    ) -> list[LegalEntity]:
        return [
            entity
            for entity in self.entities
            if entity.entity_type == entity_type
        ]


class LegalNER:
    """
    Legal NER inference wrapper.

    The implementation supports:
        - Injected Hugging Face pipeline.
        - Lazy model loading.
        - Batch inference.
        - Label normalization.
        - Entity span merging.
        - Lightweight deterministic fallback extraction.

    Recommended production model:
        A domain-specific legal NER/token-classification model
        configured through the application's model registry.

    Example:
        ner = LegalNER(
            model_name="your-org/legal-ner-model",
            model_version="v1",
        )

        result = ner.extract(text)

        for entity in result.entities:
            print(
                entity.text,
                entity.entity_type,
                entity.confidence,
            )
    """

    DEFAULT_LABEL_ALIASES: dict[str, LegalEntityType] = {
        # People
        "PER": LegalEntityType.PERSON,
        "PERSON": LegalEntityType.PERSON,
        "B-PER": LegalEntityType.PERSON,
        "I-PER": LegalEntityType.PERSON,

        # Organizations
        "ORG": LegalEntityType.ORGANIZATION,
        "ORGANIZATION": LegalEntityType.ORGANIZATION,
        "B-ORG": LegalEntityType.ORGANIZATION,
        "I-ORG": LegalEntityType.ORGANIZATION,

        # Parties
        "PARTY": LegalEntityType.PARTY,
        "LEGAL_PARTY": LegalEntityType.PARTY,
        "B-PARTY": LegalEntityType.PARTY,
        "I-PARTY": LegalEntityType.PARTY,

        # Courts
        "COURT": LegalEntityType.COURT,
        "B-COURT": LegalEntityType.COURT,
        "I-COURT": LegalEntityType.COURT,

        # Jurisdiction / geography
        "JURISDICTION": LegalEntityType.JURISDICTION,
        "GPE": LegalEntityType.LOCATION,
        "LOC": LegalEntityType.LOCATION,
        "LOCATION": LegalEntityType.LOCATION,

        # Legal authorities
        "LAW": LegalEntityType.LAW,
        "STATUTE": LegalEntityType.LAW,
        "REGULATION": LegalEntityType.REGULATION,
        "REG": LegalEntityType.REGULATION,
        "CASE": LegalEntityType.CASE,

        # Contract-related
        "CONTRACT": LegalEntityType.CONTRACT,
        "CLAUSE": LegalEntityType.CLAUSE,

        # Temporal
        "DATE": LegalEntityType.DATE,
        "TIME": LegalEntityType.DATE,
        "DURATION": LegalEntityType.DURATION,
        "EFFECTIVE_DATE": LegalEntityType.EFFECTIVE_DATE,
        "EXPIRY_DATE": LegalEntityType.EXPIRY_DATE,

        # Financial
        "MONEY": LegalEntityType.MONEY,
        "CURRENCY": LegalEntityType.MONEY,
        "PERCENT": LegalEntityType.PERCENTAGE,
        "PERCENTAGE": LegalEntityType.PERCENTAGE,

        # Contact
        "ADDRESS": LegalEntityType.ADDRESS,
        "EMAIL": LegalEntityType.EMAIL,
        "PHONE": LegalEntityType.PHONE,
        "URL": LegalEntityType.URL,

        # Other
        "PRODUCT": LegalEntityType.PRODUCT,
        "SERVICE": LegalEntityType.SERVICE,
        "IDENTIFIER": LegalEntityType.IDENTIFIER,
        "SIGNATURE": LegalEntityType.SIGNATURE,
    }

    def __init__(
        self,
        model_name: str = "legal-ner-model",
        model_version: str = "v1",
        device: Optional[str | int] = None,
        aggregation_strategy: str = "simple",
        min_confidence: float = 0.50,
        pipeline: Any = None,
        label_aliases: Optional[
            dict[str, LegalEntityType]
        ] = None,
        enable_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self.min_confidence = max(0.0, min(1.0, min_confidence))

        self._pipeline = pipeline
        self.enable_fallback = enable_fallback

        self.label_aliases = dict(self.DEFAULT_LABEL_ALIASES)

        if label_aliases:
            self.label_aliases.update(label_aliases)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LegalNERResult:
        """
        Extract legal entities from text.
        """

        self._validate_text(text)

        if not text.strip():
            return LegalNERResult(
                text=text,
                entities=[],
                model_name=self.model_name,
                model_version=self.model_version,
                metadata=metadata or {},
            )

        raw_entities: list[dict[str, Any]] = []

        if self._pipeline is not None or self._load_pipeline_available():
            try:
                raw_entities = self._run_model(text)
            except Exception:
                if not self.enable_fallback:
                    raise

        entities = self._normalize_entities(
            text=text,
            raw_entities=raw_entities,
        )

        if self.enable_fallback:
            entities = self._merge_with_fallback(
                text=text,
                entities=entities,
            )

        entities = self._sort_and_deduplicate(entities)

        return LegalNERResult(
            text=text,
            entities=entities,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata=metadata or {},
        )

    def extract_batch(
        self,
        texts: Sequence[str],
    ) -> list[LegalNERResult]:
        """
        Extract entities from multiple texts.
        """

        if not texts:
            return []

        for text in texts:
            self._validate_text(text)

        # Use individual calls when the underlying pipeline is not
        # available or does not support batch processing reliably.
        if self._pipeline is None and not self._load_pipeline_available():
            return [self.extract(text) for text in texts]

        try:
            raw_results = self._run_model_batch(texts)

            results: list[LegalNERResult] = []

            for text, raw_entities in zip(texts, raw_results):
                entities = self._normalize_entities(
                    text=text,
                    raw_entities=raw_entities,
                )

                if self.enable_fallback:
                    entities = self._merge_with_fallback(
                        text=text,
                        entities=entities,
                    )

                results.append(
                    LegalNERResult(
                        text=text,
                        entities=self._sort_and_deduplicate(
                            entities
                        ),
                        model_name=self.model_name,
                        model_version=self.model_version,
                    )
                )

            return results

        except Exception:
            return [self.extract(text) for text in texts]

    def extract_by_type(
        self,
        text: str,
        entity_type: LegalEntityType,
    ) -> list[LegalEntity]:
        """Convenience method for extracting one entity category."""

        return self.extract(text).by_type(entity_type)

    def get_model_info(self) -> dict[str, Any]:
        """Return model configuration metadata."""

        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "aggregation_strategy": self.aggregation_strategy,
            "min_confidence": self.min_confidence,
            "fallback_enabled": self.enable_fallback,
            "entity_types": [
                entity_type.value
                for entity_type in LegalEntityType
            ],
        }

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def _load_pipeline_available(self) -> bool:
        """
        Lazily load Hugging Face pipeline.

        Returns True if a pipeline is available.
        """

        if self._pipeline is not None:
            return True

        try:
            from transformers import pipeline as hf_pipeline

            kwargs: dict[str, Any] = {
                "task": "token-classification",
                "model": self.model_name,
                "aggregation_strategy": self.aggregation_strategy,
            }

            if self.device is not None:
                kwargs["device"] = self.device

            self._pipeline = hf_pipeline(**kwargs)

            return True

        except ImportError:
            return False

    def _run_model(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """Run the Hugging Face NER pipeline."""

        if self._pipeline is None:
            raise RuntimeError("NER pipeline is not initialized.")

        result = self._pipeline(text)

        if isinstance(result, dict):
            return [result]

        return list(result)

    def _run_model_batch(
        self,
        texts: Sequence[str],
    ) -> list[list[dict[str, Any]]]:
        """Run NER inference for multiple texts."""

        if self._pipeline is None:
            raise RuntimeError("NER pipeline is not initialized.")

        result = self._pipeline(list(texts))

        normalized: list[list[dict[str, Any]]] = []

        for item in result:
            if isinstance(item, dict):
                normalized.append([item])
            else:
                normalized.append(list(item))

        return normalized

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize_entities(
        self,
        *,
        text: str,
        raw_entities: Iterable[dict[str, Any]],
    ) -> list[LegalEntity]:
        entities: list[LegalEntity] = []

        for raw in raw_entities:
            if not isinstance(raw, dict):
                continue

            entity_text = str(
                raw.get("word")
                or raw.get("text")
                or ""
            ).strip()

            if not entity_text:
                continue

            start = raw.get("start")
            end = raw.get("end")

            if start is None or end is None:
                start, end = self._find_span(
                    text,
                    entity_text,
                )

            if start is None or end is None:
                continue

            try:
                start = int(start)
                end = int(end)
            except (TypeError, ValueError):
                continue

            if start < 0 or end <= start or end > len(text):
                continue

            confidence = self._confidence(raw)

            if confidence < self.min_confidence:
                continue

            raw_label = str(
                raw.get("entity_group")
                or raw.get("entity")
                or raw.get("label")
                or "UNKNOWN"
            )

            entity_type = self._map_label(raw_label)

            actual_text = text[start:end]

            entities.append(
                LegalEntity(
                    text=actual_text,
                    entity_type=entity_type,
                    start=start,
                    end=end,
                    confidence=confidence,
                    label=raw_label,
                    sentence=self._extract_sentence(
                        text,
                        start,
                        end,
                    ),
                    normalized_value=self._normalize_value(
                        actual_text,
                        entity_type,
                    ),
                    metadata={
                        "model": self.model_name,
                        "model_version": self.model_version,
                    },
                )
            )

        return entities

    def _map_label(
        self,
        label: str,
    ) -> LegalEntityType:
        normalized = label.upper().strip()

        if normalized in self.label_aliases:
            return self.label_aliases[normalized]

        # Handle labels such as:
        # B-ORG, I-ORG, B-LEGAL_PERSON
        normalized = re.sub(
            r"^[BI]-",
            "",
            normalized,
        )

        if normalized in self.label_aliases:
            return self.label_aliases[normalized]

        normalized = normalized.replace("-", "_")

        if normalized in self.label_aliases:
            return self.label_aliases[normalized]

        return LegalEntityType.UNKNOWN

    @staticmethod
    def _confidence(raw: dict[str, Any]) -> float:
        value = raw.get("score", raw.get("confidence", 0.0))

        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # Deterministic fallback
    # ------------------------------------------------------------------

    def _merge_with_fallback(
        self,
        *,
        text: str,
        entities: list[LegalEntity],
    ) -> list[LegalEntity]:
        """
        Add high-confidence deterministic entities that are easy to
        identify without an ML model.

        This is intentionally conservative.
        """

        fallback = self._fallback_entities(text)

        combined = entities + fallback

        return self._sort_and_deduplicate(combined)

    def _fallback_entities(
        self,
        text: str,
    ) -> list[LegalEntity]:
        entities: list[LegalEntity] = []

        patterns: list[
            tuple[str, LegalEntityType]
        ] = [
            (
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                LegalEntityType.DATE,
            ),
            (
                r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
                LegalEntityType.DATE,
            ),
            (
                r"\b(?:USD|EUR|GBP|INR|₹|\$|€|£)\s?"
                r"\d[\d,]*(?:\.\d+)?\b",
                LegalEntityType.MONEY,
            ),
            (
                r"\b\d+(?:\.\d+)?%\b",
                LegalEntityType.PERCENTAGE,
            ),
            (
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                LegalEntityType.EMAIL,
            ),
            (
                r"\b(?:https?://|www\.)"
                r"[^\s<>\"]+",
                LegalEntityType.URL,
            ),
        ]

        for pattern, entity_type in patterns:
            for match in re.finditer(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                entities.append(
                    LegalEntity(
                        text=match.group(0),
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.99,
                        label=f"RULE_{entity_type.value.upper()}",
                        sentence=self._extract_sentence(
                            text,
                            match.start(),
                            match.end(),
                        ),
                        normalized_value=self._normalize_value(
                            match.group(0),
                            entity_type,
                        ),
                        metadata={
                            "source": "deterministic_fallback",
                        },
                    )
                )

        return entities

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _find_span(
        text: str,
        entity_text: str,
    ) -> tuple[Optional[int], Optional[int]]:
        index = text.find(entity_text)

        if index < 0:
            return None, None

        return index, index + len(entity_text)

    @staticmethod
    def _extract_sentence(
        text: str,
        start: int,
        end: int,
    ) -> str:
        left = text.rfind(".", 0, start)

        right = text.find(".", end)

        sentence_start = 0 if left == -1 else left + 1
        sentence_end = len(text) if right == -1 else right + 1

        return text[
            sentence_start:sentence_end
        ].strip()

    @staticmethod
    def _normalize_value(
        value: str,
        entity_type: LegalEntityType,
    ) -> str:
        normalized = value.strip()

        if entity_type in {
            LegalEntityType.EMAIL,
            LegalEntityType.URL,
        }:
            return normalized.lower()

        if entity_type in {
            LegalEntityType.MONEY,
            LegalEntityType.PERCENTAGE,
            LegalEntityType.DATE,
        }:
            return re.sub(
                r"\s+",
                " ",
                normalized,
            )

        return re.sub(
            r"\s+",
            " ",
            normalized,
        )

    @staticmethod
    def _validate_text(text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if len(text) > 2_000_000:
            raise ValueError(
                "text exceeds the maximum supported size"
            )

    @staticmethod
    def _sort_and_deduplicate(
        entities: list[LegalEntity],
    ) -> list[LegalEntity]:
        """
        Remove overlapping duplicates.

        Higher-confidence entities win.
        """

        if not entities:
            return []

        ordered = sorted(
            entities,
            key=lambda entity: (
                entity.start,
                -(entity.end - entity.start),
                -entity.confidence,
            ),
        )

        result: list[LegalEntity] = []

        for candidate in ordered:
            duplicate = False

            for existing in result:
                overlaps = (
                    candidate.start < existing.end
                    and candidate.end > existing.start
                )

                if not overlaps:
                    continue

                # Keep the more confident span.
                if (
                    candidate.confidence
                    <= existing.confidence
                ):
                    duplicate = True
                    break

                result.remove(existing)
                break

            if not duplicate:
                result.append(candidate)

        return sorted(
            result,
            key=lambda entity: (
                entity.start,
                entity.end,
            ),
        )


__all__ = [
    "LegalEntityType",
    "LegalEntity",
    "LegalNERResult",
    "LegalNER",
]