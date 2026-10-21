"""
Legal Risk Classifier.

Path:
    backend/models/huggingface/risk_classifier.py

Responsibilities:
    - Identify contractual/legal risk signals.
    - Classify risk categories.
    - Produce confidence and evidence.
    - Support Hugging Face sequence-classification models.
    - Provide deterministic fallback signals.
    - Return structured output for downstream workflows.

Non-responsibilities:
    - Legal advice.
    - Determining enforceability.
    - Determining whether a contract is legally valid.
    - Final compliance decisions.
    - Automatic legal action.
    - Human-review routing.

Architecture:

    Document / Clause
          |
          v
    RiskClassifier
          |
          +-- model prediction
          |
          +-- deterministic signals
          |
          v
    Structured RiskResult
          |
          +--> Risk Agent
          +--> Escalation Engine
          +--> Fire Engine
          +--> Human Review
          +--> Analytics / Evaluation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import logging
import re


logger = logging.getLogger(__name__)


# ======================================================================
# Enums
# ======================================================================


class RiskCategory(str, Enum):
    """Canonical contractual/legal risk categories."""

    LIABILITY = "liability"
    INDEMNIFICATION = "indemnification"
    TERMINATION = "termination"
    RENEWAL = "renewal"
    PAYMENT = "payment"
    PRICING = "pricing"

    DATA_PROTECTION = "data_protection"
    PRIVACY = "privacy"
    SECURITY = "security"

    INTELLECTUAL_PROPERTY = "intellectual_property"
    CONFIDENTIALITY = "confidentiality"

    COMPLIANCE = "compliance"
    REGULATORY = "regulatory"

    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"

    WARRANTY = "warranty"
    SERVICE_LEVEL = "service_level"
    DELIVERY = "delivery"

    ASSIGNMENT = "assignment"
    SUBCONTRACTING = "subcontracting"

    NON_COMPETE = "non_compete"
    NON_SOLICITATION = "non_solicitation"

    AUDIT = "audit"
    RECORDS = "records"

    INSURANCE = "insurance"
    FORCE_MAJEURE = "force_majeure"

    OBLIGATION = "obligation"
    OPERATIONAL = "operational"

    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    """Normalized risk severity."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskSignalType(str, Enum):
    """Source of a detected risk signal."""

    MODEL = "model"
    KEYWORD = "keyword"
    PATTERN = "pattern"
    RULE = "rule"
    HYBRID = "hybrid"


# ======================================================================
# Data structures
# ======================================================================


@dataclass(slots=True)
class RiskSignal:
    """Evidence supporting a risk classification."""

    category: RiskCategory
    signal_type: RiskSignalType

    description: str

    score: float
    confidence: float

    matched_text: Optional[str] = None

    start: Optional[int] = None
    end: Optional[int] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class RiskClassification:
    """Risk classification for one category."""

    category: RiskCategory

    score: float
    confidence: float

    level: RiskLevel

    signals: list[RiskSignal] = field(
        default_factory=list
    )

    rationale: Optional[str] = None


@dataclass(slots=True)
class RiskClassificationResult:
    """Complete risk-classification response."""

    text: str

    overall_score: float
    overall_confidence: float
    overall_level: RiskLevel

    classifications: list[
        RiskClassification
    ]

    signals: list[RiskSignal]

    model_name: str
    model_version: str

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def has_risk(self) -> bool:
        return self.overall_level != RiskLevel.NONE

    def by_category(
        self,
        category: RiskCategory,
    ) -> Optional[RiskClassification]:
        for classification in self.classifications:
            if classification.category == category:
                return classification

        return None

    def high_risk_categories(
        self,
    ) -> list[RiskClassification]:
        return [
            classification
            for classification in self.classifications
            if classification.level
            in {
                RiskLevel.HIGH,
                RiskLevel.CRITICAL,
            }
        ]


# ======================================================================
# Risk classifier
# ======================================================================


class RiskClassifier:
    """
    Legal/contractual risk classifier.

    The model identifies potential risk signals. It does NOT decide
    whether a contract is legally valid, enforceable, compliant, or
    unlawful.

    A downstream human-review workflow should make consequential
    decisions.
    """

    DEFAULT_LABEL_ALIASES: dict[
        str,
        RiskCategory,
    ] = {
        "LIABILITY": RiskCategory.LIABILITY,
        "INDEMNITY": RiskCategory.INDEMNIFICATION,
        "INDEMNIFICATION": RiskCategory.INDEMNIFICATION,
        "TERMINATION": RiskCategory.TERMINATION,
        "RENEWAL": RiskCategory.RENEWAL,
        "PAYMENT": RiskCategory.PAYMENT,
        "PRICING": RiskCategory.PRICING,

        "DATA_PROTECTION": RiskCategory.DATA_PROTECTION,
        "DATA_PRIVACY": RiskCategory.DATA_PROTECTION,
        "PRIVACY": RiskCategory.PRIVACY,
        "SECURITY": RiskCategory.SECURITY,

        "IP": RiskCategory.INTELLECTUAL_PROPERTY,
        "INTELLECTUAL_PROPERTY": (
            RiskCategory.INTELLECTUAL_PROPERTY
        ),

        "CONFIDENTIALITY": (
            RiskCategory.CONFIDENTIALITY
        ),

        "COMPLIANCE": RiskCategory.COMPLIANCE,
        "REGULATORY": RiskCategory.REGULATORY,

        "GOVERNING_LAW": RiskCategory.GOVERNING_LAW,
        "DISPUTE": RiskCategory.DISPUTE_RESOLUTION,
        "DISPUTE_RESOLUTION": (
            RiskCategory.DISPUTE_RESOLUTION
        ),

        "WARRANTY": RiskCategory.WARRANTY,
        "SLA": RiskCategory.SERVICE_LEVEL,
        "SERVICE_LEVEL": RiskCategory.SERVICE_LEVEL,

        "DELIVERY": RiskCategory.DELIVERY,
        "ASSIGNMENT": RiskCategory.ASSIGNMENT,
        "SUBCONTRACTING": RiskCategory.SUBCONTRACTING,

        "NON_COMPETE": RiskCategory.NON_COMPETE,
        "NON_SOLICITATION": (
            RiskCategory.NON_SOLICITATION
        ),

        "AUDIT": RiskCategory.AUDIT,
        "RECORDS": RiskCategory.RECORDS,
        "INSURANCE": RiskCategory.INSURANCE,
        "FORCE_MAJEURE": RiskCategory.FORCE_MAJEURE,

        "OBLIGATION": RiskCategory.OBLIGATION,
        "OPERATIONAL": RiskCategory.OPERATIONAL,
    }

    KEYWORDS: dict[
        RiskCategory,
        tuple[str, ...],
    ] = {
        RiskCategory.LIABILITY: (
            "liability",
            "liable",
            "damages",
            "losses",
            "limitation of liability",
            "consequential damages",
        ),

        RiskCategory.INDEMNIFICATION: (
            "indemnify",
            "indemnification",
            "hold harmless",
            "defend and indemnify",
        ),

        RiskCategory.TERMINATION: (
            "termination",
            "terminate",
            "termination for convenience",
            "termination for cause",
        ),

        RiskCategory.RENEWAL: (
            "renewal",
            "automatically renew",
            "auto-renew",
            "renewal term",
        ),

        RiskCategory.PAYMENT: (
            "payment",
            "invoice",
            "late payment",
            "payment terms",
            "net 30",
            "net 60",
        ),

        RiskCategory.PRICING: (
            "price",
            "pricing",
            "fee",
            "fees",
            "increase",
            "price adjustment",
        ),

        RiskCategory.DATA_PROTECTION: (
            "personal data",
            "data protection",
            "data processor",
            "data controller",
            "processing personal data",
            "gdpr",
        ),

        RiskCategory.PRIVACY: (
            "privacy",
            "privacy policy",
            "personal information",
            "personal data",
        ),

        RiskCategory.SECURITY: (
            "security",
            "information security",
            "security incident",
            "cybersecurity",
            "data breach",
        ),

        RiskCategory.INTELLECTUAL_PROPERTY: (
            "intellectual property",
            "ip rights",
            "copyright",
            "patent",
            "trademark",
            "ownership",
            "work product",
        ),

        RiskCategory.CONFIDENTIALITY: (
            "confidential",
            "confidentiality",
            "non-disclosure",
            "nda",
            "trade secret",
        ),

        RiskCategory.COMPLIANCE: (
            "compliance",
            "comply",
            "applicable law",
            "legal requirement",
        ),

        RiskCategory.REGULATORY: (
            "regulation",
            "regulatory",
            "regulator",
            "regulatory authority",
        ),

        RiskCategory.GOVERNING_LAW: (
            "governing law",
            "laws of",
            "applicable law",
        ),

        RiskCategory.DISPUTE_RESOLUTION: (
            "dispute",
            "arbitration",
            "mediation",
            "jurisdiction",
            "court",
        ),

        RiskCategory.WARRANTY: (
            "warranty",
            "warranties",
            "warrants",
            "merchantability",
        ),

        RiskCategory.SERVICE_LEVEL: (
            "service level",
            "sla",
            "uptime",
            "availability",
            "service credit",
        ),

        RiskCategory.DELIVERY: (
            "delivery",
            "deliverables",
            "delivery date",
            "acceptance",
        ),

        RiskCategory.ASSIGNMENT: (
            "assignment",
            "assign",
            "transfer",
        ),

        RiskCategory.SUBCONTRACTING: (
            "subcontract",
            "subcontractor",
            "third party provider",
        ),

        RiskCategory.NON_COMPETE: (
            "non-compete",
            "non compete",
            "competition",
            "competitive activity",
        ),

        RiskCategory.NON_SOLICITATION: (
            "non-solicit",
            "non solicit",
            "solicit employees",
            "solicit customers",
        ),

        RiskCategory.AUDIT: (
            "audit",
            "audit rights",
            "audit access",
        ),

        RiskCategory.RECORDS: (
            "records",
            "record retention",
            "retain records",
        ),

        RiskCategory.INSURANCE: (
            "insurance",
            "insured",
            "coverage",
            "certificate of insurance",
        ),

        RiskCategory.FORCE_MAJEURE: (
            "force majeure",
            "act of god",
            "unforeseeable event",
        ),

        RiskCategory.OBLIGATION: (
            "shall",
            "must",
            "required to",
            "obligation",
        ),
    }

    # Signals that can increase severity.
    HIGH_RISK_PATTERNS: tuple[str, ...] = (
        "unlimited liability",
        "uncapped liability",
        "unlimited indemnity",
        "without limitation",
        "sole discretion",
        "automatic renewal",
        "irrevocable",
        "waive all claims",
        "waives any right",
        "without notice",
        "immediate termination",
        "exclusive remedy",
        "unrestricted access",
    )

    MEDIUM_RISK_PATTERNS: tuple[str, ...] = (
        "may increase",
        "at any time",
        "upon written notice",
        "reasonable efforts",
        "commercially reasonable",
        "best efforts",
        "subject to change",
        "non-refundable",
        "late fee",
    )

    def __init__(
        self,
        model_name: str = "legal-risk-classifier",
        model_version: str = "v1",
        model: Any = None,
        tokenizer: Any = None,
        device: Optional[str] = None,
        min_confidence: float = 0.50,
        enable_fallback: bool = True,
        label_aliases: Optional[
            dict[str, RiskCategory]
        ] = None,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version

        self._model = model
        self._tokenizer = tokenizer

        self.device = (
            device
            or self._detect_device()
        )

        self.min_confidence = max(
            0.0,
            min(1.0, min_confidence),
        )

        self.enable_fallback = enable_fallback

        self.label_aliases = dict(
            self.DEFAULT_LABEL_ALIASES
        )

        if label_aliases:
            self.label_aliases.update(
                label_aliases
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        text: str,
        *,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> RiskClassificationResult:
        """
        Classify potential risk signals in text.
        """

        self._validate_text(text)

        if not text.strip():
            return self._empty_result(
                text=text,
                metadata=metadata,
            )

        model_classifications: list[
            RiskClassification
        ] = []

        if self._model_available():
            try:
                model_classifications = (
                    self._classify_with_model(
                        text
                    )
                )
            except Exception:
                logger.exception(
                    "Risk model inference failed."
                )

                if not self.enable_fallback:
                    raise

        fallback_signals: list[
            RiskSignal
        ] = []

        if self.enable_fallback:
            fallback_signals = (
                self._detect_rule_signals(text)
            )

        classifications = (
            self._merge_classifications(
                model_classifications,
                fallback_signals,
            )
        )

        overall_score = (
            self._calculate_overall_score(
                classifications
            )
        )

        overall_confidence = (
            self._calculate_overall_confidence(
                classifications
            )
        )

        overall_level = self._score_to_level(
            overall_score
        )

        all_signals = [
            signal
            for classification in classifications
            for signal in classification.signals
        ]

        return RiskClassificationResult(
            text=text,
            overall_score=overall_score,
            overall_confidence=overall_confidence,
            overall_level=overall_level,
            classifications=classifications,
            signals=all_signals,
            model_name=self.model_name,
            model_version=self.model_version,
            metadata={
                **(metadata or {}),
                "device": self.device,
                "fallback_enabled": (
                    self.enable_fallback
                ),
            },
        )

    def classify_batch(
        self,
        texts: Sequence[str],
    ) -> list[RiskClassificationResult]:
        """Classify multiple text segments."""

        return [
            self.classify(text)
            for text in texts
        ]

    def classify_with_model_output(
        self,
        text: str,
        model_output: Sequence[
            dict[str, Any]
        ],
    ) -> RiskClassificationResult:
        """
        Build a classification from externally supplied model output.

        Useful for testing and model-service integration.
        """

        self._validate_text(text)

        classifications = (
            self._parse_model_output(
                text,
                model_output,
            )
        )

        fallback_signals = (
            self._detect_rule_signals(text)
            if self.enable_fallback
            else []
        )

        classifications = (
            self._merge_classifications(
                classifications,
                fallback_signals,
            )
        )

        score = self._calculate_overall_score(
            classifications
        )

        confidence = (
            self._calculate_overall_confidence(
                classifications
            )
        )

        return RiskClassificationResult(
            text=text,
            overall_score=score,
            overall_confidence=confidence,
            overall_level=self._score_to_level(
                score
            ),
            classifications=classifications,
            signals=[
                signal
                for classification in classifications
                for signal in classification.signals
            ],
            model_name=self.model_name,
            model_version=self.model_version,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "device": self.device,
            "min_confidence": self.min_confidence,
            "fallback_enabled": self.enable_fallback,
            "categories": [
                category.value
                for category in RiskCategory
            ],
        }

    # ------------------------------------------------------------------
    # Model inference
    # ------------------------------------------------------------------

    def _model_available(self) -> bool:
        return (
            self._model is not None
            and self._tokenizer is not None
        )

    def _classify_with_model(
        self,
        text: str,
    ) -> list[RiskClassification]:
        """
        Run a generic Hugging Face sequence-classification model.

        Expected model output:
            logits shaped [1, number_of_labels].
        """

        if not self._model_available():
            return []

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for risk classification."
            ) from exc

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        self._model.eval()

        with torch.no_grad():
            output = self._model(
                **encoded
            )

        logits = output.logits[0]

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        classifications: list[
            RiskClassification
        ] = []

        id2label = getattr(
            self._model.config,
            "id2label",
            {},
        )

        for index, probability in enumerate(
            probabilities
        ):
            confidence = float(
                probability.item()
            )

            if confidence < self.min_confidence:
                continue

            label = id2label.get(
                index,
                str(index),
            )

            category = self._map_label(
                str(label)
            )

            if category == RiskCategory.UNKNOWN:
                continue

            score = self._clamp(
                confidence
            )

            signal = RiskSignal(
                category=category,
                signal_type=RiskSignalType.MODEL,
                description=(
                    "Model identified a potential "
                    f"{category.value} risk signal."
                ),
                score=score,
                confidence=confidence,
                metadata={
                    "label": label,
                    "model": self.model_name,
                },
            )

            classifications.append(
                RiskClassification(
                    category=category,
                    score=score,
                    confidence=confidence,
                    level=self._score_to_level(
                        score
                    ),
                    signals=[signal],
                    rationale=(
                        "Model-based risk signal; "
                        "requires evidence review."
                    ),
                )
            )

        return classifications

    # ------------------------------------------------------------------
    # Model-output parsing
    # ------------------------------------------------------------------

    def _parse_model_output(
        self,
        text: str,
        model_output: Sequence[
            dict[str, Any]
        ],
    ) -> list[RiskClassification]:
        classifications: list[
            RiskClassification
        ] = []

        for item in model_output:
            if not isinstance(item, dict):
                continue

            label = str(
                item.get(
                    "label",
                    "UNKNOWN",
                )
            )

            category = self._map_label(
                label
            )

            if category == RiskCategory.UNKNOWN:
                continue

            score = self._clamp(
                item.get(
                    "score",
                    item.get(
                        "confidence",
                        0.0,
                    ),
                )
            )

            confidence = self._clamp(
                item.get(
                    "confidence",
                    score,
                )
            )

            if confidence < self.min_confidence:
                continue

            matched_text = item.get(
                "matched_text"
            )

            signal = RiskSignal(
                category=category,
                signal_type=RiskSignalType.MODEL,
                description=str(
                    item.get(
                        "description",
                        "Model-generated risk signal.",
                    )
                ),
                score=score,
                confidence=confidence,
                matched_text=matched_text,
                start=item.get("start"),
                end=item.get("end"),
                metadata=dict(
                    item.get(
                        "metadata",
                        {},
                    )
                ),
            )

            classifications.append(
                RiskClassification(
                    category=category,
                    score=score,
                    confidence=confidence,
                    level=self._score_to_level(
                        score
                    ),
                    signals=[signal],
                    rationale=(
                        "Model-generated risk signal; "
                        "requires evidence review."
                    ),
                )
            )

        return classifications

    # ------------------------------------------------------------------
    # Deterministic signals
    # ------------------------------------------------------------------

    def _detect_rule_signals(
        self,
        text: str,
    ) -> list[RiskSignal]:
        normalized = text.lower()

        signals: list[RiskSignal] = []

        for category, keywords in self.KEYWORDS.items():
            for keyword in keywords:
                start = normalized.find(
                    keyword.lower()
                )

                if start == -1:
                    continue

                end = start + len(keyword)

                signals.append(
                    RiskSignal(
                        category=category,
                        signal_type=(
                            RiskSignalType.KEYWORD
                        ),
                        description=(
                            f"Detected {category.value} "
                            "related contractual language."
                        ),
                        score=0.45,
                        confidence=0.75,
                        matched_text=text[
                            start:end
                        ],
                        start=start,
                        end=end,
                        metadata={
                            "keyword": keyword,
                        },
                    )
                )

        for pattern in self.HIGH_RISK_PATTERNS:
            for match in re.finditer(
                re.escape(pattern),
                normalized,
            ):
                category = (
                    self._infer_category_from_context(
                        normalized,
                        match.start(),
                        match.end(),
                    )
                )

                signals.append(
                    RiskSignal(
                        category=category,
                        signal_type=(
                            RiskSignalType.PATTERN
                        ),
                        description=(
                            "High-risk contractual "
                            "language pattern detected."
                        ),
                        score=0.90,
                        confidence=0.92,
                        matched_text=text[
                            match.start():match.end()
                        ],
                        start=match.start(),
                        end=match.end(),
                        metadata={
                            "pattern": pattern,
                            "severity_hint": "high",
                        },
                    )
                )

        for pattern in self.MEDIUM_RISK_PATTERNS:
            for match in re.finditer(
                re.escape(pattern),
                normalized,
            ):
                category = (
                    self._infer_category_from_context(
                        normalized,
                        match.start(),
                        match.end(),
                    )
                )

                signals.append(
                    RiskSignal(
                        category=category,
                        signal_type=(
                            RiskSignalType.PATTERN
                        ),
                        description=(
                            "Potentially significant "
                            "contractual language detected."
                        ),
                        score=0.65,
                        confidence=0.80,
                        matched_text=text[
                            match.start():match.end()
                        ],
                        start=match.start(),
                        end=match.end(),
                        metadata={
                            "pattern": pattern,
                            "severity_hint": "medium",
                        },
                    )
                )

        return self._deduplicate_signals(
            signals
        )

    # ------------------------------------------------------------------
    # Classification merging
    # ------------------------------------------------------------------

    def _merge_classifications(
        self,
        model_classifications: Sequence[
            RiskClassification
        ],
        fallback_signals: Sequence[
            RiskSignal
        ],
    ) -> list[RiskClassification]:
        grouped: dict[
            RiskCategory,
            list[RiskSignal],
        ] = {}

        for classification in model_classifications:
            grouped.setdefault(
                classification.category,
                [],
            ).extend(
                classification.signals
            )

        for signal in fallback_signals:
            grouped.setdefault(
                signal.category,
                [],
            ).append(signal)

        results: list[
            RiskClassification
        ] = []

        for category, signals in grouped.items():
            if not signals:
                continue

            score = self._aggregate_signal_score(
                signals
            )

            confidence = self._aggregate_confidence(
                signals
            )

            signal_types = {
                signal.signal_type
                for signal in signals
            }

            if len(signal_types) > 1:
                for signal in signals:
                    if signal.signal_type in {
                        RiskSignalType.MODEL,
                        RiskSignalType.KEYWORD,
                        RiskSignalType.PATTERN,
                    }:
                        signal.metadata[
                            "evidence_mode"
                        ] = "hybrid"

            results.append(
                RiskClassification(
                    category=category,
                    score=score,
                    confidence=confidence,
                    level=self._score_to_level(
                        score
                    ),
                    signals=list(signals),
                    rationale=self._build_rationale(
                        category,
                        signals,
                    ),
                )
            )

        return sorted(
            results,
            key=lambda item: (
                -item.score,
                -item.confidence,
                item.category.value,
            ),
        )

    @staticmethod
    def _aggregate_signal_score(
        signals: Sequence[RiskSignal],
    ) -> float:
        """
        Aggregate evidence without allowing repeated keywords
        to inflate the score indefinitely.
        """

        if not signals:
            return 0.0

        strongest = max(
            signal.score
            for signal in signals
        )

        supporting = [
            signal.score
            for signal in signals
            if signal.score != strongest
        ]

        if not supporting:
            return max(
                0.0,
                min(1.0, strongest),
            )

        support_bonus = min(
            0.25,
            sum(supporting) * 0.10,
        )

        return max(
            0.0,
            min(
                1.0,
                strongest + support_bonus,
            ),
        )

    @staticmethod
    def _aggregate_confidence(
        signals: Sequence[RiskSignal],
    ) -> float:
        if not signals:
            return 0.0

        strongest = max(
            signal.confidence
            for signal in signals
        )

        return max(
            0.0,
            min(1.0, strongest),
        )

    @staticmethod
    def _build_rationale(
        category: RiskCategory,
        signals: Sequence[RiskSignal],
    ) -> str:
        descriptions = [
            signal.description
            for signal in signals[:3]
        ]

        return (
            f"Potential {category.value} risk identified. "
            "Evidence: "
            + " ".join(descriptions)
            + " Human/legal review may be required."
        )

    # ------------------------------------------------------------------
    # Risk scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_overall_score(
        classifications: Sequence[
            RiskClassification
        ],
    ) -> float:
        if not classifications:
            return 0.0

        scores = sorted(
            (
                classification.score
                for classification in classifications
            ),
            reverse=True,
        )

        strongest = scores[0]

        # Smaller contribution from additional categories.
        secondary = sum(
            score * 0.15
            for score in scores[1:4]
        )

        return max(
            0.0,
            min(
                1.0,
                strongest + secondary,
            ),
        )

    @staticmethod
    def _calculate_overall_confidence(
        classifications: Sequence[
            RiskClassification
        ],
    ) -> float:
        if not classifications:
            return 0.0

        return max(
            classification.confidence
            for classification in classifications
        )

    @staticmethod
    def _score_to_level(
        score: float,
    ) -> RiskLevel:
        score = max(
            0.0,
            min(1.0, score),
        )

        if score < 0.25:
            return RiskLevel.NONE

        if score < 0.50:
            return RiskLevel.LOW

        if score < 0.75:
            return RiskLevel.MEDIUM

        if score < 0.90:
            return RiskLevel.HIGH

        return RiskLevel.CRITICAL

    # ------------------------------------------------------------------
    # Label handling
    # ------------------------------------------------------------------

    def _map_label(
        self,
        label: str,
    ) -> RiskCategory:
        normalized = (
            label
            .upper()
            .strip()
            .replace("-", "_")
            .replace(" ", "_")
        )

        normalized = re.sub(
            r"^[BI]_",
            "",
            normalized,
        )

        return self.label_aliases.get(
            normalized,
            RiskCategory.UNKNOWN,
        )

    # ------------------------------------------------------------------
    # Context inference
    # ------------------------------------------------------------------

    def _infer_category_from_context(
        self,
        text: str,
        start: int,
        end: int,
    ) -> RiskCategory:
        window_start = max(
            0,
            start - 100,
        )

        window_end = min(
            len(text),
            end + 100,
        )

        context = text[
            window_start:window_end
        ]

        # Prioritized context mapping.
        priorities = [
            (
                RiskCategory.LIABILITY,
                (
                    "liability",
                    "damages",
                    "liable",
                ),
            ),
            (
                RiskCategory.INDEMNIFICATION,
                (
                    "indemn",
                    "hold harmless",
                ),
            ),
            (
                RiskCategory.TERMINATION,
                (
                    "termination",
                    "terminate",
                ),
            ),
            (
                RiskCategory.RENEWAL,
                (
                    "renew",
                ),
            ),
            (
                RiskCategory.DATA_PROTECTION,
                (
                    "personal data",
                    "gdpr",
                    "data protection",
                ),
            ),
            (
                RiskCategory.SECURITY,
                (
                    "security",
                    "breach",
                    "cyber",
                ),
            ),
            (
                RiskCategory.PAYMENT,
                (
                    "payment",
                    "invoice",
                    "fee",
                ),
            ),
        ]

        for category, keywords in priorities:
            if any(
                keyword in context
                for keyword in keywords
            ):
                return category

        return RiskCategory.UNKNOWN

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_signals(
        signals: Sequence[RiskSignal],
    ) -> list[RiskSignal]:
        unique: list[RiskSignal] = []
        seen: set[
            tuple[
                RiskCategory,
                Optional[int],
                Optional[int],
            ]
        ] = set()

        for signal in signals:
            key = (
                signal.category,
                signal.start,
                signal.end,
            )

            if key in seen:
                continue

            seen.add(key)
            unique.append(signal)

        return unique

    @staticmethod
    def _clamp(
        value: Any,
    ) -> float:
        try:
            numeric = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(1.0, numeric),
        )

    @staticmethod
    def _validate_text(
        text: str,
    ) -> None:
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string."
            )

        if len(text) > 2_000_000:
            raise ValueError(
                "text exceeds the maximum supported size."
            )

    def _empty_result(
        self,
        *,
        text: str,
        metadata: Optional[
            dict[str, Any]
        ],
    ) -> RiskClassificationResult:
        return RiskClassificationResult(
            text=text,
            overall_score=0.0,
            overall_confidence=0.0,
            overall_level=RiskLevel.NONE,
            classifications=[],
            signals=[],
            model_name=self.model_name,
            model_version=self.model_version,
            metadata=metadata or {},
        )

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                return "mps"

        except ImportError:
            pass

        return "cpu"


__all__ = [
    "RiskCategory",
    "RiskLevel",
    "RiskSignalType",
    "RiskSignal",
    "RiskClassification",
    "RiskClassificationResult",
    "RiskClassifier",
]