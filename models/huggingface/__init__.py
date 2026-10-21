"""
Legal Clause Classifier
=======================

Classifies legal contract clauses into predefined legal categories.

Example categories
------------------
- confidentiality
- termination
- payment
- liability
- indemnification
- intellectual_property
- data_protection
- governing_law
- dispute_resolution
- non_compete
- non_solicitation
- warranty
- limitation_of_liability
- force_majeure
- renewal
- assignment
- audit
- insurance
- compliance
- miscellaneous

Responsibilities
-----------------
- Load a Hugging Face sequence-classification model.
- Classify individual clauses.
- Support batch classification.
- Return confidence scores.
- Return top-k predictions.
- Normalize text safely.
- Support CPU/GPU inference.
- Remain independent from FastAPI and database code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# ----------------------------------------------------------------------
# Clause taxonomy
# ----------------------------------------------------------------------

DEFAULT_CLAUSE_LABELS: Tuple[str, ...] = (
    "confidentiality",
    "termination",
    "payment",
    "indemnification",
    "liability",
    "limitation_of_liability",
    "intellectual_property",
    "data_protection",
    "privacy",
    "governing_law",
    "dispute_resolution",
    "arbitration",
    "jurisdiction",
    "non_compete",
    "non_solicitation",
    "warranty",
    "force_majeure",
    "renewal",
    "assignment",
    "audit",
    "insurance",
    "compliance",
    "security",
    "service_levels",
    "delivery",
    "acceptance",
    "pricing",
    "tax",
    "notices",
    "records_retention",
    "miscellaneous",
)


# ----------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------

class ClauseClassifierError(Exception):
    """Base exception for clause-classification failures."""


class ModelNotLoadedError(ClauseClassifierError):
    """Raised when inference is attempted before loading the model."""


class InvalidClauseError(ClauseClassifierError):
    """Raised when clause text is empty or invalid."""


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class ClausePrediction:
    """
    One predicted clause category.
    """

    label: str
    confidence: float
    rank: int


@dataclass(frozen=True)
class ClauseClassification:
    """
    Complete classification result.
    """

    text: str
    label: str
    confidence: float
    predictions: Sequence[ClausePrediction]

    model_name: str
    model_version: Optional[str] = None


@dataclass(frozen=True)
class ClauseClassifierConfig:
    """
    Configuration for the classifier.
    """

    model_name: str

    labels: Sequence[str] = DEFAULT_CLAUSE_LABELS

    max_length: int = 512

    device: Optional[str] = None

    top_k: int = 5

    confidence_threshold: float = 0.50


# ----------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------

class ClauseClassifier:
    """
    Hugging Face based legal clause classifier.

    Expected model architecture:
        AutoModelForSequenceClassification

    The model should have been fine-tuned on a legal clause dataset.
    """

    def __init__(
        self,
        config: ClauseClassifierConfig,
    ) -> None:

        self.config = config

        self.tokenizer = None
        self.model = None

        self.device = self._resolve_device(
            config.device
        )

        self.label_to_id: Dict[str, int] = {
            label: index
            for index, label in enumerate(config.labels)
        }

        self.id_to_label: Dict[int, str] = {
            index: label
            for index, label in enumerate(config.labels)
        }

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(
        requested_device: Optional[str],
    ) -> torch.device:

        if requested_device:
            return torch.device(requested_device)

        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load tokenizer and classification model.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name
        )

        self.model.to(self.device)
        self.model.eval()

    @property
    def is_loaded(self) -> bool:
        return (
            self.model is not None
            and self.tokenizer is not None
        )

    # ------------------------------------------------------------------
    # Text normalization
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize clause text without destroying legal meaning.
        """

        if not isinstance(text, str):
            raise InvalidClauseError(
                "Clause must be a string."
            )

        text = text.replace("\x00", " ")

        # Preserve paragraph semantics while removing
        # excessive whitespace.
        text = " ".join(
            text.split()
        )

        if not text:
            raise InvalidClauseError(
                "Clause text cannot be empty."
            )

        return text.strip()

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        text: str,
        top_k: Optional[int] = None,
    ) -> ClauseClassification:

        if not self.is_loaded:
            raise ModelNotLoadedError(
                "Clause classifier has not been loaded."
            )

        text = self.normalize_text(text)

        k = top_k or self.config.top_k

        k = max(
            1,
            min(
                k,
                len(self.config.labels),
            ),
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        outputs = self.model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        values, indices = torch.topk(
            probabilities,
            k=k,
        )

        predictions: List[ClausePrediction] = []

        for rank, (
            probability,
            index,
        ) in enumerate(
            zip(values, indices),
            start=1,
        ):
            label = self._label_for_index(
                int(index)
            )

            predictions.append(
                ClausePrediction(
                    label=label,
                    confidence=float(
                        probability.item()
                    ),
                    rank=rank,
                )
            )

        best = predictions[0]

        return ClauseClassification(
            text=text,
            label=best.label,
            confidence=best.confidence,
            predictions=predictions,
            model_name=self.config.model_name,
        )

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict_batch(
        self,
        texts: Sequence[str],
        top_k: Optional[int] = None,
    ) -> List[ClauseClassification]:

        if not self.is_loaded:
            raise ModelNotLoadedError(
                "Clause classifier has not been loaded."
            )

        if not texts:
            return []

        normalized = [
            self.normalize_text(text)
            for text in texts
        ]

        k = top_k or self.config.top_k

        k = max(
            1,
            min(
                k,
                len(self.config.labels),
            ),
        )

        encoded = self.tokenizer(
            normalized,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        outputs = self.model(
            **encoded
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )

        results: List[ClauseClassification] = []

        for row_index, text in enumerate(
            normalized
        ):
            row = probabilities[row_index]

            values, indices = torch.topk(
                row,
                k=k,
            )

            predictions = [
                ClausePrediction(
                    label=self._label_for_index(
                        int(index)
                    ),
                    confidence=float(
                        probability.item()
                    ),
                    rank=rank,
                )
                for rank, (
                    probability,
                    index,
                ) in enumerate(
                    zip(values, indices),
                    start=1,
                )
            ]

            best = predictions[0]

            results.append(
                ClauseClassification(
                    text=text,
                    label=best.label,
                    confidence=best.confidence,
                    predictions=predictions,
                    model_name=self.config.model_name,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Confidence handling
    # ------------------------------------------------------------------

    def is_confident(
        self,
        result: ClauseClassification,
    ) -> bool:

        return (
            result.confidence
            >= self.config.confidence_threshold
        )

    def classify_with_review_flag(
        self,
        text: str,
    ) -> Dict[str, object]:

        result = self.predict(text)

        return {
            "label": result.label,
            "confidence": result.confidence,
            "requires_review": not self.is_confident(
                result
            ),
            "predictions": [
                {
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "rank": prediction.rank,
                }
                for prediction in result.predictions
            ],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _label_for_index(
        self,
        index: int,
    ) -> str:

        if index in self.id_to_label:
            return self.id_to_label[index]

        # Support models whose config contains
        # id2label metadata.
        if self.model is not None:
            model_config = getattr(
                self.model,
                "config",
                None,
            )

            if model_config is not None:
                id2label = getattr(
                    model_config,
                    "id2label",
                    {},
                )

                if index in id2label:
                    return id2label[index]

                string_index = str(index)

                if string_index in id2label:
                    return id2label[string_index]

        return f"label_{index}"

    # ------------------------------------------------------------------
    # Model information
    # ------------------------------------------------------------------

    def model_info(self) -> Dict[str, object]:

        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "max_length": self.config.max_length,
            "labels": list(self.config.labels),
            "num_labels": len(self.config.labels),
            "loaded": self.is_loaded,
        }


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def create_clause_classifier(
    model_name: str,
    *,
    device: Optional[str] = None,
    max_length: int = 512,
    top_k: int = 5,
    confidence_threshold: float = 0.50,
    labels: Sequence[str] = DEFAULT_CLAUSE_LABELS,
) -> ClauseClassifier:

    config = ClauseClassifierConfig(
        model_name=model_name,
        labels=labels,
        max_length=max_length,
        device=device,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )

    classifier = ClauseClassifier(
        config=config
    )

    classifier.load()

    return classifier