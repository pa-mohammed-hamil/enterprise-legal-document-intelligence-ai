# backend/models/huggingface/clause_classifier.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
import re


class ClauseType(str, Enum):
    CONFIDENTIALITY = "confidentiality"
    TERMINATION = "termination"
    RENEWAL = "renewal"
    PAYMENT = "payment"
    PRICING = "pricing"
    LIABILITY = "liability"
    INDEMNIFICATION = "indemnification"
    WARRANTY = "warranty"
    REPRESENTATION = "representation"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DATA_PROTECTION = "data_protection"
    PRIVACY = "privacy"
    SECURITY = "security"
    AUDIT = "audit"
    COMPLIANCE = "compliance"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    FORCE_MAJEURE = "force_majeure"
    ASSIGNMENT = "assignment"
    INSURANCE = "insurance"
    SERVICE_LEVEL = "service_level"
    DELIVERY = "delivery"
    ACCEPTANCE = "acceptance"
    NON_COMPETE = "non_compete"
    NON_SOLICITATION = "non_solicitation"
    NOTICE = "notice"
    CHANGE_CONTROL = "change_control"
    SUBCONTRACTING = "subcontracting"
    RECORDS = "records"
    DEFINITIONS = "definitions"
    OTHER = "other"


@dataclass(frozen=True)
class ClauseClassification:
    clause_type: ClauseType
    confidence: float
    model_name: str
    model_version: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClauseClassificationResult:
    text: str
    primary: ClauseClassification
    alternatives: List[ClauseClassification] = field(default_factory=list)


class ClauseClassifier:
    """
    Legal clause classifier.

    The classifier identifies the likely semantic clause category.
    It does NOT determine:
        - legal validity
        - enforceability
        - breach
        - legal advice
        - contractual risk

    Those concerns belong to downstream legal/risk services.
    """

    LABEL_ALIASES: Dict[str, ClauseType] = {
        "confidentiality": ClauseType.CONFIDENTIALITY,
        "nda": ClauseType.CONFIDENTIALITY,
        "termination": ClauseType.TERMINATION,
        "renewal": ClauseType.RENEWAL,
        "payment": ClauseType.PAYMENT,
        "pricing": ClauseType.PRICING,
        "liability": ClauseType.LIABILITY,
        "limitation_of_liability": ClauseType.LIABILITY,
        "indemnity": ClauseType.INDEMNIFICATION,
        "indemnification": ClauseType.INDEMNIFICATION,
        "warranty": ClauseType.WARRANTY,
        "representations": ClauseType.REPRESENTATION,
        "representation": ClauseType.REPRESENTATION,
        "ip": ClauseType.INTELLECTUAL_PROPERTY,
        "intellectual_property": ClauseType.INTELLECTUAL_PROPERTY,
        "data_protection": ClauseType.DATA_PROTECTION,
        "privacy": ClauseType.PRIVACY,
        "security": ClauseType.SECURITY,
        "audit": ClauseType.AUDIT,
        "compliance": ClauseType.COMPLIANCE,
        "governing_law": ClauseType.GOVERNING_LAW,
        "dispute_resolution": ClauseType.DISPUTE_RESOLUTION,
        "arbitration": ClauseType.DISPUTE_RESOLUTION,
        "force_majeure": ClauseType.FORCE_MAJEURE,
        "assignment": ClauseType.ASSIGNMENT,
        "insurance": ClauseType.INSURANCE,
        "sla": ClauseType.SERVICE_LEVEL,
        "service_level": ClauseType.SERVICE_LEVEL,
        "delivery": ClauseType.DELIVERY,
        "acceptance": ClauseType.ACCEPTANCE,
        "non_compete": ClauseType.NON_COMPETE,
        "non_solicitation": ClauseType.NON_SOLICITATION,
        "notice": ClauseType.NOTICE,
        "change_control": ClauseType.CHANGE_CONTROL,
        "subcontracting": ClauseType.SUBCONTRACTING,
        "records": ClauseType.RECORDS,
        "definitions": ClauseType.DEFINITIONS,
    }

    KEYWORDS: Dict[ClauseType, Sequence[str]] = {
        ClauseType.CONFIDENTIALITY: (
            "confidential", "confidentiality", "non-disclosure",
            "proprietary information", "trade secret",
        ),
        ClauseType.TERMINATION: (
            "terminate", "termination", "terminates",
            "termination for cause", "termination for convenience",
        ),
        ClauseType.RENEWAL: (
            "renew", "renewal", "automatic renewal",
            "renewal term", "evergreen",
        ),
        ClauseType.PAYMENT: (
            "payment", "invoice", "invoicing", "payable",
            "payment terms", "late payment",
        ),
        ClauseType.PRICING: (
            "price", "pricing", "fee", "fees",
            "rate card", "charges",
        ),
        ClauseType.LIABILITY: (
            "liability", "liable", "limitation of liability",
            "aggregate liability", "consequential damages",
        ),
        ClauseType.INDEMNIFICATION: (
            "indemnify", "indemnifies", "indemnification",
            "hold harmless",
        ),
        ClauseType.WARRANTY: (
            "warranty", "warranties", "warrants",
            "merchantability", "fitness for purpose",
        ),
        ClauseType.REPRESENTATION: (
            "represents", "representation", "representations",
            "covenants", "warrants",
        ),
        ClauseType.INTELLECTUAL_PROPERTY: (
            "intellectual property", "copyright", "patent",
            "trademark", "ip rights", "ownership of work product",
        ),
        ClauseType.DATA_PROTECTION: (
            "data protection", "personal data", "data processor",
            "data controller", "processing of personal data",
            "gdpr",
        ),
        ClauseType.PRIVACY: (
            "privacy", "privacy policy", "personal information",
            "consumer privacy",
        ),
        ClauseType.SECURITY: (
            "information security", "security controls",
            "security incident", "cybersecurity",
            "access controls",
        ),
        ClauseType.AUDIT: (
            "audit", "auditing", "audit rights",
            "inspection rights",
        ),
        ClauseType.COMPLIANCE: (
            "compliance", "applicable laws", "regulations",
            "regulatory", "anti-bribery", "anti-corruption",
        ),
        ClauseType.GOVERNING_LAW: (
            "governing law", "governed by the laws",
            "laws of", "applicable law",
        ),
        ClauseType.DISPUTE_RESOLUTION: (
            "dispute", "arbitration", "mediation",
            "dispute resolution", "jurisdiction",
        ),
        ClauseType.FORCE_MAJEURE: (
            "force majeure", "act of god",
            "natural disaster", "unavoidable event",
        ),
        ClauseType.ASSIGNMENT: (
            "assignment", "assign", "assigned",
            "transfer this agreement",
        ),
        ClauseType.INSURANCE: (
            "insurance", "insured", "insurance coverage",
            "certificate of insurance",
        ),
        ClauseType.SERVICE_LEVEL: (
            "service level", "sla", "uptime",
            "availability", "response time",
            "service credits",
        ),
        ClauseType.DELIVERY: (
            "delivery", "deliverables", "shipment",
            "delivery date", "delivery schedule",
        ),
        ClauseType.ACCEPTANCE: (
            "acceptance", "acceptance criteria",
            "accepted", "reject deliverables",
        ),
        ClauseType.NON_COMPETE: (
            "non-compete", "non compete",
            "competition", "competitive activities",
        ),
        ClauseType.NON_SOLICITATION: (
            "non-solicitation", "non solicitation",
            "solicit employees", "solicit customers",
        ),
        ClauseType.NOTICE: (
            "notice", "written notice",
            "notice address", "notice period",
        ),
        ClauseType.CHANGE_CONTROL: (
            "change control", "change request",
            "change order", "scope change",
        ),
        ClauseType.SUBCONTRACTING: (
            "subcontract", "subcontractor",
            "subcontracting", "third party provider",
        ),
        ClauseType.RECORDS: (
            "records", "record keeping",
            "books and records", "retention of records",
        ),
        ClauseType.DEFINITIONS: (
            "means", "defined terms",
            "definition", "for purposes of this agreement",
        ),
    }

    def __init__(
        self,
        model: Optional[Any] = None,
        *,
        model_name: str = "legal-clause-classifier",
        model_version: Optional[str] = None,
        confidence_threshold: float = 0.55,
        use_keyword_fallback: bool = True,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.use_keyword_fallback = use_keyword_fallback

    def classify(
        self,
        text: str,
        *,
        top_k: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClauseClassificationResult:
        """
        Classify one clause.

        If a trained model is supplied, model inference is preferred.
        Otherwise the controlled keyword fallback is used.
        """

        normalized = self._normalize(text)

        if not normalized:
            raise ValueError("Clause text cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        if self.model is not None:
            classifications = self._predict_model(
                normalized,
                top_k=top_k,
                metadata=metadata or {},
            )
        elif self.use_keyword_fallback:
            classifications = self._predict_keywords(
                normalized,
                top_k=top_k,
                metadata=metadata or {},
            )
        else:
            classifications = [
                self._create_classification(
                    ClauseType.OTHER,
                    0.0,
                    [],
                    metadata or {},
                )
            ]

        primary = classifications[0]

        return ClauseClassificationResult(
            text=text,
            primary=primary,
            alternatives=classifications[1:],
        )

    def classify_batch(
        self,
        texts: Sequence[str],
        *,
        top_k: int = 3,
    ) -> List[ClauseClassificationResult]:
        """
        Classify multiple clauses.
        """

        return [
            self.classify(text, top_k=top_k)
            for text in texts
        ]

    def classify_with_model_output(
        self,
        text: str,
        model_outputs: Sequence[Dict[str, Any]],
        *,
        top_k: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ClauseClassificationResult:
        """
        Convert generic Hugging Face-style classifier output into
        the application's controlled ClauseType taxonomy.

        Expected model output:

            [
                {"label": "CONFIDENTIALITY", "score": 0.94},
                {"label": "DATA_PROTECTION", "score": 0.03},
                ...
            ]
        """

        normalized = self._normalize(text)

        if not normalized:
            raise ValueError("Clause text cannot be empty")

        classifications: List[ClauseClassification] = []

        for output in model_outputs:
            label = str(output.get("label", "")).strip()
            score = self._clamp(
                float(output.get("score", 0.0))
            )

            clause_type = self._map_label(label)

            if clause_type is None:
                continue

            classifications.append(
                self._create_classification(
                    clause_type=clause_type,
                    confidence=score,
                    evidence=self._extract_evidence(
                        normalized,
                        clause_type,
                    ),
                    metadata=metadata or {},
                )
            )

        classifications.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        if not classifications:
            classifications = [
                self._create_classification(
                    ClauseType.OTHER,
                    0.0,
                    [],
                    metadata or {},
                )
            ]

        return ClauseClassificationResult(
            text=text,
            primary=classifications[0],
            alternatives=classifications[1:top_k],
        )

    def _predict_model(
        self,
        text: str,
        *,
        top_k: int,
        metadata: Dict[str, Any],
    ) -> List[ClauseClassification]:
        """
        Supports either:
            - a callable model
            - a Hugging Face pipeline-like object
        """

        raw_output = self.model(
            text,
            top_k=top_k,
        )

        if isinstance(raw_output, dict):
            raw_output = [raw_output]

        return self._convert_model_output(
            text=text,
            outputs=raw_output,
            metadata=metadata,
            top_k=top_k,
        )

    def _convert_model_output(
        self,
        *,
        text: str,
        outputs: Sequence[Dict[str, Any]],
        metadata: Dict[str, Any],
        top_k: int,
    ) -> List[ClauseClassification]:
        classifications: List[ClauseClassification] = []

        for output in outputs:
            label = str(output.get("label", ""))
            clause_type = self._map_label(label)

            if clause_type is None:
                continue

            score = self._clamp(
                float(output.get("score", 0.0))
            )

            evidence = self._extract_evidence(
                text,
                clause_type,
            )

            classifications.append(
                self._create_classification(
                    clause_type=clause_type,
                    confidence=score,
                    evidence=evidence,
                    metadata=metadata,
                )
            )

        classifications.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        if not classifications:
            return [
                self._create_classification(
                    ClauseType.OTHER,
                    0.0,
                    [],
                    metadata,
                )
            ]

        return classifications[:top_k]

    def _predict_keywords(
        self,
        text: str,
        *,
        top_k: int,
        metadata: Dict[str, Any],
    ) -> List[ClauseClassification]:
        """
        Deterministic fallback useful for:
            - local development
            - tests
            - model unavailable scenarios
            - explainability baseline
        """

        scores: List[ClauseClassification] = []

        for clause_type, keywords in self.KEYWORDS.items():
            matched = [
                keyword
                for keyword in keywords
                if self._contains_term(text, keyword)
            ]

            if not matched:
                continue

            # Saturating score prevents many repeated keywords
            # from producing an artificial confidence of 1.0.
            raw_score = min(
                0.45 + (0.10 * len(matched)),
                0.90,
            )

            scores.append(
                self._create_classification(
                    clause_type=clause_type,
                    confidence=raw_score,
                    evidence=matched,
                    metadata={
                        **metadata,
                        "classification_method": "keyword_fallback",
                    },
                )
            )

        if not scores:
            return [
                self._create_classification(
                    clause_type=ClauseType.OTHER,
                    confidence=0.20,
                    evidence=[],
                    metadata={
                        **metadata,
                        "classification_method": "fallback",
                    },
                )
            ]

        scores.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        # If the strongest result is weak, preserve OTHER as a
        # safer classification rather than overstating certainty.
        if scores[0].confidence < self.confidence_threshold:
            scores.insert(
                0,
                self._create_classification(
                    clause_type=ClauseType.OTHER,
                    confidence=1.0 - scores[0].confidence,
                    evidence=[],
                    metadata={
                        **metadata,
                        "classification_method": "low_confidence_fallback",
                    },
                ),
            )

        return scores[:top_k]

    def _extract_evidence(
        self,
        text: str,
        clause_type: ClauseType,
    ) -> List[str]:
        keywords = self.KEYWORDS.get(clause_type, ())

        return [
            keyword
            for keyword in keywords
            if self._contains_term(text, keyword)
        ][:10]

    def _map_label(
        self,
        label: str,
    ) -> Optional[ClauseType]:
        normalized = (
            label.strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        if normalized.startswith("label_"):
            normalized = normalized[6:]

        return self.LABEL_ALIASES.get(normalized)

    def _create_classification(
        self,
        clause_type: ClauseType,
        confidence: float,
        evidence: List[str],
        metadata: Dict[str, Any],
    ) -> ClauseClassification:
        return ClauseClassification(
            clause_type=clause_type,
            confidence=round(
                self._clamp(confidence),
                4,
            ),
            model_name=self.model_name,
            model_version=self.model_version,
            evidence=evidence,
            metadata=metadata,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            text.strip().lower(),
        )

    @staticmethod
    def _contains_term(
        text: str,
        term: str,
    ) -> bool:
        escaped = re.escape(term.lower())

        # Word boundaries for normal words; allows phrases.
        return bool(
            re.search(
                rf"(?<!\w){escaped}(?!\w)",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(
            0.0,
            min(float(value), 1.0),
        )