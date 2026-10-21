"""
backend/app/rag/generation/uncertainty_detector.py

Uncertainty detection for Enterprise Legal Document Intelligence AI.

Detects:
- weak evidence support
- missing evidence
- unsupported claims
- conflicting evidence
- hedging / uncertainty language
- low retrieval confidence
- low reranker confidence
- incomplete legal facts
- citation weakness
- potentially overconfident legal conclusions

This component does NOT determine legal correctness.
It estimates whether the generated answer should be treated with caution.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Types
# ============================================================================


class UncertaintyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UncertaintyReason(str, Enum):
    NO_CONTEXT = "no_context"
    WEAK_CONTEXT = "weak_context"
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
    LOW_RERANKER_CONFIDENCE = "low_reranker_confidence"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    MISSING_LEGAL_FACTS = "missing_legal_facts"
    HEDGED_ANSWER = "hedged_answer"
    WEAK_CITATIONS = "weak_citations"
    OVERCONFIDENT_LANGUAGE = "overconfident_language"


# ============================================================================
# Models
# ============================================================================


@dataclass(slots=True)
class UncertaintySignal:
    reason: UncertaintyReason
    score: float
    message: str

    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "score": self.score,
            "message": self.message,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class UncertaintyResult:
    level: UncertaintyLevel

    uncertainty_score: float
    confidence_score: float

    requires_review: bool

    signals: list[UncertaintySignal] = field(
        default_factory=list
    )

    unsupported_claims: list[str] = field(
        default_factory=list
    )

    conflicts: list[str] = field(
        default_factory=list
    )

    missing_information: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "uncertainty_score": self.uncertainty_score,
            "confidence_score": self.confidence_score,
            "requires_review": self.requires_review,
            "signals": [
                signal.to_dict()
                for signal in self.signals
            ],
            "unsupported_claims": self.unsupported_claims,
            "conflicts": self.conflicts,
            "missing_information": self.missing_information,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class UncertaintyConfig:
    medium_threshold: float = 0.35
    high_threshold: float = 0.60
    critical_threshold: float = 0.82

    claim_support_threshold: float = 0.30

    weak_retrieval_threshold: float = 0.30
    weak_reranker_threshold: float = 0.35

    require_context: bool = True

    review_threshold: float = 0.60

    minimum_claim_words: int = 4


# ============================================================================
# Detector
# ============================================================================


class UncertaintyDetector:
    """
    Multi-signal uncertainty detector.

    Example:

        result = detector.detect(
            answer=answer,
            contexts=reranked_chunks,
        )

        if result.requires_review:
            send_to_human_review(...)
    """

    SENTENCE_PATTERN = re.compile(
        r"(?<=[.!?])\s+"
    )

    HEDGE_PATTERNS = (
        re.compile(r"\bmay\b", re.I),
        re.compile(r"\bmight\b", re.I),
        re.compile(r"\bcould\b", re.I),
        re.compile(r"\bpossibly\b", re.I),
        re.compile(r"\bappears?\b", re.I),
        re.compile(r"\bseems?\b", re.I),
        re.compile(r"\blikely\b", re.I),
        re.compile(r"\bpotentially\b", re.I),
        re.compile(r"\bunclear\b", re.I),
        re.compile(r"\buncertain\b", re.I),
    )

    INSUFFICIENT_EVIDENCE_PATTERNS = (
        re.compile(
            r"not enough information",
            re.I,
        ),
        re.compile(
            r"insufficient information",
            re.I,
        ),
        re.compile(
            r"insufficient evidence",
            re.I,
        ),
        re.compile(
            r"cannot determine",
            re.I,
        ),
        re.compile(
            r"cannot be determined",
            re.I,
        ),
        re.compile(
            r"not provided",
            re.I,
        ),
        re.compile(
            r"not specified",
            re.I,
        ),
        re.compile(
            r"does not specify",
            re.I,
        ),
        re.compile(
            r"not stated",
            re.I,
        ),
        re.compile(
            r"not available",
            re.I,
        ),
    )

    CONFLICT_PATTERNS = (
        re.compile(
            r"\bconflicting\b",
            re.I,
        ),
        re.compile(
            r"\bconflict between\b",
            re.I,
        ),
        re.compile(
            r"\binconsistent\b",
            re.I,
        ),
        re.compile(
            r"\bcontradictory\b",
            re.I,
        ),
        re.compile(
            r"\bcontradicts?\b",
            re.I,
        ),
        re.compile(
            r"\bdiscrepancy\b",
            re.I,
        ),
    )

    OVERCONFIDENT_PATTERNS = (
        re.compile(
            r"\bdefinitely\b",
            re.I,
        ),
        re.compile(
            r"\bcertainly\b",
            re.I,
        ),
        re.compile(
            r"\bguaranteed\b",
            re.I,
        ),
        re.compile(
            r"\bwithout question\b",
            re.I,
        ),
        re.compile(
            r"\bno legal risk\b",
            re.I,
        ),
        re.compile(
            r"\b100%\s+(?:legal|valid|enforceable)\b",
            re.I,
        ),
    )

    LEGAL_FACT_PATTERNS: dict[str, re.Pattern[str]] = {
        "date": re.compile(
            r"\b(?:date|deadline|expiry|expiration|effective)\b",
            re.I,
        ),
        "notice_period": re.compile(
            r"\bnotice\s+period\b",
            re.I,
        ),
        "renewal": re.compile(
            r"\brenew(?:al|s|ed)?\b",
            re.I,
        ),
        "termination": re.compile(
            r"\bterminat(?:e|ion|ed)\b",
            re.I,
        ),
        "payment": re.compile(
            r"\b(?:payment|fee|amount|price|cost)\b",
            re.I,
        ),
        "liability": re.compile(
            r"\bliabilit(?:y|ies)\b",
            re.I,
        ),
        "jurisdiction": re.compile(
            r"\b(?:jurisdiction|governing law)\b",
            re.I,
        ),
    }

    def __init__(
        self,
        config: UncertaintyConfig | None = None,
    ) -> None:

        self.config = (
            config
            or UncertaintyConfig()
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def detect(
        self,
        answer: str,
        contexts: Sequence[Any],
        *,
        query: str | None = None,
        validation: Any = None,
        citations: Any = None,
    ) -> UncertaintyResult:

        answer = str(
            answer or ""
        ).strip()

        contexts = list(
            contexts or []
        )

        signals: list[
            UncertaintySignal
        ] = []

        unsupported_claims: list[str] = []
        conflicts: list[str] = []
        missing_information: list[str] = []

        # --------------------------------------------------------------------
        # Empty answer
        # --------------------------------------------------------------------

        if not answer:

            return UncertaintyResult(
                level=UncertaintyLevel.CRITICAL,
                uncertainty_score=1.0,
                confidence_score=0.0,
                requires_review=True,
                signals=[
                    UncertaintySignal(
                        reason=UncertaintyReason.WEAK_CONTEXT,
                        score=1.0,
                        message=(
                            "No generated answer was available."
                        ),
                    )
                ],
            )

        # --------------------------------------------------------------------
        # Normalize contexts
        # --------------------------------------------------------------------

        context_texts = [
            text
            for context in contexts
            if (
                text
                := self._extract_text(
                    context
                )
            )
        ]

        # --------------------------------------------------------------------
        # 1. Missing context
        # --------------------------------------------------------------------

        no_context_score = 0.0

        if not context_texts:

            no_context_score = 1.0

            signals.append(
                UncertaintySignal(
                    reason=UncertaintyReason.NO_CONTEXT,
                    score=1.0,
                    message=(
                        "No retrieved legal evidence "
                        "was available."
                    ),
                )
            )

        # --------------------------------------------------------------------
        # 2. Claim support
        # --------------------------------------------------------------------

        claims = self._extract_claims(
            answer
        )

        support_scores: list[float] = []

        for claim in claims:

            score = (
                self._best_support_score(
                    claim,
                    context_texts,
                )
            )

            support_scores.append(
                score
            )

            if (
                score
                < self.config.claim_support_threshold
            ):

                unsupported_claims.append(
                    claim
                )

        unsupported_ratio = (
            len(unsupported_claims)
            / len(claims)
            if claims
            else 0.0
        )

        if unsupported_claims:

            signals.append(
                UncertaintySignal(
                    reason=(
                        UncertaintyReason.UNSUPPORTED_CLAIMS
                    ),
                    score=self._clamp(
                        unsupported_ratio
                    ),
                    message=(
                        f"{len(unsupported_claims)} "
                        "answer claim(s) have weak "
                        "support in retrieved evidence."
                    ),
                    evidence=(
                        unsupported_claims[:5]
                    ),
                )
            )

        average_support = (
            sum(support_scores)
            / len(support_scores)
            if support_scores
            else 0.0
        )

        weak_context_score = (
            1.0 - average_support
            if context_texts
            else 1.0
        )

        if (
            context_texts
            and weak_context_score >= 0.5
        ):

            signals.append(
                UncertaintySignal(
                    reason=(
                        UncertaintyReason.WEAK_CONTEXT
                    ),
                    score=self._clamp(
                        weak_context_score
                    ),
                    message=(
                        "Generated answer has weak "
                        "support from retrieved context."
                    ),
                )
            )

        # --------------------------------------------------------------------
        # 3. Retrieval confidence
        # --------------------------------------------------------------------

        retrieval_scores = (
            self._collect_scores(
                contexts,
                (
                    "retrieval_score",
                    "score",
                ),
            )
        )

        retrieval_uncertainty = 0.0

        if retrieval_scores:

            retrieval_confidence = (
                sum(
                    self._normalize_score(
                        score
                    )
                    for score in retrieval_scores
                )
                / len(retrieval_scores)
            )

            if (
                retrieval_confidence
                < self.config.weak_retrieval_threshold
            ):

                retrieval_uncertainty = (
                    1.0
                    - retrieval_confidence
                )

                signals.append(
                    UncertaintySignal(
                        reason=(
                            UncertaintyReason
                            .LOW_RETRIEVAL_CONFIDENCE
                        ),
                        score=self._clamp(
                            retrieval_uncertainty
                        ),
                        message=(
                            "Retrieved documents have "
                            "low retrieval confidence."
                        ),
                    )
                )

        # --------------------------------------------------------------------
        # 4. Reranker confidence
        # --------------------------------------------------------------------

        reranker_scores = (
            self._collect_scores(
                contexts,
                (
                    "reranker_score",
                ),
            )
        )

        reranker_uncertainty = 0.0

        if reranker_scores:

            reranker_confidence = (
                sum(
                    self._normalize_score(
                        score
                    )
                    for score in reranker_scores
                )
                / len(reranker_scores)
            )

            if (
                reranker_confidence
                < self.config.weak_reranker_threshold
            ):

                reranker_uncertainty = (
                    1.0
                    - reranker_confidence
                )

                signals.append(
                    UncertaintySignal(
                        reason=(
                            UncertaintyReason
                            .LOW_RERANKER_CONFIDENCE
                        ),
                        score=self._clamp(
                            reranker_uncertainty
                        ),
                        message=(
                            "Reranked evidence has "
                            "low semantic relevance."
                        ),
                    )
                )

        # --------------------------------------------------------------------
        # 5. Hedging
        # --------------------------------------------------------------------

        hedge_matches = (
            self._find_matches(
                answer,
                self.HEDGE_PATTERNS,
            )
        )

        hedge_score = min(
            1.0,
            len(hedge_matches) / 5.0,
        )

        if hedge_matches:

            signals.append(
                UncertaintySignal(
                    reason=(
                        UncertaintyReason.HEDGED_ANSWER
                    ),
                    score=hedge_score,
                    message=(
                        "Answer contains uncertainty "
                        "or hedging language."
                    ),
                    evidence=hedge_matches[:5],
                )
            )

        # --------------------------------------------------------------------
        # 6. Explicit missing information
        # --------------------------------------------------------------------

        missing_matches = (
            self._find_matches(
                answer,
                self.INSUFFICIENT_EVIDENCE_PATTERNS,
            )
        )

        missing_information.extend(
            missing_matches
        )

        missing_score = min(
            1.0,
            len(missing_matches) / 3.0,
        )

        if missing_matches:

            signals.append(
                UncertaintySignal(
                    reason=(
                        UncertaintyReason
                        .MISSING_LEGAL_FACTS
                    ),
                    score=missing_score,
                    message=(
                        "The generated answer indicates "
                        "that required legal information "
                        "is missing."
                    ),
                    evidence=missing_matches[:5],
                )
            )

        # --------------------------------------------------------------------
        # 7. Evidence conflicts
        # --------------------------------------------------------------------

        conflict_matches = (
            self._find_matches(
                answer,
                self.CONFLICT_PATTERNS,
            )
        )

        conflicts.extend(
            conflict_matches
        )

        conflict_score = min(
            1.0,
            len(conflict_matches) / 2.0,
        )

        if conflict_matches:

            signals.append(
                UncertaintySignal(
                    reason=(
                        UncertaintyReason
                        .CONFLICTING_EVIDENCE
                    ),
                    score=conflict_score,
                    message=(
                        "Potential conflicting legal "
                        "evidence was identified."
                    ),
                    evidence=conflict_matches[:5],
                )
            )

        # --------------------------------------------------------------------
        # 8. Overconfident language
        # --------------------------------------------------------------------

        overconfident_matches = (
            self._find_matches(
                answer,
                self.OVERCONFIDENT_PATTERNS,
            )
        )

        overconfidence_score = min(
            1.0,
            len(overconfident_matches) / 2.0,
        )

        if overconfident_matches:

            signals.append(
                UncertaintySignal(
                    reason=(
                        UncertaintyReason
                        .OVERCONFIDENT_LANGUAGE
                    ),
                    score=overconfidence_score,
                    message=(
                        "Potentially overconfident legal "
                        "language was detected."
                    ),
                    evidence=(
                        overconfident_matches[:5]
                    ),
                )
            )

        # --------------------------------------------------------------------
        # 9. Citation signal
        # --------------------------------------------------------------------

        citation_uncertainty = (
            self._citation_uncertainty(
                citations
            )
        )

        if citation_uncertainty > 0.0:

            signals.append(
                UncertaintySignal(
                    reason=(
                        UncertaintyReason.WEAK_CITATIONS
                    ),
                    score=citation_uncertainty,
                    message=(
                        "Citation coverage or citation "
                        "support is weak."
                    ),
                )
            )

        # --------------------------------------------------------------------
        # 10. Validator signal
        # --------------------------------------------------------------------

        validation_uncertainty = (
            self._validation_uncertainty(
                validation
            )
        )

        # --------------------------------------------------------------------
        # Final ensemble uncertainty
        # --------------------------------------------------------------------

        components = {
            "no_context": no_context_score,
            "unsupported_claims": unsupported_ratio,
            "weak_context": weak_context_score,
            "retrieval": retrieval_uncertainty,
            "reranker": reranker_uncertainty,
            "hedging": hedge_score,
            "missing_information": missing_score,
            "conflicts": conflict_score,
            "overconfidence": overconfidence_score,
            "citations": citation_uncertainty,
            "validation": validation_uncertainty,
        }

        uncertainty = (
            0.22 * no_context_score
            + 0.20 * unsupported_ratio
            + 0.18 * weak_context_score
            + 0.07 * retrieval_uncertainty
            + 0.07 * reranker_uncertainty
            + 0.04 * hedge_score
            + 0.06 * missing_score
            + 0.06 * conflict_score
            + 0.03 * overconfidence_score
            + 0.03 * citation_uncertainty
            + 0.04 * validation_uncertainty
        )

        # Missing all context is always critical.
        if (
            self.config.require_context
            and not context_texts
        ):
            uncertainty = 1.0

        uncertainty = self._clamp(
            uncertainty
        )

        confidence = self._clamp(
            1.0 - uncertainty
        )

        level = self._level(
            uncertainty
        )

        requires_review = (
            uncertainty
            >= self.config.review_threshold
            or level
            in {
                UncertaintyLevel.HIGH,
                UncertaintyLevel.CRITICAL,
            }
        )

        result = UncertaintyResult(
            level=level,
            uncertainty_score=round(
                uncertainty,
                4,
            ),
            confidence_score=round(
                confidence,
                4,
            ),
            requires_review=requires_review,
            signals=signals,
            unsupported_claims=unsupported_claims,
            conflicts=conflicts,
            missing_information=missing_information,
            metadata={
                "query": query,
                "claim_count": len(claims),
                "context_count": len(context_texts),
                "average_claim_support": round(
                    average_support,
                    4,
                ),
                "components": {
                    key: round(
                        value,
                        4,
                    )
                    for key, value
                    in components.items()
                },
            },
        )

        logger.debug(
            "Uncertainty level=%s score=%.4f confidence=%.4f "
            "review=%s",
            result.level.value,
            result.uncertainty_score,
            result.confidence_score,
            result.requires_review,
        )

        return result

    # ========================================================================
    # Claim extraction
    # ========================================================================

    def _extract_claims(
        self,
        answer: str,
    ) -> list[str]:

        sentences = (
            self.SENTENCE_PATTERN.split(
                answer
            )
        )

        output: list[str] = []

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if (
                len(
                    self._tokenize(
                        sentence
                    )
                )
                >= self.config.minimum_claim_words
            ):
                output.append(
                    sentence
                )

        return output

    # ========================================================================
    # Evidence support
    # ========================================================================

    def _best_support_score(
        self,
        claim: str,
        contexts: Sequence[str],
    ) -> float:

        if not contexts:
            return 0.0

        claim_tokens = set(
            self._tokenize(
                claim
            )
        )

        if not claim_tokens:
            return 0.0

        best = 0.0

        for context in contexts:

            context_tokens = set(
                self._tokenize(
                    context
                )
            )

            if not context_tokens:
                continue

            overlap = (
                claim_tokens
                & context_tokens
            )

            score = (
                len(overlap)
                / len(claim_tokens)
            )

            best = max(
                best,
                score,
            )

        return self._clamp(
            best
        )

    # ========================================================================
    # Citation integration
    # ========================================================================

    def _citation_uncertainty(
        self,
        citations: Any,
    ) -> float:

        if citations is None:
            return 0.0

        # CitationResult object.
        citation_count = getattr(
            citations,
            "citation_count",
            None,
        )

        uncited_claims = getattr(
            citations,
            "uncited_claims",
            None,
        )

        if citation_count is not None:

            uncited_count = len(
                uncited_claims or []
            )

            total = (
                int(citation_count)
                + uncited_count
            )

            if total <= 0:
                return 1.0

            return self._clamp(
                uncited_count / total
            )

        # Dict representation.
        if isinstance(
            citations,
            Mapping,
        ):

            count = citations.get(
                "citation_count"
            )

            uncited = citations.get(
                "uncited_claims",
                [],
            )

            if count is not None:

                total = (
                    int(count)
                    + len(uncited)
                )

                if total <= 0:
                    return 1.0

                return self._clamp(
                    len(uncited) / total
                )

        # Plain citation list.
        if isinstance(
            citations,
            Sequence,
        ) and not isinstance(
            citations,
            (str, bytes),
        ):

            return (
                0.0
                if len(citations) > 0
                else 1.0
            )

        return 0.0

    # ========================================================================
    # Validator integration
    # ========================================================================

    @staticmethod
    def _validation_uncertainty(
        validation: Any,
    ) -> float:

        if validation is None:
            return 0.0

        score = None

        if isinstance(
            validation,
            Mapping,
        ):
            score = validation.get(
                "score"
            )

        else:
            score = getattr(
                validation,
                "score",
                None,
            )

        if score is None:
            return 0.0

        try:
            score = float(score)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                1.0 - score,
            ),
        )

    # ========================================================================
    # Scores
    # ========================================================================

    def _collect_scores(
        self,
        contexts: Sequence[Any],
        names: Sequence[str],
    ) -> list[float]:

        output: list[float] = []

        for context in contexts:

            value = self._get_value(
                context,
                *names,
                default=None,
            )

            if value is None:
                continue

            try:
                number = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if (
                math.isnan(number)
                or math.isinf(number)
            ):
                continue

            output.append(
                number
            )

        return output

    @staticmethod
    def _normalize_score(
        score: float,
    ) -> float:

        if 0.0 <= score <= 1.0:
            return score

        # Positive cross-encoder/logit-style score.
        if score > 1.0:
            return score / (
                1.0 + score
            )

        return 0.0

    # ========================================================================
    # Pattern matching
    # ========================================================================

    @staticmethod
    def _find_matches(
        text: str,
        patterns: Sequence[
            re.Pattern[str]
        ],
    ) -> list[str]:

        findings: list[str] = []

        seen: set[str] = set()

        for pattern in patterns:

            for match in pattern.finditer(
                text
            ):

                value = (
                    match.group(0)
                    .strip()
                )

                normalized = (
                    value.lower()
                )

                if normalized in seen:
                    continue

                seen.add(
                    normalized
                )

                findings.append(
                    value
                )

        return findings

    # ========================================================================
    # Context extraction
    # ========================================================================

    @staticmethod
    def _extract_text(
        context: Any,
    ) -> str:

        if context is None:
            return ""

        if isinstance(
            context,
            str,
        ):
            return context.strip()

        if isinstance(
            context,
            Mapping,
        ):

            for key in (
                "text",
                "content",
                "page_content",
                "chunk_text",
            ):

                value = context.get(
                    key
                )

                if value:
                    return str(
                        value
                    ).strip()

            return ""

        for attribute in (
            "text",
            "content",
            "page_content",
            "chunk_text",
        ):

            value = getattr(
                context,
                attribute,
                None,
            )

            if value:
                return str(
                    value
                ).strip()

        return ""

    # ========================================================================
    # Generic reader
    # ========================================================================

    @staticmethod
    def _get_value(
        obj: Any,
        *names: str,
        default: Any = None,
    ) -> Any:

        if isinstance(
            obj,
            Mapping,
        ):

            for name in names:

                if name in obj:
                    return obj[
                        name
                    ]

            return default

        for name in names:

            if hasattr(
                obj,
                name,
            ):
                return getattr(
                    obj,
                    name,
                )

        return default

    # ========================================================================
    # Tokenization
    # ========================================================================

    @staticmethod
    def _tokenize(
        text: str,
    ) -> list[str]:

        tokens = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_\-']+",
            str(text).lower(),
        )

        stopwords = {
            "a", "an", "the",
            "and", "or", "but",
            "of", "to", "in",
            "on", "for", "with",
            "from", "by", "at",
            "as", "is", "are",
            "was", "were", "be",
            "been", "being",
            "this", "that",
            "these", "those",
            "it", "its",
        }

        return [
            token
            for token in tokens
            if token not in stopwords
        ]

    # ========================================================================
    # Level mapping
    # ========================================================================

    def _level(
        self,
        score: float,
    ) -> UncertaintyLevel:

        if (
            score
            >= self.config.critical_threshold
        ):
            return (
                UncertaintyLevel.CRITICAL
            )

        if (
            score
            >= self.config.high_threshold
        ):
            return (
                UncertaintyLevel.HIGH
            )

        if (
            score
            >= self.config.medium_threshold
        ):
            return (
                UncertaintyLevel.MEDIUM
            )

        return UncertaintyLevel.LOW

    # ========================================================================
    # Utility
    # ========================================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        if (
            math.isnan(value)
            or math.isinf(value)
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )