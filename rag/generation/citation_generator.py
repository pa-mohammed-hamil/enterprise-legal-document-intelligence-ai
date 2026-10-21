"""
backend/app/rag/generation/answer_validator.py

Grounded answer validation for Enterprise Legal Document Intelligence AI.

Validates:
- Groundedness
- Citation presence / validity
- Source support
- Hallucination risk
- Legal-language safety
- Unsupported claims
- Context coverage
- Confidence
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "warning"
    claim: str | None = None
    source_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "claim": self.claim,
            "source_id": self.source_id,
        }


@dataclass(slots=True)
class AnswerValidationResult:
    status: ValidationStatus

    valid: bool
    score: float

    groundedness_score: float
    citation_score: float
    context_coverage_score: float
    hallucination_risk: float

    supported_claims: int = 0
    unsupported_claims: int = 0

    issues: list[ValidationIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "valid": self.valid,
            "score": self.score,
            "groundedness_score": self.groundedness_score,
            "citation_score": self.citation_score,
            "context_coverage_score": self.context_coverage_score,
            "hallucination_risk": self.hallucination_risk,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class AnswerValidatorConfig:
    min_validation_score: float = 0.65
    warning_score: float = 0.78

    min_groundedness: float = 0.60
    min_citation_score: float = 0.50

    hallucination_threshold: float = 0.40

    require_context: bool = True
    require_citations: bool = False

    minimum_claim_words: int = 4


class AnswerValidator:
    """
    Deterministic first-pass validator for generated legal RAG answers.

    Important:
    This validator should complement, not replace, stronger NLI/LLM-based
    groundedness evaluation in production.
    """

    CITATION_PATTERN = re.compile(
        r"""
        (?:
            \[(?:source|document|doc|citation)?\s*\d+\]
            |
            \[\d+\]
            |
            \(source\s+\d+\)
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    LEGAL_ABSOLUTE_PATTERNS = (
        re.compile(r"\bguaranteed\b", re.I),
        re.compile(r"\bdefinitely legal\b", re.I),
        re.compile(r"\bdefinitely illegal\b", re.I),
        re.compile(r"\b100%\s+(?:legal|valid|enforceable)\b", re.I),
        re.compile(r"\bwill certainly win\b", re.I),
        re.compile(r"\bno legal risk\b", re.I),
    )

    def __init__(
        self,
        config: AnswerValidatorConfig | None = None,
    ) -> None:
        self.config = config or AnswerValidatorConfig()

    def validate(
        self,
        answer: str,
        contexts: Sequence[Any],
    ) -> AnswerValidationResult:

        answer = (answer or "").strip()
        contexts = list(contexts or [])

        issues: list[ValidationIssue] = []

        # -------------------------------------------------------------
        # Empty answer
        # -------------------------------------------------------------

        if not answer:
            return AnswerValidationResult(
                status=ValidationStatus.FAIL,
                valid=False,
                score=0.0,
                groundedness_score=0.0,
                citation_score=0.0,
                context_coverage_score=0.0,
                hallucination_risk=1.0,
                issues=[
                    ValidationIssue(
                        code="EMPTY_ANSWER",
                        message="Generated answer is empty.",
                        severity="error",
                    )
                ],
            )

        # -------------------------------------------------------------
        # Context extraction
        # -------------------------------------------------------------

        context_texts = [
            text
            for context in contexts
            if (text := self._extract_text(context))
        ]

        if not context_texts:
            issues.append(
                ValidationIssue(
                    code="NO_CONTEXT",
                    message=(
                        "No retrieved legal context was available "
                        "for answer validation."
                    ),
                    severity="error",
                )
            )

        # -------------------------------------------------------------
        # Claim extraction
        # -------------------------------------------------------------

        claims = self._extract_claims(answer)

        supported = 0
        unsupported = 0

        support_scores: list[float] = []

        for claim in claims:

            score = self._claim_support_score(
                claim,
                context_texts,
            )

            support_scores.append(score)

            if score >= 0.35:
                supported += 1
            else:
                unsupported += 1

                issues.append(
                    ValidationIssue(
                        code="UNSUPPORTED_CLAIM",
                        message=(
                            "The answer contains a claim with weak "
                            "support in retrieved context."
                        ),
                        severity="warning",
                        claim=claim,
                    )
                )

        # -------------------------------------------------------------
        # Groundedness
        # -------------------------------------------------------------

        if support_scores:
            groundedness = sum(support_scores) / len(support_scores)
        elif context_texts:
            groundedness = self._text_support_score(
                answer,
                context_texts,
            )
        else:
            groundedness = 0.0

        groundedness = self._clamp(groundedness)

        if groundedness < self.config.min_groundedness:
            issues.append(
                ValidationIssue(
                    code="LOW_GROUNDEDNESS",
                    message=(
                        "Generated answer has insufficient lexical "
                        "support from retrieved documents."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Citations
        # -------------------------------------------------------------

        citation_score = self._citation_score(
            answer,
            contexts,
        )

        if (
            self.config.require_citations
            and citation_score < self.config.min_citation_score
        ):
            issues.append(
                ValidationIssue(
                    code="MISSING_CITATIONS",
                    message=(
                        "Answer does not contain sufficient "
                        "source citations."
                    ),
                )
            )

        # -------------------------------------------------------------
        # Context coverage
        # -------------------------------------------------------------

        context_coverage = self._context_coverage(
            answer,
            context_texts,
        )

        # -------------------------------------------------------------
        # Legal overconfidence
        # -------------------------------------------------------------

        absolute_claims = self._find_absolute_legal_claims(answer)

        for claim in absolute_claims:
            issues.append(
                ValidationIssue(
                    code="LEGAL_OVERCONFIDENCE",
                    message=(
                        "Potentially overconfident legal statement "
                        "detected."
                    ),
                    severity="warning",
                    claim=claim,
                )
            )

        # -------------------------------------------------------------
        # Hallucination risk
        # -------------------------------------------------------------

        unsupported_ratio = (
            unsupported / len(claims)
            if claims
            else 0.0
        )

        hallucination_risk = (
            0.60 * (1.0 - groundedness)
            + 0.30 * unsupported_ratio
            + 0.10 * (1.0 - context_coverage)
        )

        if not context_texts:
            hallucination_risk = 1.0

        hallucination_risk = self._clamp(hallucination_risk)

        if hallucination_risk >= self.config.hallucination_threshold:
            issues.append(
                ValidationIssue(
                    code="HALLUCINATION_RISK",
                    message=(
                        "Answer has elevated hallucination risk."
                    ),
                    severity="warning",
                )
            )

        # -------------------------------------------------------------
        # Final validation score
        # -------------------------------------------------------------

        validation_score = (
            0.50 * groundedness
            + 0.25 * citation_score
            + 0.15 * context_coverage
            + 0.10 * (1.0 - hallucination_risk)
        )

        # Citation absence should not heavily punish deployments where
        # inline citations are intentionally disabled.

        if not self.config.require_citations:
            validation_score = (
                0.60 * groundedness
                + 0.20 * context_coverage
                + 0.20 * (1.0 - hallucination_risk)
            )

        validation_score = self._clamp(validation_score)

        # -------------------------------------------------------------
        # Determine status
        # -------------------------------------------------------------

        hard_failure = (
            self.config.require_context
            and not context_texts
        )

        if (
            hard_failure
            or validation_score < self.config.min_validation_score
        ):
            status = ValidationStatus.FAIL

        elif (
            validation_score < self.config.warning_score
            or hallucination_risk
            >= self.config.hallucination_threshold
        ):
            status = ValidationStatus.WARNING

        else:
            status = ValidationStatus.PASS

        valid = status != ValidationStatus.FAIL

        result = AnswerValidationResult(
            status=status,
            valid=valid,
            score=round(validation_score, 4),
            groundedness_score=round(groundedness, 4),
            citation_score=round(citation_score, 4),
            context_coverage_score=round(context_coverage, 4),
            hallucination_risk=round(hallucination_risk, 4),
            supported_claims=supported,
            unsupported_claims=unsupported,
            issues=issues,
            metadata={
                "claim_count": len(claims),
                "context_count": len(context_texts),
                "absolute_legal_claims": len(absolute_claims),
            },
        )

        logger.debug(
            "Answer validation status=%s score=%.4f "
            "groundedness=%.4f hallucination=%.4f",
            result.status.value,
            result.score,
            result.groundedness_score,
            result.hallucination_risk,
        )

        return result

    # ------------------------------------------------------------------
    # Claims
    # ------------------------------------------------------------------

    def _extract_claims(
        self,
        answer: str,
    ) -> list[str]:

        sentences = self.SENTENCE_PATTERN.split(answer)

        claims: list[str] = []

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            # Remove common citation syntax before word count.
            clean = self.CITATION_PATTERN.sub("", sentence)

            words = self._tokenize(clean)

            if len(words) >= self.config.minimum_claim_words:
                claims.append(sentence)

        return claims

    # ------------------------------------------------------------------
    # Groundedness
    # ------------------------------------------------------------------

    def _claim_support_score(
        self,
        claim: str,
        contexts: Sequence[str],
    ) -> float:

        if not contexts:
            return 0.0

        claim_tokens = set(self._tokenize(claim))

        if not claim_tokens:
            return 0.0

        best_score = 0.0

        for context in contexts:

            context_tokens = set(self._tokenize(context))

            if not context_tokens:
                continue

            overlap = claim_tokens.intersection(context_tokens)

            score = len(overlap) / len(claim_tokens)

            best_score = max(best_score, score)

        return self._clamp(best_score)

    def _text_support_score(
        self,
        answer: str,
        contexts: Sequence[str],
    ) -> float:

        answer_tokens = set(self._tokenize(answer))

        if not answer_tokens:
            return 0.0

        context_tokens: set[str] = set()

        for context in contexts:
            context_tokens.update(
                self._tokenize(context)
            )

        overlap = answer_tokens.intersection(context_tokens)

        return self._clamp(
            len(overlap) / len(answer_tokens)
        )

    # ------------------------------------------------------------------
    # Citation validation
    # ------------------------------------------------------------------

    def _citation_score(
        self,
        answer: str,
        contexts: Sequence[Any],
    ) -> float:

        citations = self.CITATION_PATTERN.findall(answer)

        if not citations:
            return 0.0

        # Avoid rewarding arbitrarily many citation markers.
        expected = max(1, min(len(contexts), 5))

        return self._clamp(
            len(citations) / expected
        )

    # ------------------------------------------------------------------
    # Context coverage
    # ------------------------------------------------------------------

    def _context_coverage(
        self,
        answer: str,
        contexts: Sequence[str],
    ) -> float:

        if not contexts:
            return 0.0

        answer_tokens = set(self._tokenize(answer))

        if not answer_tokens:
            return 0.0

        covered = 0

        for context in contexts:

            context_tokens = set(
                self._tokenize(context)
            )

            overlap = (
                answer_tokens
                .intersection(context_tokens)
            )

            if len(overlap) >= 3:
                covered += 1

        return self._clamp(
            covered / len(contexts)
        )

    # ------------------------------------------------------------------
    # Legal safety
    # ------------------------------------------------------------------

    def _find_absolute_legal_claims(
        self,
        answer: str,
    ) -> list[str]:

        findings: list[str] = []

        for pattern in self.LEGAL_ABSOLUTE_PATTERNS:

            for match in pattern.finditer(answer):
                findings.append(match.group(0))

        return findings

    # ------------------------------------------------------------------
    # Context extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(context: Any) -> str:

        if context is None:
            return ""

        if isinstance(context, str):
            return context.strip()

        if isinstance(context, Mapping):

            for key in (
                "text",
                "content",
                "page_content",
                "chunk_text",
            ):
                value = context.get(key)

                if value:
                    return str(value).strip()

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
                return str(value).strip()

        return ""

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        tokens = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_\-']+",
            text.lower(),
        )

        stopwords = {
            "the", "a", "an", "and", "or", "of",
            "to", "in", "on", "for", "with",
            "is", "are", "was", "were", "be",
            "been", "being", "this", "that",
            "these", "those", "as", "at", "by",
            "from", "it", "its",
        }

        return [
            token
            for token in tokens
            if token not in stopwords
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:

        if math.isnan(value) or math.isinf(value):
            return 0.0

        return max(
            0.0,
            min(1.0, float(value)),
        )