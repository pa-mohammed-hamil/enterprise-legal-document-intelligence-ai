"""
Enterprise Legal Document Intelligence AI
------------------------------------------

Review Agent

Responsible for reviewing AI-generated legal-document findings and
organizing them into an evidence-grounded human-review workflow.

Responsibilities
----------------
- Aggregate document / clause / obligation / risk findings.
- Prioritize findings for human review.
- Validate evidence and confidence.
- Detect unresolved or contradictory findings.
- Produce reviewer recommendations.
- Support approve / reject / revise / escalate decisions.
- Enforce tenant isolation.
- Detect prompt injection in supplied evidence.
- Apply output guardrails.
- Produce structured review packets.

The agent does NOT:
- provide legal advice,
- autonomously approve legally binding decisions,
- bypass reviewer authorization,
- modify source contracts,
- send notifications,
- schedule review tasks.

Those responsibilities belong to service, authorization, task,
notification, and human-review layers.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

from backend.app.core.enums import (
    ReviewPriority,
    ReviewStatus,
    RiskLevel,
)
from backend.app.core.exceptions import (
    AgentExecutionError,
    ForbiddenError,
    ValidationError,
)
from backend.app.core.logging import get_logger
from backend.app.core.telemetry import telemetry
from backend.app.security.content_filter import (
    get_content_filter,
)
from backend.app.security.output_guardrails import (
    get_output_guardrails,
)
from backend.app.security.prompt_injection import (
    InputSource,
    get_prompt_injection_detector,
)
from backend.app.security.tenant_isolation import (
    assert_tenant_access,
)

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class ReviewAgentMode(str, Enum):
    """Review-analysis modes."""

    TRIAGE = "triage"
    EVIDENCE = "evidence"
    RISK = "risk"
    QUALITY = "quality"
    FULL = "full"


class ReviewDecision(str, Enum):
    """Possible reviewer decisions."""

    APPROVE = "approve"
    REJECT = "reject"
    REVISE = "revise"
    ESCALATE = "escalate"
    DEFER = "defer"
    PENDING = "pending"


class ReviewFindingType(str, Enum):
    """Types of findings that may require review."""

    RISK = "risk"
    CLAUSE = "clause"
    OBLIGATION = "obligation"
    RENEWAL = "renewal"
    EXPIRY = "expiry"
    COMPLIANCE = "compliance"
    EXTRACTION = "extraction"
    RETRIEVAL = "retrieval"
    CITATION = "citation"
    SECURITY = "security"
    DATA_QUALITY = "data_quality"
    CONTRADICTION = "contradiction"


class ReviewSeverity(str, Enum):
    """Severity used for review triage."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(slots=True)
class ReviewDocument:
    """Document under review."""

    document_id: str

    tenant_id: str

    title: Optional[str] = None

    text: Optional[str] = None

    version_id: Optional[str] = None

    contract_id: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not self.document_id:
            raise ValidationError(
                "document_id is required"
            )

        if not self.tenant_id:
            raise ValidationError(
                "tenant_id is required"
            )


@dataclass(slots=True)
class ReviewEvidence:
    """Evidence supporting a review item."""

    evidence_id: str

    document_id: str

    text: str

    source_type: str = "document"

    page: Optional[int] = None

    section: Optional[str] = None

    clause_id: Optional[str] = None

    confidence: float = 0.0

    citation: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewFinding:
    """Individual issue requiring review."""

    finding_id: str

    finding_type: ReviewFindingType

    title: str

    description: str

    severity: ReviewSeverity = (
        ReviewSeverity.MEDIUM
    )

    risk_level: RiskLevel = (
        RiskLevel.MEDIUM
    )

    confidence: float = 0.0

    requires_human_review: bool = True

    evidence_ids: list[str] = field(
        default_factory=list
    )

    source_agent: Optional[str] = None

    source_entity_id: Optional[str] = None

    recommended_action: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["finding_type"] = (
            self.finding_type.value
        )

        result["severity"] = (
            self.severity.value
        )

        result["risk_level"] = (
            self.risk_level.value
        )

        return result


@dataclass(slots=True)
class ReviewConflict:
    """Contradiction between findings or sources."""

    conflict_id: str

    description: str

    finding_ids: list[str] = field(
        default_factory=list
    )

    severity: ReviewSeverity = (
        ReviewSeverity.HIGH
    )

    resolution_required: bool = True

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["severity"] = (
            self.severity.value
        )
        return result


@dataclass(slots=True)
class ReviewItem:
    """Prioritized review queue item."""

    review_item_id: str

    finding_id: str

    priority: ReviewPriority

    severity: ReviewSeverity

    title: str

    rationale: str

    recommended_action: str

    evidence_ids: list[str] = field(
        default_factory=list
    )

    estimated_effort: str = "medium"

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        result["priority"] = (
            self.priority.value
        )

        result["severity"] = (
            self.severity.value
        )

        return result


@dataclass(slots=True)
class ReviewDecisionProposal:
    """AI-generated proposal for a human reviewer."""

    decision: ReviewDecision

    rationale: str

    confidence: float

    required_reviewer_role: Optional[str] = None

    conditions: list[str] = field(
        default_factory=list
    )

    evidence_ids: list[str] = field(
        default_factory=list
    )

    is_binding: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["decision"] = (
            self.decision.value
        )
        return result


@dataclass(slots=True)
class ReviewAnalysis:
    """Complete review-agent output."""

    document_id: str

    tenant_id: str

    status: ReviewStatus

    findings: list[
        ReviewFinding
    ] = field(default_factory=list)

    evidence: list[
        ReviewEvidence
    ] = field(default_factory=list)

    conflicts: list[
        ReviewConflict
    ] = field(default_factory=list)

    review_queue: list[
        ReviewItem
    ] = field(default_factory=list)

    proposal: Optional[
        ReviewDecisionProposal
    ] = None

    high_priority_count: int = 0

    critical_count: int = 0

    unresolved_count: int = 0

    overall_risk: RiskLevel = (
        RiskLevel.LOW
    )

    confidence: float = 0.0

    recommendations: list[str] = (
        field(default_factory=list)
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "findings": [
                item.to_dict()
                for item in self.findings
            ],
            "evidence": [
                item.to_dict()
                for item in self.evidence
            ],
            "conflicts": [
                item.to_dict()
                for item in self.conflicts
            ],
            "review_queue": [
                item.to_dict()
                for item in self.review_queue
            ],
            "proposal": (
                self.proposal.to_dict()
                if self.proposal
                else None
            ),
            "high_priority_count": (
                self.high_priority_count
            ),
            "critical_count": (
                self.critical_count
            ),
            "unresolved_count": (
                self.unresolved_count
            ),
            "overall_risk": (
                self.overall_risk.value
            ),
            "confidence": self.confidence,
            "recommendations": (
                self.recommendations
            ),
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class ReviewAgentConfig:
    """Runtime configuration."""

    max_findings: int = 200

    max_evidence: int = 300

    max_review_items: int = 100

    max_document_chars: int = 200_000

    minimum_confidence: float = 0.55

    critical_risk_requires_review: bool = True

    low_confidence_requires_review: bool = True

    enable_llm: bool = True

    enable_security_scanning: bool = True

    enable_output_guardrails: bool = True

    refuse_on_prompt_injection: bool = True


# ============================================================================
# REVIEW AGENT
# ============================================================================


class ReviewAgent:
    """
    Aggregate and triage AI-generated legal findings.

    Pipeline:

        Findings
           |
           v
        Normalize
           |
           v
        Tenant Authorization
           |
           v
        Security Scan
           |
           v
        Evidence Validation
           |
           v
        Contradiction Detection
           |
           v
        Risk / Confidence Analysis
           |
           v
        Review Prioritization
           |
           v
        Reviewer Proposal
           |
           v
        Output Guardrails
           |
           v
        Human Review Queue
    """

    AGENT_NAME = "review_agent"

    AGENT_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        config: Optional[
            ReviewAgentConfig
        ] = None,
        llm: Optional[
            Callable[..., Any]
        ] = None,
        review_service: Optional[
            Any
        ] = None,
    ) -> None:
        self.config = (
            config
            or ReviewAgentConfig()
        )

        self.llm = llm

        self.review_service = (
            review_service
        )

        self.injection_detector = (
            get_prompt_injection_detector()
        )

        self.content_filter = (
            get_content_filter()
        )

        self.output_guardrails = (
            get_output_guardrails()
        )

        logger.info(
            "review_agent_initialized",
            extra={
                "agent": self.AGENT_NAME,
                "version": self.AGENT_VERSION,
                "llm_enabled": bool(
                    llm
                ),
            },
        )

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def analyze(
        self,
        document: (
            ReviewDocument
            | Mapping[str, Any]
        ),
        findings: Optional[
            Sequence[
                ReviewFinding
                | Mapping[str, Any]
            ]
        ] = None,
        *,
        mode: ReviewAgentMode = (
            ReviewAgentMode.FULL
        ),
        tenant_id: Optional[str] = None,
    ) -> ReviewAnalysis:
        """
        Analyze findings and construct a human-review packet.
        """

        started_at = datetime.now(
            timezone.utc
        )

        normalized_document = (
            self._normalize_document(
                document,
                tenant_id=tenant_id,
            )
        )

        normalized_document.validate()

        effective_tenant = (
            tenant_id
            or normalized_document.tenant_id
        )

        if (
            normalized_document.tenant_id
            != effective_tenant
        ):
            raise ForbiddenError(
                "Document tenant does not match current tenant"
            )

        if (
            normalized_document.text
            and len(
                normalized_document.text
            )
            > self.config.max_document_chars
        ):
            raise ValidationError(
                "Document exceeds maximum review size"
            )

        self._authorize_document(
            normalized_document,
            effective_tenant,
        )

        try:
            normalized_findings = (
                self._normalize_findings(
                    findings or []
                )
            )

            security = (
                self._security_scan(
                    normalized_document,
                    normalized_findings,
                )
            )

            if (
                security[
                    "prompt_injection_detected"
                ]
                and self.config.refuse_on_prompt_injection
            ):
                return self._security_refusal(
                    normalized_document,
                    effective_tenant,
                    started_at,
                )

            evidence = (
                self._extract_evidence(
                    normalized_findings
                )
            )

            if mode in (
                ReviewAgentMode.EVIDENCE,
                ReviewAgentMode.FULL,
            ):
                evidence = (
                    self._validate_evidence(
                        evidence
                    )
                )

            conflicts = (
                self._detect_conflicts(
                    normalized_findings
                )
            )

            analyzed_findings = (
                self._analyze_findings(
                    normalized_findings,
                    conflicts,
                    mode=mode,
                )
            )

            review_queue = (
                self._build_review_queue(
                    analyzed_findings,
                    conflicts,
                )
            )

            if mode == (
                ReviewAgentMode.RISK
            ):
                analyzed_findings = [
                    item
                    for item in analyzed_findings
                    if item.risk_level
                    in (
                        RiskLevel.HIGH,
                        RiskLevel.CRITICAL,
                    )
                ]

                review_queue = [
                    item
                    for item in review_queue
                    if item.priority
                    in (
                        self._priority_high(),
                        self._priority_critical(),
                    )
                ]

            proposal = (
                self._build_decision_proposal(
                    analyzed_findings,
                    conflicts,
                )
            )

            recommendations = (
                self._build_recommendations(
                    analyzed_findings,
                    conflicts,
                )
            )

            critical_count = sum(
                1
                for item in analyzed_findings
                if item.severity
                == ReviewSeverity.CRITICAL
            )

            high_priority_count = sum(
                1
                for item in review_queue
                if item.priority
                in (
                    self._priority_high(),
                    self._priority_critical(),
                )
            )

            unresolved_count = sum(
                1
                for item in analyzed_findings
                if item.requires_human_review
            )

            result = ReviewAnalysis(
                document_id=(
                    normalized_document.document_id
                ),
                tenant_id=effective_tenant,
                status=self._calculate_status(
                    analyzed_findings,
                    conflicts,
                ),
                findings=analyzed_findings[
                    : self.config.max_findings
                ],
                evidence=evidence[
                    : self.config.max_evidence
                ],
                conflicts=conflicts,
                review_queue=review_queue[
                    : self.config.max_review_items
                ],
                proposal=proposal,
                high_priority_count=(
                    high_priority_count
                ),
                critical_count=critical_count,
                unresolved_count=(
                    unresolved_count
                ),
                overall_risk=(
                    self._overall_risk(
                        analyzed_findings
                    )
                ),
                confidence=(
                    self._overall_confidence(
                        analyzed_findings
                    )
                ),
                recommendations=recommendations,
                metadata={
                    "agent": self.AGENT_NAME,
                    "agent_version": (
                        self.AGENT_VERSION
                    ),
                    "mode": mode.value,
                    "security": security,
                    "binding_decision": False,
                },
                created_at=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

            self._apply_output_guardrails(
                result
            )

            self._record_telemetry(
                result,
                started_at,
            )

            return result

        except (
            ValidationError,
            ForbiddenError,
        ):
            raise

        except Exception as exc:
            logger.exception(
                "review_agent_failed",
                extra={
                    "document_id": (
                        normalized_document.document_id
                    ),
                    "tenant_id": effective_tenant,
                },
            )

            raise AgentExecutionError(
                message="Review analysis failed",
                details={
                    "document_id": (
                        normalized_document.document_id
                    ),
                    "error_type": type(
                        exc
                    ).__name__,
                },
            ) from exc

    # ========================================================================
    # NORMALIZATION
    # ========================================================================

    def _normalize_document(
        self,
        document: (
            ReviewDocument
            | Mapping[str, Any]
        ),
        *,
        tenant_id: Optional[str],
    ) -> ReviewDocument:
        if isinstance(
            document,
            ReviewDocument,
        ):
            return document

        if not isinstance(
            document,
            Mapping,
        ):
            raise ValidationError(
                "Unsupported document format"
            )

        return ReviewDocument(
            document_id=str(
                document.get(
                    "document_id"
                )
                or ""
            ),
            tenant_id=str(
                document.get(
                    "tenant_id"
                )
                or tenant_id
                or ""
            ),
            title=(
                str(
                    document["title"]
                )
                if document.get(
                    "title"
                ) is not None
                else None
            ),
            text=(
                str(
                    document["text"]
                )
                if document.get(
                    "text"
                ) is not None
                else (
                    str(
                        document["content"]
                    )
                    if document.get(
                        "content"
                    ) is not None
                    else None
                )
            ),
            version_id=(
                str(
                    document["version_id"]
                )
                if document.get(
                    "version_id"
                ) is not None
                else None
            ),
            contract_id=(
                str(
                    document["contract_id"]
                )
                if document.get(
                    "contract_id"
                ) is not None
                else None
            ),
            metadata=dict(
                document.get(
                    "metadata"
                )
                or {}
            ),
        )

    def _normalize_findings(
        self,
        findings: Sequence[
            ReviewFinding
            | Mapping[str, Any]
        ],
    ) -> list[ReviewFinding]:
        result: list[
            ReviewFinding
        ] = []

        for index, item in enumerate(
            findings
        ):
            if isinstance(
                item,
                ReviewFinding,
            ):
                result.append(item)
                continue

            if not isinstance(
                item,
                Mapping,
            ):
                continue

            result.append(
                ReviewFinding(
                    finding_id=str(
                        item.get(
                            "finding_id"
                        )
                        or self._stable_id(
                            index,
                            item,
                        )
                    ),
                    finding_type=(
                        self._coerce_finding_type(
                            item.get(
                                "finding_type"
                            )
                        )
                    ),
                    title=str(
                        item.get(
                            "title"
                        )
                        or "Legal review finding"
                    ),
                    description=str(
                        item.get(
                            "description"
                        )
                        or ""
                    ),
                    severity=(
                        self._coerce_severity(
                            item.get(
                                "severity"
                            )
                        )
                    ),
                    risk_level=(
                        self._coerce_risk(
                            item.get(
                                "risk_level"
                            )
                        )
                    ),
                    confidence=self._float(
                        item.get(
                            "confidence"
                        ),
                        default=0.0,
                    ),
                    requires_human_review=bool(
                        item.get(
                            "requires_human_review",
                            True,
                        )
                    ),
                    evidence_ids=(
                        self._string_list(
                            item.get(
                                "evidence_ids"
                            )
                            or []
                        )
                    ),
                    source_agent=(
                        self._optional_str(
                            item.get(
                                "source_agent"
                            )
                        )
                    ),
                    source_entity_id=(
                        self._optional_str(
                            item.get(
                                "source_entity_id"
                            )
                        )
                    ),
                    recommended_action=(
                        self._optional_str(
                            item.get(
                                "recommended_action"
                            )
                        )
                    ),
                    metadata=dict(
                        item.get(
                            "metadata"
                        )
                        or {}
                    ),
                )
            )

        return result[
            : self.config.max_findings
        ]

    # ========================================================================
    # AUTHORIZATION
    # ========================================================================

    def _authorize_document(
        self,
        document: ReviewDocument,
        tenant_id: str,
    ) -> None:
        assert_tenant_access(
            tenant_id,
            resource_type="document",
            resource_id=document.document_id,
        )

    # ========================================================================
    # SECURITY
    # ========================================================================

    def _security_scan(
        self,
        document: ReviewDocument,
        findings: Sequence[
            ReviewFinding
        ],
    ) -> dict[str, Any]:
        if not self.config.enable_security_scanning:
            return {
                "prompt_injection_detected": False,
                "content_filter_decision": (
                    "disabled"
                ),
            }

        texts: list[str] = []

        if document.text:
            texts.append(
                document.text
            )

        texts.extend(
            item.description
            for item in findings
            if item.description
        )

        texts.extend(
            item.title
            for item in findings
            if item.title
        )

        injection_detected = False
        content_decisions: list[str] = []

        for text in texts:
            injection = (
                self.injection_detector.detect(
                    text,
                    source=InputSource.DOCUMENT,
                )
            )

            injection_findings = getattr(
                injection,
                "findings",
                None,
            )

            if injection_findings:
                injection_detected = True

            content = (
                self.content_filter.scan(
                    text
                )
            )

            decision = getattr(
                content,
                "decision",
                None,
            )

            content_decisions.append(
                (
                    decision.value
                    if hasattr(
                        decision,
                        "value",
                    )
                    else str(decision)
                )
            )

        return {
            "prompt_injection_detected": (
                injection_detected
            ),
            "content_filter_decisions": (
                content_decisions
            ),
        }

    def _security_refusal(
        self,
        document: ReviewDocument,
        tenant_id: str,
        started_at: datetime,
    ) -> ReviewAnalysis:
        result = ReviewAnalysis(
            document_id=document.document_id,
            tenant_id=tenant_id,
            status=self._status_pending(),
            findings=[
                ReviewFinding(
                    finding_id=self._stable_id(
                        document.document_id,
                        "security_refusal",
                    ),
                    finding_type=(
                        ReviewFindingType.SECURITY
                    ),
                    title=(
                        "Security review required"
                    ),
                    description=(
                        "Automated review was stopped because "
                        "potentially malicious instructional "
                        "content was detected."
                    ),
                    severity=ReviewSeverity.CRITICAL,
                    risk_level=RiskLevel.HIGH,
                    confidence=0.95,
                    requires_human_review=True,
                    recommended_action=(
                        "Manually inspect the source before "
                        "continuing automated review."
                    ),
                )
            ],
            review_queue=[
                ReviewItem(
                    review_item_id=self._stable_id(
                        document.document_id,
                        "security_queue",
                    ),
                    finding_id=self._stable_id(
                        document.document_id,
                        "security_refusal",
                    ),
                    priority=(
                        self._priority_critical()
                    ),
                    severity=(
                        ReviewSeverity.CRITICAL
                    ),
                    title=(
                        "Security review required"
                    ),
                    rationale=(
                        "Potential prompt injection or "
                        "malicious document instructions."
                    ),
                    recommended_action=(
                        "Perform manual security inspection."
                    ),
                    estimated_effort="high",
                )
            ],
            critical_count=1,
            high_priority_count=1,
            unresolved_count=1,
            overall_risk=RiskLevel.HIGH,
            confidence=0.0,
            recommendations=[
                (
                    "Do not treat embedded document instructions "
                    "as trusted system instructions."
                )
            ],
            metadata={
                "agent": self.AGENT_NAME,
                "security_refusal": True,
                "binding_decision": False,
            },
            created_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        self._record_telemetry(
            result,
            started_at,
        )

        return result

    # ========================================================================
    # EVIDENCE
    # ========================================================================

    def _extract_evidence(
        self,
        findings: Sequence[
            ReviewFinding
        ],
    ) -> list[ReviewEvidence]:
        evidence: list[
            ReviewEvidence
        ] = []

        for finding in findings:
            evidence_text = (
                finding.metadata.get(
                    "evidence_text"
                )
                or finding.metadata.get(
                    "source_text"
                )
                or ""
            )

            if not evidence_text:
                continue

            evidence.append(
                ReviewEvidence(
                    evidence_id=(
                        finding.evidence_ids[0]
                        if finding.evidence_ids
                        else self._stable_id(
                            finding.finding_id,
                            evidence_text,
                        )
                    ),
                    document_id=str(
                        finding.metadata.get(
                            "document_id"
                        )
                        or ""
                    ),
                    text=str(
                        evidence_text
                    ),
                    source_type=str(
                        finding.metadata.get(
                            "source_type"
                        )
                        or "document"
                    ),
                    page=self._optional_int(
                        finding.metadata.get(
                            "page"
                        )
                    ),
                    section=self._optional_str(
                        finding.metadata.get(
                            "section"
                        )
                    ),
                    clause_id=self._optional_str(
                        finding.metadata.get(
                            "clause_id"
                        )
                    ),
                    confidence=(
                        finding.confidence
                    ),
                    citation=self._optional_str(
                        finding.metadata.get(
                            "citation"
                        )
                    ),
                )
            )

        return evidence

    def _validate_evidence(
        self,
        evidence: Sequence[
            ReviewEvidence
        ],
    ) -> list[ReviewEvidence]:
        result: list[
            ReviewEvidence
        ] = []

        for item in evidence:
            if not item.text.strip():
                continue

            if not item.evidence_id:
                continue

            item.confidence = max(
                0.0,
                min(
                    1.0,
                    item.confidence,
                ),
            )

            result.append(item)

        return result

    # ========================================================================
    # CONTRADICTION DETECTION
    # ========================================================================

    def _detect_conflicts(
        self,
        findings: Sequence[
            ReviewFinding
        ],
    ) -> list[ReviewConflict]:
        conflicts: list[
            ReviewConflict
        ] = []

        grouped: dict[
            str,
            list[ReviewFinding],
        ] = {}

        for finding in findings:
            entity = (
                finding.source_entity_id
                or finding.metadata.get(
                    "entity_id"
                )
                or finding.finding_id
            )

            grouped.setdefault(
                str(entity),
                [],
            ).append(finding)

        for entity, items in grouped.items():
            for left_index, left in enumerate(
                items
            ):
                for right in items[
                    left_index + 1 :
                ]:
                    if self._is_contradictory(
                        left,
                        right,
                    ):
                        conflicts.append(
                            ReviewConflict(
                                conflict_id=(
                                    self._stable_id(
                                        left.finding_id,
                                        right.finding_id,
                                    )
                                ),
                                description=(
                                    f"Potential contradiction "
                                    f"between '{left.title}' and "
                                    f"'{right.title}'."
                                ),
                                finding_ids=[
                                    left.finding_id,
                                    right.finding_id,
                                ],
                                severity=(
                                    ReviewSeverity.HIGH
                                ),
                            )
                        )

        return conflicts

    @staticmethod
    def _is_contradictory(
        left: ReviewFinding,
        right: ReviewFinding,
    ) -> bool:
        if (
            left.finding_type
            != right.finding_type
        ):
            return False

        left_text = (
            left.description.lower()
        )

        right_text = (
            right.description.lower()
        )

        negative_terms = (
            "not",
            "no",
            "prohibited",
            "forbidden",
            "does not",
            "cannot",
            "never",
        )

        left_negative = any(
            term in left_text
            for term in negative_terms
        )

        right_negative = any(
            term in right_text
            for term in negative_terms
        )

        return (
            left_negative
            != right_negative
        )

    # ========================================================================
    # FINDING ANALYSIS
    # ========================================================================

    def _analyze_findings(
        self,
        findings: Sequence[
            ReviewFinding
        ],
        conflicts: Sequence[
            ReviewConflict
        ],
        *,
        mode: ReviewAgentMode,
    ) -> list[ReviewFinding]:
        conflict_ids = {
            finding_id
            for conflict in conflicts
            for finding_id in conflict.finding_ids
        }

        result: list[
            ReviewFinding
        ] = []

        for finding in findings:
            if (
                finding.finding_id
                in conflict_ids
            ):
                finding.requires_human_review = True
                finding.severity = (
                    ReviewSeverity.HIGH
                )
                finding.recommended_action = (
                    finding.recommended_action
                    or "Resolve conflicting findings."
                )

            if (
                finding.confidence
                < self.config.minimum_confidence
                and self.config.low_confidence_requires_review
            ):
                finding.requires_human_review = True
                finding.metadata[
                    "low_confidence"
                ] = True

            if (
                finding.risk_level
                in (
                    RiskLevel.HIGH,
                    RiskLevel.CRITICAL,
                )
                and self.config.critical_risk_requires_review
            ):
                finding.requires_human_review = True

            if mode == ReviewAgentMode.QUALITY:
                if finding.confidence >= (
                    self.config.minimum_confidence
                ):
                    finding.metadata[
                        "quality_pass"
                    ] = True
                else:
                    finding.metadata[
                        "quality_pass"
                    ] = False

            result.append(finding)

        return result

    # ========================================================================
    # REVIEW QUEUE
    # ========================================================================

    def _build_review_queue(
        self,
        findings: Sequence[
            ReviewFinding
        ],
        conflicts: Sequence[
            ReviewConflict
        ],
    ) -> list[ReviewItem]:
        result: list[
            ReviewItem
        ] = []

        conflict_finding_ids = {
            finding_id
            for conflict in conflicts
            for finding_id in conflict.finding_ids
        }

        for finding in findings:
            if not finding.requires_human_review:
                continue

            priority = (
                self._calculate_priority(
                    finding,
                    finding.finding_id
                    in conflict_finding_ids,
                )
            )

            rationale_parts: list[str] = []

            if finding.risk_level in (
                RiskLevel.HIGH,
                RiskLevel.CRITICAL,
            ):
                rationale_parts.append(
                    "High legal or operational risk."
                )

            if finding.confidence < (
                self.config.minimum_confidence
            ):
                rationale_parts.append(
                    "Low extraction confidence."
                )

            if finding.finding_id in (
                conflict_finding_ids
            ):
                rationale_parts.append(
                    "Conflicting evidence or findings."
                )

            if not rationale_parts:
                rationale_parts.append(
                    "Finding requires human validation."
                )

            action = (
                finding.recommended_action
                or self._default_review_action(
                    finding
                )
            )

            result.append(
                ReviewItem(
                    review_item_id=self._stable_id(
                        finding.finding_id,
                        "review_item",
                    ),
                    finding_id=finding.finding_id,
                    priority=priority,
                    severity=finding.severity,
                    title=finding.title,
                    rationale=" ".join(
                        rationale_parts
                    ),
                    recommended_action=action,
                    evidence_ids=(
                        finding.evidence_ids
                    ),
                    estimated_effort=(
                        self._estimate_effort(
                            finding
                        )
                    ),
                )
            )

        return sorted(
            result,
            key=self._priority_sort_key,
        )

    def _calculate_priority(
        self,
        finding: ReviewFinding,
        has_conflict: bool,
    ) -> ReviewPriority:
        if (
            finding.risk_level
            == RiskLevel.CRITICAL
            or finding.severity
            == ReviewSeverity.CRITICAL
            or has_conflict
        ):
            return self._priority_critical()

        if (
            finding.risk_level
            == RiskLevel.HIGH
            or finding.severity
            == ReviewSeverity.HIGH
        ):
            return self._priority_high()

        if (
            finding.risk_level
            == RiskLevel.MEDIUM
            or finding.severity
            == ReviewSeverity.MEDIUM
        ):
            return self._priority_medium()

        return self._priority_low()

    @staticmethod
    def _priority_sort_key(
        item: ReviewItem,
    ) -> tuple[int, int]:
        order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        return (
            order.get(
                item.priority.value,
                99,
            ),
            -int(
                item.severity
                in (
                    ReviewSeverity.CRITICAL,
                    ReviewSeverity.HIGH,
                )
            ),
        )

    @staticmethod
    def _default_review_action(
        finding: ReviewFinding,
    ) -> str:
        if finding.finding_type == (
            ReviewFindingType.CITATION
        ):
            return (
                "Verify the cited source and supporting passage."
            )

        if finding.finding_type == (
            ReviewFindingType.CONTRADICTION
        ):
            return (
                "Resolve conflicting findings using the source contract."
            )

        if finding.finding_type == (
            ReviewFindingType.SECURITY
        ):
            return (
                "Perform a security review before accepting the finding."
            )

        if finding.risk_level in (
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ):
            return (
                "Perform authorized legal review."
            )

        return (
            "Validate the finding against the source document."
        )

    @staticmethod
    def _estimate_effort(
        finding: ReviewFinding,
    ) -> str:
        if (
            finding.risk_level
            == RiskLevel.CRITICAL
        ):
            return "high"

        if (
            finding.risk_level
            == RiskLevel.HIGH
        ):
            return "medium"

        return "low"

    # ========================================================================
    # DECISION PROPOSAL
    # ========================================================================

    def _build_decision_proposal(
        self,
        findings: Sequence[
            ReviewFinding
        ],
        conflicts: Sequence[
            ReviewConflict
        ],
    ) -> ReviewDecisionProposal:
        if conflicts:
            return ReviewDecisionProposal(
                decision=(
                    ReviewDecision.ESCALATE
                ),
                rationale=(
                    "Conflicting findings require human "
                    "resolution before approval."
                ),
                confidence=0.90,
                required_reviewer_role="legal_reviewer",
                conditions=[
                    "Resolve all material conflicts.",
                    "Verify supporting evidence.",
                ],
                evidence_ids=[
                    evidence_id
                    for finding in findings
                    for evidence_id
                    in finding.evidence_ids
                ],
                is_binding=False,
            )

        critical = [
            item
            for item in findings
            if item.risk_level
            == RiskLevel.CRITICAL
        ]

        if critical:
            return ReviewDecisionProposal(
                decision=(
                    ReviewDecision.ESCALATE
                ),
                rationale=(
                    "Critical-risk findings require "
                    "authorized human review."
                ),
                confidence=0.90,
                required_reviewer_role="legal_reviewer",
                conditions=[
                    "Review critical-risk findings.",
                    "Verify source evidence.",
                ],
                evidence_ids=[
                    evidence_id
                    for item in critical
                    for evidence_id
                    in item.evidence_ids
                ],
                is_binding=False,
            )

        low_confidence = [
            item
            for item in findings
            if item.confidence
            < self.config.minimum_confidence
        ]

        if low_confidence:
            return ReviewDecisionProposal(
                decision=(
                    ReviewDecision.REVISE
                ),
                rationale=(
                    "One or more findings have insufficient "
                    "confidence for autonomous acceptance."
                ),
                confidence=0.85,
                required_reviewer_role="reviewer",
                conditions=[
                    "Verify low-confidence findings."
                ],
                evidence_ids=[
                    evidence_id
                    for item in low_confidence
                    for evidence_id
                    in item.evidence_ids
                ],
                is_binding=False,
            )

        return ReviewDecisionProposal(
            decision=ReviewDecision.PENDING,
            rationale=(
                "Findings passed automated triage but remain "
                "subject to authorized human review."
            ),
            confidence=0.75,
            required_reviewer_role="reviewer",
            conditions=[
                "Confirm findings against source documents."
            ],
            evidence_ids=[
                evidence_id
                for item in findings
                for evidence_id
                in item.evidence_ids
            ],
            is_binding=False,
        )

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================

    def _build_recommendations(
        self,
        findings: Sequence[
            ReviewFinding
        ],
        conflicts: Sequence[
            ReviewConflict
        ],
    ) -> list[str]:
        recommendations: list[str] = []

        if conflicts:
            recommendations.append(
                (
                    "Resolve conflicting findings before "
                    "finalizing the review."
                )
            )

        if any(
            item.risk_level
            == RiskLevel.CRITICAL
            for item in findings
        ):
            recommendations.append(
                (
                    "Escalate critical-risk findings to an "
                    "authorized legal reviewer."
                )
            )

        if any(
            item.confidence
            < self.config.minimum_confidence
            for item in findings
        ):
            recommendations.append(
                (
                    "Validate low-confidence extraction and "
                    "retrieval results against source evidence."
                )
            )

        if any(
            item.finding_type
            == ReviewFindingType.CITATION
            for item in findings
        ):
            recommendations.append(
                (
                    "Verify citations before using findings "
                    "for contractual decisions."
                )
            )

        if not recommendations:
            recommendations.append(
                (
                    "Complete the standard human-review workflow "
                    "before final approval."
                )
            )

        return self._unique_strings(
            recommendations
        )

    # ========================================================================
    # STATUS
    # ========================================================================

    def _calculate_status(
        self,
        findings: Sequence[
            ReviewFinding
        ],
        conflicts: Sequence[
            ReviewConflict
        ],
    ) -> ReviewStatus:
        if conflicts:
            return self._status_pending()

        if any(
            item.requires_human_review
            for item in findings
        ):
            return self._status_pending()

        return self._status_pending()

    # ========================================================================
    # RISK / CONFIDENCE
    # ========================================================================

    @staticmethod
    def _overall_risk(
        findings: Sequence[
            ReviewFinding
        ],
    ) -> RiskLevel:
        ranking = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }

        highest = max(
            (
                ranking.get(
                    item.risk_level,
                    1,
                )
                for item in findings
            ),
            default=1,
        )

        for level, score in ranking.items():
            if score == highest:
                return level

        return RiskLevel.LOW

    @staticmethod
    def _overall_confidence(
        findings: Sequence[
            ReviewFinding
        ],
    ) -> float:
        if not findings:
            return 0.0

        return round(
            sum(
                max(
                    0.0,
                    min(
                        1.0,
                        item.confidence,
                    ),
                )
                for item in findings
            )
            / len(findings),
            4,
        )

    # ========================================================================
    # GUARDRAILS
    # ========================================================================

    def _apply_output_guardrails(
        self,
        result: ReviewAnalysis,
    ) -> None:
        if not self.config.enable_output_guardrails:
            return

        try:
            if hasattr(
                self.output_guardrails,
                "validate",
            ):
                self.output_guardrails.validate(
                    result.to_dict()
                )

            result.metadata[
                "output_guardrails_validated"
            ] = True

        except Exception:
            logger.warning(
                "review_output_guardrails_failed",
                exc_info=True,
            )

            result.metadata[
                "output_guardrails_validated"
            ] = False

    # ========================================================================
    # TELEMETRY
    # ========================================================================

    def _record_telemetry(
        self,
        result: ReviewAnalysis,
        started_at: datetime,
    ) -> None:
        duration_ms = (
            datetime.now(
                timezone.utc
            ) - started_at
        ).total_seconds() * 1000

        try:
            telemetry.increment(
                "agent.review.executions",
                1,
            )

            telemetry.increment(
                "agent.review.findings",
                len(result.findings),
            )

            telemetry.increment(
                "agent.review.queue_items",
                len(result.review_queue),
            )

            telemetry.increment(
                "agent.review.conflicts",
                len(result.conflicts),
            )

            telemetry.increment(
                "agent.review.critical",
                result.critical_count,
            )

            telemetry.observe(
                "agent.review.confidence",
                result.confidence,
            )

            telemetry.observe(
                "agent.review.duration_ms",
                duration_ms,
            )

            telemetry.increment(
                (
                    "agent.review."
                    f"risk.{result.overall_risk.value}"
                ),
                1,
            )

        except Exception:
            logger.debug(
                "review_telemetry_failed",
                exc_info=True,
            )

    # ========================================================================
    # ENUM COMPATIBILITY HELPERS
    # ========================================================================

    @staticmethod
    def _priority_critical() -> ReviewPriority:
        return ReviewAgent._enum_value(
            ReviewPriority,
            (
                "CRITICAL",
                "URGENT",
                "HIGH",
            ),
        )

    @staticmethod
    def _priority_high() -> ReviewPriority:
        return ReviewAgent._enum_value(
            ReviewPriority,
            (
                "HIGH",
                "URGENT",
                "CRITICAL",
            ),
        )

    @staticmethod
    def _priority_medium() -> ReviewPriority:
        return ReviewAgent._enum_value(
            ReviewPriority,
            (
                "MEDIUM",
                "NORMAL",
            ),
        )

    @staticmethod
    def _priority_low() -> ReviewPriority:
        return ReviewAgent._enum_value(
            ReviewPriority,
            (
                "LOW",
                "NORMAL",
                "MEDIUM",
            ),
        )

    @staticmethod
    def _status_pending() -> ReviewStatus:
        return ReviewAgent._enum_value(
            ReviewStatus,
            (
                "PENDING",
                "OPEN",
                "IN_REVIEW",
            ),
        )

    @staticmethod
    def _enum_value(
        enum_cls: Any,
        names: Sequence[str],
    ) -> Any:
        for name in names:
            if hasattr(
                enum_cls,
                name,
            ):
                return getattr(
                    enum_cls,
                    name,
                )

        members = list(
            enum_cls
        )

        if members:
            return members[0]

        raise ValidationError(
            f"{enum_cls.__name__} has no usable members"
        )

    # ========================================================================
    # GENERIC HELPERS
    # ========================================================================

    @staticmethod
    def _coerce_finding_type(
        value: Any,
    ) -> ReviewFindingType:
        if isinstance(
            value,
            ReviewFindingType,
        ):
            return value

        raw = str(
            value or ""
        ).lower().strip()

        for item in ReviewFindingType:
            if (
                raw == item.value
                or raw == item.name.lower()
            ):
                return item

        return ReviewFindingType.DATA_QUALITY

    @staticmethod
    def _coerce_severity(
        value: Any,
    ) -> ReviewSeverity:
        if isinstance(
            value,
            ReviewSeverity,
        ):
            return value

        raw = str(
            value or ""
        ).lower().strip()

        for item in ReviewSeverity:
            if (
                raw == item.value
                or raw == item.name.lower()
            ):
                return item

        return ReviewSeverity.MEDIUM

    @staticmethod
    def _coerce_risk(
        value: Any,
    ) -> RiskLevel:
        if isinstance(
            value,
            RiskLevel,
        ):
            return value

        raw = str(
            value or ""
        ).lower().strip()

        for item in RiskLevel:
            if (
                raw == item.value
                or raw == item.name.lower()
            ):
                return item

        return RiskLevel.MEDIUM

    @staticmethod
    def _optional_str(
        value: Any,
    ) -> Optional[str]:
        if value is None:
            return None

        value = str(
            value
        ).strip()

        return value or None

    @staticmethod
    def _optional_int(
        value: Any,
    ) -> Optional[int]:
        if value is None:
            return None

        try:
            return int(value)
        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _float(
        value: Any,
        *,
        default: float = 0.0,
    ) -> float:
        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _string_list(
        value: Any,
    ) -> list[str]:
        if isinstance(
            value,
            (str, bytes),
        ):
            return []

        if not isinstance(
            value,
            Sequence,
        ):
            return []

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    @staticmethod
    def _stable_id(
        *parts: Any,
    ) -> str:
        try:
            serialized = json.dumps(
                parts,
                sort_keys=True,
                default=str,
            )
        except TypeError:
            serialized = repr(parts)

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()[:24]

    @staticmethod
    def _unique_strings(
        values: Sequence[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            value = str(
                value
            ).strip()

            if (
                not value
                or value in seen
            ):
                continue

            seen.add(value)
            result.append(value)

        return result


# ============================================================================
# SINGLETON
# ============================================================================


_review_agent: Optional[
    ReviewAgent
] = None


def get_review_agent(
    *,
    config: Optional[
        ReviewAgentConfig
    ] = None,
    llm: Optional[
        Callable[..., Any]
    ] = None,
    review_service: Optional[
        Any
    ] = None,
) -> ReviewAgent:
    """Return the process-level review agent."""

    global _review_agent

    if _review_agent is None:
        _review_agent = ReviewAgent(
            config=config,
            llm=llm,
            review_service=review_service,
        )

    return _review_agent


# ============================================================================
# FUNCTIONAL API
# ============================================================================


def analyze_review(
    document: (
        ReviewDocument
        | Mapping[str, Any]
    ),
    findings: Optional[
        Sequence[
            ReviewFinding
            | Mapping[str, Any]
        ]
    ] = None,
    *,
    mode: ReviewAgentMode = (
        ReviewAgentMode.FULL
    ),
    tenant_id: Optional[str] = None,
) -> ReviewAnalysis:
    """Analyze findings and create a review packet."""

    return get_review_agent().analyze(
        document,
        findings,
        mode=mode,
        tenant_id=tenant_id,
    )


def triage_review(
    document: (
        ReviewDocument
        | Mapping[str, Any]
    ),
    findings: Sequence[
        ReviewFinding
        | Mapping[str, Any]
    ],
    *,
    tenant_id: Optional[str] = None,
) -> list[ReviewItem]:
    """Return prioritized human-review items."""

    result = analyze_review(
        document,
        findings,
        mode=ReviewAgentMode.TRIAGE,
        tenant_id=tenant_id,
    )

    return result.review_queue


def review_risk(
    document: (
        ReviewDocument
        | Mapping[str, Any]
    ),
    findings: Sequence[
        ReviewFinding
        | Mapping[str, Any]
    ],
    *,
    tenant_id: Optional[str] = None,
) -> ReviewAnalysis:
    """Return risk-focused review analysis."""

    return analyze_review(
        document,
        findings,
        mode=ReviewAgentMode.RISK,
        tenant_id=tenant_id,
    )


__all__ = [
    "ReviewAgentMode",
    "ReviewDecision",
    "ReviewFindingType",
    "ReviewSeverity",
    "ReviewDocument",
    "ReviewEvidence",
    "ReviewFinding",
    "ReviewConflict",
    "ReviewItem",
    "ReviewDecisionProposal",
    "ReviewAnalysis",
    "ReviewAgentConfig",
    "ReviewAgent",
    "get_review_agent",
    "analyze_review",
    "triage_review",
    "review_risk",
]