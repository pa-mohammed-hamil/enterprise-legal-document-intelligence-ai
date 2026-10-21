# backend/rag/query/query_classifier.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence
import math
import re
import time


class QueryType(str, Enum):
    """High-level intent used by the RAG query router."""

    LEGAL_QA = "legal_qa"
    DOCUMENT_LOOKUP = "document_lookup"
    CLAUSE_LOOKUP = "clause_lookup"
    DEFINITION = "definition"
    OBLIGATION = "obligation"
    RISK = "risk"
    EXPIRY = "expiry"
    RENEWAL = "renewal"
    TERMINATION = "termination"
    PAYMENT = "payment"
    PRICING = "pricing"
    LIABILITY = "liability"
    INDEMNIFICATION = "indemnification"
    CONFIDENTIALITY = "confidentiality"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DATA_PROTECTION = "data_protection"
    PRIVACY = "privacy"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    AUDIT = "audit"
    INSURANCE = "insurance"
    SERVICE_LEVEL = "service_level"
    GOVERNING_LAW = "governing_law"
    DISPUTE_RESOLUTION = "dispute_resolution"
    FORCE_MAJEURE = "force_majeure"
    ASSIGNMENT = "assignment"
    WARRANTY = "warranty"
    NOTICE = "notice"
    CONTRACT_COMPARISON = "contract_comparison"
    DOCUMENT_SUMMARY = "document_summary"
    TABLE_LOOKUP = "table_lookup"
    METADATA_LOOKUP = "metadata_lookup"
    REVIEW = "review"
    SEARCH = "search"
    GENERAL = "general"
    UNKNOWN = "unknown"


class QueryScope(str, Enum):
    CURRENT_DOCUMENT = "current_document"
    MULTIPLE_DOCUMENTS = "multiple_documents"
    ORGANIZATION = "organization"
    GLOBAL = "global"
    UNKNOWN = "unknown"


class QueryComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class ClassificationMethod(str, Enum):
    RULE = "rule"
    MODEL = "model"
    HYBRID = "hybrid"
    FALLBACK = "fallback"


@dataclass(slots=True)
class QuerySignal:
    query_type: QueryType
    score: float

    source: str
    matched_text: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "score": self.score,
            "source": self.source,
            "matched_text": self.matched_text,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class QueryClassification:
    query_type: QueryType

    confidence: float

    scope: QueryScope

    complexity: QueryComplexity

    method: ClassificationMethod

    secondary_types: list[
        QueryType
    ] = field(default_factory=list)

    signals: list[
        QuerySignal
    ] = field(default_factory=list)

    requires_retrieval: bool = True
    requires_reranking: bool = True
    requires_generation: bool = True

    requires_multi_query: bool = False
    requires_comparison: bool = False
    requires_table_retrieval: bool = False
    requires_permission_filter: bool = True

    document_ids: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "scope": self.scope.value,
            "complexity": self.complexity.value,
            "method": self.method.value,
            "secondary_types": [
                item.value
                for item in self.secondary_types
            ],
            "signals": [
                signal.to_dict()
                for signal in self.signals
            ],
            "requires_retrieval":
                self.requires_retrieval,
            "requires_reranking":
                self.requires_reranking,
            "requires_generation":
                self.requires_generation,
            "requires_multi_query":
                self.requires_multi_query,
            "requires_comparison":
                self.requires_comparison,
            "requires_table_retrieval":
                self.requires_table_retrieval,
            "requires_permission_filter":
                self.requires_permission_filter,
            "document_ids":
                self.document_ids,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class QueryClassifierConfig:
    minimum_confidence: float = 0.35

    secondary_threshold: float = 0.25

    maximum_secondary_types: int = 3

    enable_model_classifier: bool = True
    enable_rule_classifier: bool = True

    default_type: QueryType = (
        QueryType.LEGAL_QA
    )

    default_scope: QueryScope = (
        QueryScope.CURRENT_DOCUMENT
    )

    enable_permission_filter: bool = True

    long_query_word_threshold: int = 40
    complex_query_word_threshold: int = 80

    comparison_min_documents: int = 2

    normalize_query: bool = True


class QueryClassificationError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "QUERY_CLASSIFICATION_ERROR",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause


class QueryClassifier:
    """
    Hybrid deterministic + optional model query classifier.

    The classifier does NOT answer legal questions. It only determines
    retrieval/routing intent.

    Typical flow:

        User Query
            ↓
        QueryClassifier
            ↓
        QueryRewriter
            ↓
        QueryRouter
            ↓
        BM25 + Vector Search
            ↓
        Metadata / Permission Filter
            ↓
        Reranker
            ↓
        Legal LLM
    """

    INTENT_PATTERNS: dict[
        QueryType,
        tuple[str, ...],
    ] = {
        QueryType.DEFINITION: (
            r"\bwhat\s+does\b.*\bmean\b",
            r"\bdefine\b",
            r"\bdefinition\s+of\b",
            r"\bmeaning\s+of\b",
            r"\bdefined\s+(?:as|term)\b",
        ),

        QueryType.OBLIGATION: (
            r"\bobligation(?:s)?\b",
            r"\brequired\s+to\b",
            r"\bmust\b",
            r"\bshall\b",
            r"\bresponsibilit(?:y|ies)\b",
            r"\bdut(?:y|ies)\b",
            r"\bwhat\s+(?:do|does)\b.*"
            r"\bhave\s+to\b",
        ),

        QueryType.RISK: (
            r"\brisk(?:s)?\b",
            r"\brisky\b",
            r"\bhigh[- ]risk\b",
            r"\blegal\s+risk\b",
            r"\brisk\s+exposure\b",
            r"\bproblematic\s+clause\b",
            r"\bred\s+flag(?:s)?\b",
            r"\bunfavourable\b",
            r"\bunfavorable\b",
        ),

        QueryType.EXPIRY: (
            r"\bexpir(?:y|e|es|ed|ation)\b",
            r"\bend\s+date\b",
            r"\bcontract\s+end\b",
            r"\bexpiration\s+date\b",
            r"\bwhen\s+does\b.*\bend\b",
        ),

        QueryType.RENEWAL: (
            r"\brenew(?:al|s|ed|ing)?\b",
            r"\bauto[- ]?renew(?:al)?\b",
            r"\bautomatic\s+renewal\b",
            r"\brenewal\s+date\b",
            r"\brenewal\s+term\b",
        ),

        QueryType.TERMINATION: (
            r"\bterminat(?:e|es|ed|ion)\b",
            r"\bcancel(?:lation|led|ing)?\b",
            r"\bend\s+this\s+agreement\b",
            r"\bterminate\s+for\s+cause\b",
            r"\bterminate\s+for\s+convenience\b",
        ),

        QueryType.PAYMENT: (
            r"\bpayment(?:s)?\b",
            r"\bpayment\s+term(?:s)?\b",
            r"\binvoice(?:s|d)?\b",
            r"\blate\s+fee(?:s)?\b",
            r"\bpayable\b",
            r"\bnet\s+\d+\b",
        ),

        QueryType.PRICING: (
            r"\bpricing\b",
            r"\bprice(?:s)?\b",
            r"\bcost(?:s)?\b",
            r"\bfee(?:s)?\b",
            r"\brate(?:s)?\b",
            r"\bcharges?\b",
        ),

        QueryType.LIABILITY: (
            r"\bliabilit(?:y|ies)\b",
            r"\bliability\s+cap\b",
            r"\blimitation\s+of\s+liability\b",
            r"\bunlimited\s+liability\b",
            r"\bconsequential\s+damages\b",
        ),

        QueryType.INDEMNIFICATION: (
            r"\bindemnif(?:y|ies|ied|ication)\b",
            r"\bindemnity\b",
            r"\bhold\s+harmless\b",
        ),

        QueryType.CONFIDENTIALITY: (
            r"\bconfidential(?:ity)?\b",
            r"\bnon[- ]disclosure\b",
            r"\bnda\b",
            r"\bconfidential\s+information\b",
        ),

        QueryType.INTELLECTUAL_PROPERTY: (
            r"\bintellectual\s+property\b",
            r"\b(?:^|\s)ip(?:\s|$)\b",
            r"\bcopyright\b",
            r"\bpatent(?:s)?\b",
            r"\btrademark(?:s)?\b",
            r"\bownership\s+of\b.*"
            r"\b(?:code|software|work|content)\b",
        ),

        QueryType.DATA_PROTECTION: (
            r"\bdata\s+protection\b",
            r"\bpersonal\s+data\b",
            r"\bdata\s+processing\b",
            r"\bdpa\b",
            r"\bgdpr\b",
            r"\bdata\s+processor\b",
            r"\bdata\s+controller\b",
        ),

        QueryType.PRIVACY: (
            r"\bprivacy\b",
            r"\bpersonal\s+information\b",
            r"\bprivacy\s+policy\b",
            r"\bprivacy\s+rights\b",
        ),

        QueryType.SECURITY: (
            r"\bsecurity\b",
            r"\bcybersecurity\b",
            r"\bsecurity\s+incident\b",
            r"\bdata\s+breach\b",
            r"\bbreach\s+notification\b",
            r"\bencryption\b",
            r"\baccess\s+control\b",
        ),

        QueryType.COMPLIANCE: (
            r"\bcompliance\b",
            r"\bcomply\b",
            r"\bregulatory\b",
            r"\bregulation(?:s)?\b",
            r"\bstatutory\b",
            r"\blaw(?:s)?\s+and\s+regulations\b",
        ),

        QueryType.AUDIT: (
            r"\baudit(?:s|ing)?\b",
            r"\baudit\s+right(?:s)?\b",
            r"\binspection\s+right(?:s)?\b",
            r"\brecords\s+inspection\b",
        ),

        QueryType.INSURANCE: (
            r"\binsurance\b",
            r"\bcoverage\b",
            r"\bpolicy\s+limit(?:s)?\b",
            r"\bcertificate\s+of\s+insurance\b",
        ),

        QueryType.SERVICE_LEVEL: (
            r"\bservice\s+level\b",
            r"\bsla\b",
            r"\buptime\b",
            r"\bavailability\s+target\b",
            r"\bservice\s+credit(?:s)?\b",
            r"\bresponse\s+time\b",
        ),

        QueryType.GOVERNING_LAW: (
            r"\bgoverning\s+law\b",
            r"\bjurisdiction\b",
            r"\bchoice\s+of\s+law\b",
            r"\bwhich\s+law\s+governs\b",
        ),

        QueryType.DISPUTE_RESOLUTION: (
            r"\bdispute\s+resolution\b",
            r"\barbitration\b",
            r"\bmediation\b",
            r"\bvenue\b",
            r"\blitigation\b",
        ),

        QueryType.FORCE_MAJEURE: (
            r"\bforce\s+majeure\b",
            r"\bact(?:s)?\s+of\s+god\b",
            r"\bunforeseeable\s+event(?:s)?\b",
        ),

        QueryType.ASSIGNMENT: (
            r"\bassignment\b",
            r"\bassign\s+this\s+agreement\b",
            r"\btransfer\s+this\s+agreement\b",
        ),

        QueryType.WARRANTY: (
            r"\bwarrant(?:y|ies)\b",
            r"\brepresentation(?:s)?\s+and\s+"
            r"warrant(?:y|ies)\b",
            r"\bwarrants?\s+that\b",
        ),

        QueryType.NOTICE: (
            r"\bnotice(?:s)?\b",
            r"\bnotice\s+period\b",
            r"\bwritten\s+notice\b",
            r"\bnotification\s+period\b",
        ),

        QueryType.CONTRACT_COMPARISON: (
            r"\bcompare\b",
            r"\bcomparison\b",
            r"\bdifference(?:s)?\b",
            r"\bversus\b",
            r"\bvs\.?\b",
            r"\bwhich\s+contract\b",
            r"\bbetween\s+these\s+contracts\b",
        ),

        QueryType.DOCUMENT_SUMMARY: (
            r"\bsummar(?:y|ize|ise)\b",
            r"\boverview\b",
            r"\bkey\s+points\b",
            r"\bmain\s+terms\b",
            r"\bimportant\s+terms\b",
            r"\bexecutive\s+summary\b",
        ),

        QueryType.CLAUSE_LOOKUP: (
            r"\bclause\b",
            r"\bsection\s+\d+",
            r"\barticle\s+\d+",
            r"\bprovision\b",
        ),

        QueryType.DOCUMENT_LOOKUP: (
            r"\bdocument\b",
            r"\bcontract\b",
            r"\bagreement\b",
            r"\bnda\b",
            r"\bmsa\b",
            r"\bsow\b",
        ),

        QueryType.TABLE_LOOKUP: (
            r"\btable\b",
            r"\bschedule\b",
            r"\bmatrix\b",
            r"\bpricing\s+table\b",
            r"\bfee\s+schedule\b",
            r"\bpayment\s+schedule\b",
        ),

        QueryType.METADATA_LOOKUP: (
            r"\bauthor\b",
            r"\bcreated\s+(?:on|date)\b",
            r"\bmodified\s+(?:on|date)\b",
            r"\bfile\s+name\b",
            r"\bfilename\b",
            r"\bdocument\s+type\b",
            r"\bversion\b",
        ),

        QueryType.REVIEW: (
            r"\breview\b",
            r"\banaly[sz]e\s+this\s+contract\b",
            r"\bcheck\s+this\s+contract\b",
            r"\blegal\s+review\b",
        ),
    }

    TYPE_WEIGHTS: dict[
        QueryType,
        float,
    ] = {
        QueryType.DEFINITION: 1.20,
        QueryType.OBLIGATION: 1.25,
        QueryType.RISK: 1.35,
        QueryType.EXPIRY: 1.30,
        QueryType.RENEWAL: 1.30,
        QueryType.TERMINATION: 1.25,
        QueryType.PAYMENT: 1.15,
        QueryType.PRICING: 1.10,
        QueryType.LIABILITY: 1.35,
        QueryType.INDEMNIFICATION: 1.35,
        QueryType.CONFIDENTIALITY: 1.25,
        QueryType.INTELLECTUAL_PROPERTY: 1.30,
        QueryType.DATA_PROTECTION: 1.30,
        QueryType.PRIVACY: 1.20,
        QueryType.SECURITY: 1.25,
        QueryType.COMPLIANCE: 1.30,
        QueryType.AUDIT: 1.20,
        QueryType.INSURANCE: 1.15,
        QueryType.SERVICE_LEVEL: 1.25,
        QueryType.GOVERNING_LAW: 1.30,
        QueryType.DISPUTE_RESOLUTION: 1.30,
        QueryType.FORCE_MAJEURE: 1.25,
        QueryType.ASSIGNMENT: 1.20,
        QueryType.WARRANTY: 1.20,
        QueryType.NOTICE: 1.15,
        QueryType.CONTRACT_COMPARISON: 1.45,
        QueryType.DOCUMENT_SUMMARY: 1.35,
        QueryType.CLAUSE_LOOKUP: 0.90,
        QueryType.DOCUMENT_LOOKUP: 0.70,
        QueryType.TABLE_LOOKUP: 1.10,
        QueryType.METADATA_LOOKUP: 1.10,
        QueryType.REVIEW: 1.15,
    }

    def __init__(
        self,
        config: Optional[
            QueryClassifierConfig
        ] = None,
        *,
        model: Any = None,
    ) -> None:
        self.config = (
            config
            or QueryClassifierConfig()
        )

        self.model = model

        self._compiled_patterns = {
            query_type: [
                re.compile(
                    pattern,
                    re.IGNORECASE,
                )
                for pattern in patterns
            ]
            for query_type, patterns
            in self.INTENT_PATTERNS.items()
        }

        self._validate_config()

    # ==========================================================
    # Public API
    # ==========================================================

    def classify(
        self,
        query: str,
        *,
        document_ids: Optional[
            Sequence[str]
        ] = None,
        current_document_id: Optional[
            str
        ] = None,
        metadata: Optional[
            dict[str, Any]
        ] = None,
    ) -> QueryClassification:

        started = time.perf_counter()

        normalized = self._normalize_query(
            query
        )

        if not normalized:
            return QueryClassification(
                query_type=QueryType.UNKNOWN,
                confidence=0.0,
                scope=QueryScope.UNKNOWN,
                complexity=(
                    QueryComplexity.SIMPLE
                ),
                method=(
                    ClassificationMethod
                    .FALLBACK
                ),
                requires_retrieval=False,
                requires_reranking=False,
                requires_generation=False,
                requires_permission_filter=(
                    self.config
                    .enable_permission_filter
                ),
                metadata={
                    "reason": "empty_query",
                    "latency_ms":
                        self._elapsed(
                            started
                        ),
                },
            )

        supplied_document_ids = list(
            dict.fromkeys(
                str(document_id)
                for document_id
                in (document_ids or [])
                if document_id
            )
        )

        if (
            current_document_id
            and current_document_id
            not in supplied_document_ids
        ):
            supplied_document_ids.insert(
                0,
                current_document_id,
            )

        rule_signals = (
            self._classify_rules(
                normalized
            )
            if self.config
            .enable_rule_classifier
            else []
        )

        model_scores: dict[
            QueryType,
            float,
        ] = {}

        model_error: Optional[str] = None

        if (
            self.config
            .enable_model_classifier
            and self.model is not None
        ):
            try:
                model_scores = (
                    self._classify_model(
                        normalized
                    )
                )

            except Exception as exc:
                # Classification remains operational if the
                # optional ML classifier fails.
                model_error = str(exc)

        scores = self._merge_scores(
            rule_signals,
            model_scores,
        )

        if scores:
            ranked = sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            primary_type = ranked[
                0
            ][0]

            confidence = min(
                1.0,
                max(
                    0.0,
                    ranked[0][1],
                ),
            )

            if (
                confidence
                < self.config
                .minimum_confidence
            ):
                primary_type = (
                    self.config
                    .default_type
                )

        else:
            ranked = []

            primary_type = (
                self.config.default_type
            )

            confidence = 0.25

        secondary_types = (
            self._secondary_types(
                ranked,
                primary_type,
            )
        )

        scope = self._detect_scope(
            normalized,
            supplied_document_ids,
            current_document_id=(
                current_document_id
            ),
        )

        complexity = (
            self._detect_complexity(
                normalized,
                primary_type,
                secondary_types,
                supplied_document_ids,
            )
        )

        method = (
            self._classification_method(
                bool(rule_signals),
                bool(model_scores),
            )
        )

        flags = self._routing_flags(
            primary_type,
            secondary_types,
            complexity,
        )

        latency_ms = self._elapsed(
            started
        )

        return QueryClassification(
            query_type=primary_type,
            confidence=round(
                confidence,
                4,
            ),
            scope=scope,
            complexity=complexity,
            method=method,
            secondary_types=(
                secondary_types
            ),
            signals=sorted(
                rule_signals,
                key=lambda signal:
                    signal.score,
                reverse=True,
            ),
            requires_retrieval=(
                flags[
                    "requires_retrieval"
                ]
            ),
            requires_reranking=(
                flags[
                    "requires_reranking"
                ]
            ),
            requires_generation=(
                flags[
                    "requires_generation"
                ]
            ),
            requires_multi_query=(
                flags[
                    "requires_multi_query"
                ]
            ),
            requires_comparison=(
                flags[
                    "requires_comparison"
                ]
            ),
            requires_table_retrieval=(
                flags[
                    "requires_table_retrieval"
                ]
            ),
            requires_permission_filter=(
                self.config
                .enable_permission_filter
            ),
            document_ids=(
                supplied_document_ids
            ),
            metadata={
                **(metadata or {}),
                "normalized_query":
                    normalized,
                "model_error":
                    model_error,
                "score_map": {
                    key.value: round(
                        value,
                        4,
                    )
                    for key, value
                    in ranked
                },
                "latency_ms":
                    latency_ms,
            },
        )

    def classify_many(
        self,
        queries: Sequence[str],
        *,
        document_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> list[
        QueryClassification
    ]:

        return [
            self.classify(
                query,
                document_ids=document_ids,
            )
            for query in queries
        ]

    # ==========================================================
    # Deterministic classification
    # ==========================================================

    def _classify_rules(
        self,
        query: str,
    ) -> list[QuerySignal]:

        signals: list[
            QuerySignal
        ] = []

        for (
            query_type,
            patterns,
        ) in (
            self._compiled_patterns.items()
        ):
            matches: list[
                re.Match[str]
            ] = []

            for pattern in patterns:
                match = pattern.search(
                    query
                )

                if match:
                    matches.append(
                        match
                    )

            if not matches:
                continue

            weight = (
                self.TYPE_WEIGHTS.get(
                    query_type,
                    1.0,
                )
            )

            # Multiple independent matches strengthen intent,
            # but cap rule confidence.
            score = min(
                0.95,
                (
                    0.40
                    + 0.12
                    * min(
                        len(matches),
                        4,
                    )
                )
                * weight,
            )

            for match in matches[
                :3
            ]:
                signals.append(
                    QuerySignal(
                        query_type=(
                            query_type
                        ),
                        score=score,
                        source="rule",
                        matched_text=(
                            match.group(0)
                        ),
                    )
                )

        self._add_contextual_signals(
            query,
            signals,
        )

        return signals

    def _add_contextual_signals(
        self,
        query: str,
        signals: list[
            QuerySignal
        ],
    ) -> None:

        # Explicit "what are my obligations" style queries.
        if (
            "what are" in query
            and "obligation" in query
        ):
            signals.append(
                QuerySignal(
                    query_type=(
                        QueryType.OBLIGATION
                    ),
                    score=0.90,
                    source=(
                        "context_rule"
                    ),
                    matched_text=(
                        "what are ... "
                        "obligations"
                    ),
                )
            )

        # "When does this contract renew?"
        if (
            re.search(
                r"\bwhen\b",
                query,
            )
            and re.search(
                r"\brenew",
                query,
            )
        ):
            signals.append(
                QuerySignal(
                    query_type=(
                        QueryType.RENEWAL
                    ),
                    score=0.92,
                    source=(
                        "context_rule"
                    ),
                    matched_text=(
                        "when + renewal"
                    ),
                )
            )

        # Comparison queries should dominate generic document lookup.
        if (
            re.search(
                r"\bcompare\b|\bvs\.?\b|"
                r"\bversus\b|\bdifference",
                query,
            )
            and re.search(
                r"\bcontract\b|\bagreement\b|"
                r"\bdocument\b",
                query,
            )
        ):
            signals.append(
                QuerySignal(
                    query_type=(
                        QueryType
                        .CONTRACT_COMPARISON
                    ),
                    score=0.97,
                    source=(
                        "context_rule"
                    ),
                    matched_text=(
                        "document comparison"
                    ),
                )
            )

        # Summaries should dominate generic document lookup.
        if (
            re.search(
                r"\bsummar(?:ize|ise|y)\b|"
                r"\boverview\b",
                query,
            )
            and re.search(
                r"\bcontract\b|\bagreement\b|"
                r"\bdocument\b|\bthis\b",
                query,
            )
        ):
            signals.append(
                QuerySignal(
                    query_type=(
                        QueryType
                        .DOCUMENT_SUMMARY
                    ),
                    score=0.96,
                    source=(
                        "context_rule"
                    ),
                    matched_text=(
                        "document summary"
                    ),
                )
            )

    # ==========================================================
    # Optional ML classification
    # ==========================================================

    def _classify_model(
        self,
        query: str,
    ) -> dict[
        QueryType,
        float,
    ]:
        """
        Supports common classifier outputs:

        {"risk": 0.91, "payment": 0.05}

        [{"label": "risk", "score": 0.91}, ...]

        {"label": "risk", "score": 0.91}

        Hugging Face pipeline-style nested lists are also supported.
        """

        if hasattr(
            self.model,
            "classify",
        ):
            result = (
                self.model.classify(
                    query
                )
            )

        elif hasattr(
            self.model,
            "predict",
        ):
            result = (
                self.model.predict(
                    query
                )
            )

        elif callable(self.model):
            result = self.model(
                query
            )

        else:
            raise (
                QueryClassificationError(
                    "Unsupported query "
                    "classifier model.",
                    code=(
                        "UNSUPPORTED_MODEL"
                    ),
                )
            )

        return (
            self._parse_model_result(
                result
            )
        )

    def _parse_model_result(
        self,
        result: Any,
    ) -> dict[
        QueryType,
        float,
    ]:

        if result is None:
            return {}

        if (
            isinstance(result, list)
            and len(result) == 1
            and isinstance(
                result[0],
                list,
            )
        ):
            result = result[0]

        scores: dict[
            QueryType,
            float,
        ] = {}

        if isinstance(
            result,
            dict,
        ):
            if (
                "label" in result
                and "score" in result
            ):
                query_type = (
                    self._label_to_type(
                        result["label"]
                    )
                )

                if query_type:
                    scores[
                        query_type
                    ] = self._safe_score(
                        result["score"]
                    )

                return scores

            for label, score in (
                result.items()
            ):
                query_type = (
                    self._label_to_type(
                        label
                    )
                )

                if query_type:
                    scores[
                        query_type
                    ] = self._safe_score(
                        score
                    )

            return scores

        if isinstance(
            result,
            (list, tuple),
        ):
            for item in result:
                if not isinstance(
                    item,
                    dict,
                ):
                    continue

                label = item.get(
                    "label"
                )

                score = item.get(
                    "score"
                )

                query_type = (
                    self._label_to_type(
                        label
                    )
                )

                if (
                    query_type
                    and score is not None
                ):
                    scores[
                        query_type
                    ] = self._safe_score(
                        score
                    )

            return scores

        return {}

    @staticmethod
    def _safe_score(
        value: Any,
    ) -> float:

        try:
            score = float(value)

        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        if not math.isfinite(
            score
        ):
            return 0.0

        return min(
            1.0,
            max(
                0.0,
                score,
            ),
        )

    @staticmethod
    def _label_to_type(
        label: Any,
    ) -> Optional[QueryType]:

        if label is None:
            return None

        normalized = (
            str(label)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        aliases = {
            "qa": QueryType.LEGAL_QA,
            "legal": QueryType.LEGAL_QA,
            "legal_qa":
                QueryType.LEGAL_QA,

            "document":
                QueryType.DOCUMENT_LOOKUP,

            "clause":
                QueryType.CLAUSE_LOOKUP,

            "definition":
                QueryType.DEFINITION,

            "obligations":
                QueryType.OBLIGATION,

            "risk":
                QueryType.RISK,

            "expiry":
                QueryType.EXPIRY,

            "expiration":
                QueryType.EXPIRY,

            "renewal":
                QueryType.RENEWAL,

            "termination":
                QueryType.TERMINATION,

            "payment":
                QueryType.PAYMENT,

            "pricing":
                QueryType.PRICING,

            "liability":
                QueryType.LIABILITY,

            "indemnity":
                QueryType
                .INDEMNIFICATION,

            "indemnification":
                QueryType
                .INDEMNIFICATION,

            "confidentiality":
                QueryType
                .CONFIDENTIALITY,

            "ip":
                QueryType
                .INTELLECTUAL_PROPERTY,

            "intellectual_property":
                QueryType
                .INTELLECTUAL_PROPERTY,

            "data_protection":
                QueryType
                .DATA_PROTECTION,

            "privacy":
                QueryType.PRIVACY,

            "security":
                QueryType.SECURITY,

            "compliance":
                QueryType.COMPLIANCE,

            "audit":
                QueryType.AUDIT,

            "insurance":
                QueryType.INSURANCE,

            "sla":
                QueryType.SERVICE_LEVEL,

            "service_level":
                QueryType.SERVICE_LEVEL,

            "governing_law":
                QueryType.GOVERNING_LAW,

            "dispute":
                QueryType
                .DISPUTE_RESOLUTION,

            "dispute_resolution":
                QueryType
                .DISPUTE_RESOLUTION,

            "force_majeure":
                QueryType.FORCE_MAJEURE,

            "assignment":
                QueryType.ASSIGNMENT,

            "warranty":
                QueryType.WARRANTY,

            "notice":
                QueryType.NOTICE,

            "comparison":
                QueryType
                .CONTRACT_COMPARISON,

            "contract_comparison":
                QueryType
                .CONTRACT_COMPARISON,

            "summary":
                QueryType
                .DOCUMENT_SUMMARY,

            "table":
                QueryType.TABLE_LOOKUP,

            "metadata":
                QueryType.METADATA_LOOKUP,

            "review":
                QueryType.REVIEW,

            "search":
                QueryType.SEARCH,

            "general":
                QueryType.GENERAL,
        }

        if normalized in aliases:
            return aliases[
                normalized
            ]

        try:
            return QueryType(
                normalized
            )

        except ValueError:
            return None

    # ==========================================================
    # Score fusion
    # ==========================================================

    def _merge_scores(
        self,
        rule_signals: Sequence[
            QuerySignal
        ],
        model_scores: dict[
            QueryType,
            float,
        ],
    ) -> dict[
        QueryType,
        float,
    ]:

        rule_scores: dict[
            QueryType,
            float,
        ] = {}

        for signal in rule_signals:
            existing = (
                rule_scores.get(
                    signal.query_type,
                    0.0,
                )
            )

            rule_scores[
                signal.query_type
            ] = max(
                existing,
                signal.score,
            )

        types = (
            set(rule_scores)
            | set(model_scores)
        )

        combined: dict[
            QueryType,
            float,
        ] = {}

        for query_type in types:
            rule = rule_scores.get(
                query_type,
                0.0,
            )

            model = model_scores.get(
                query_type,
                0.0,
            )

            if rule and model:
                # Agreement between deterministic and model
                # classification gets a small confidence bonus.
                score = min(
                    1.0,
                    (
                        rule * 0.55
                        + model * 0.45
                        + 0.08
                    ),
                )

            elif rule:
                score = rule

            else:
                score = model

            combined[
                query_type
            ] = score

        return combined

    def _secondary_types(
        self,
        ranked: Sequence[
            tuple[
                QueryType,
                float,
            ]
        ],
        primary: QueryType,
    ) -> list[QueryType]:

        output: list[
            QueryType
        ] = []

        generic_types = {
            QueryType.DOCUMENT_LOOKUP,
            QueryType.CLAUSE_LOOKUP,
        }

        for query_type, score in ranked:
            if query_type == primary:
                continue

            if (
                score
                < self.config
                .secondary_threshold
            ):
                continue

            # Avoid adding generic "document" merely because
            # every legal question mentions a contract.
            if (
                query_type
                in generic_types
                and primary
                not in generic_types
            ):
                continue

            output.append(
                query_type
            )

            if (
                len(output)
                >= self.config
                .maximum_secondary_types
            ):
                break

        return output

    # ==========================================================
    # Scope
    # ==========================================================

    def _detect_scope(
        self,
        query: str,
        document_ids: Sequence[str],
        *,
        current_document_id: Optional[
            str
        ],
    ) -> QueryScope:

        if (
            len(document_ids)
            >= self.config
            .comparison_min_documents
        ):
            return (
                QueryScope
                .MULTIPLE_DOCUMENTS
            )

        if re.search(
            r"\ball\s+(?:documents|contracts|agreements)\b",
            query,
        ):
            return QueryScope.ORGANIZATION

        if re.search(
            r"\bacross\s+(?:the\s+)?"
            r"(?:organization|company|workspace)\b",
            query,
        ):
            return QueryScope.ORGANIZATION

        if re.search(
            r"\bglobal(?:ly)?\b|"
            r"\bpublic\s+law\b",
            query,
        ):
            return QueryScope.GLOBAL

        if re.search(
            r"\bthese\s+(?:documents|contracts|agreements)\b",
            query,
        ):
            return (
                QueryScope
                .MULTIPLE_DOCUMENTS
            )

        if (
            current_document_id
            or len(document_ids) == 1
        ):
            return (
                QueryScope
                .CURRENT_DOCUMENT
            )

        if re.search(
            r"\bthis\s+(?:document|contract|agreement|nda|msa|sow)\b",
            query,
        ):
            return (
                QueryScope
                .CURRENT_DOCUMENT
            )

        return self.config.default_scope

    # ==========================================================
    # Complexity
    # ==========================================================

    def _detect_complexity(
        self,
        query: str,
        primary_type: QueryType,
        secondary_types: Sequence[
            QueryType
        ],
        document_ids: Sequence[str],
    ) -> QueryComplexity:

        word_count = len(
            query.split()
        )

        complex_types = {
            QueryType.CONTRACT_COMPARISON,
            QueryType.REVIEW,
            QueryType.DOCUMENT_SUMMARY,
            QueryType.RISK,
        }

        if (
            word_count
            >= self.config
            .complex_query_word_threshold
        ):
            return QueryComplexity.COMPLEX

        if (
            primary_type
            == QueryType
            .CONTRACT_COMPARISON
            or len(document_ids) > 1
        ):
            return QueryComplexity.COMPLEX

        if (
            len(secondary_types) >= 2
            and primary_type
            in complex_types
        ):
            return QueryComplexity.COMPLEX

        if (
            word_count
            >= self.config
            .long_query_word_threshold
        ):
            return QueryComplexity.MODERATE

        if secondary_types:
            return QueryComplexity.MODERATE

        if primary_type in {
            QueryType.RISK,
            QueryType.REVIEW,
            QueryType.DOCUMENT_SUMMARY,
        }:
            return QueryComplexity.MODERATE

        return QueryComplexity.SIMPLE

    # ==========================================================
    # Retrieval routing flags
    # ==========================================================

    @staticmethod
    def _routing_flags(
        primary: QueryType,
        secondary: Sequence[
            QueryType
        ],
        complexity: QueryComplexity,
    ) -> dict[str, bool]:

        all_types = {
            primary,
            *secondary,
        }

        comparison = (
            QueryType
            .CONTRACT_COMPARISON
            in all_types
        )

        table = bool(
            all_types
            & {
                QueryType.TABLE_LOOKUP,
                QueryType.PRICING,
                QueryType.PAYMENT,
                QueryType.SERVICE_LEVEL,
            }
        )

        multi_query = (
            complexity
            == QueryComplexity.COMPLEX
            or comparison
            or QueryType.REVIEW
            in all_types
            or QueryType.DOCUMENT_SUMMARY
            in all_types
        )

        # Metadata can usually be answered from structured DB
        # metadata without vector retrieval.
        if (
            primary
            == QueryType.METADATA_LOOKUP
        ):
            return {
                "requires_retrieval": False,
                "requires_reranking": False,
                "requires_generation": True,
                "requires_multi_query": False,
                "requires_comparison": False,
                "requires_table_retrieval": False,
            }

        return {
            "requires_retrieval": True,
            "requires_reranking": True,
            "requires_generation": True,
            "requires_multi_query":
                multi_query,
            "requires_comparison":
                comparison,
            "requires_table_retrieval":
                table,
        }

    # ==========================================================
    # Method
    # ==========================================================

    @staticmethod
    def _classification_method(
        has_rules: bool,
        has_model: bool,
    ) -> ClassificationMethod:

        if has_rules and has_model:
            return (
                ClassificationMethod.HYBRID
            )

        if has_rules:
            return (
                ClassificationMethod.RULE
            )

        if has_model:
            return (
                ClassificationMethod.MODEL
            )

        return (
            ClassificationMethod.FALLBACK
        )

    # ==========================================================
    # Query normalization
    # ==========================================================

    def _normalize_query(
        self,
        query: Any,
    ) -> str:

        if query is None:
            return ""

        value = str(query).strip()

        if not self.config.normalize_query:
            return value

        value = value.replace(
            "\r\n",
            "\n",
        )

        value = value.replace(
            "\r",
            "\n",
        )

        value = re.sub(
            r"[ \t]+",
            " ",
            value,
        )

        value = re.sub(
            r"\n{3,}",
            "\n\n",
            value,
        )

        return value.strip().lower()

    # ==========================================================
    # Config validation
    # ==========================================================

    def _validate_config(
        self,
    ) -> None:

        if not (
            0.0
            <= self.config
            .minimum_confidence
            <= 1.0
        ):
            raise ValueError(
                "minimum_confidence must "
                "be between 0 and 1."
            )

        if not (
            0.0
            <= self.config
            .secondary_threshold
            <= 1.0
        ):
            raise ValueError(
                "secondary_threshold must "
                "be between 0 and 1."
            )

        if (
            self.config
            .maximum_secondary_types
            < 0
        ):
            raise ValueError(
                "maximum_secondary_types "
                "cannot be negative."
            )

    @staticmethod
    def _elapsed(
        started: float,
    ) -> float:

        return round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            3,
        )


def classify_query(
    query: str,
    *,
    document_ids: Optional[
        Sequence[str]
    ] = None,
    current_document_id: Optional[
        str
    ] = None,
    model: Any = None,
) -> QueryClassification:
    """Convenience entry point."""

    classifier = QueryClassifier(
        model=model
    )

    return classifier.classify(
        query,
        document_ids=document_ids,
        current_document_id=(
            current_document_id
        ),
    )