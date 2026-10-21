# backend/models/huggingface/document_classifier.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
import re


class DocumentType(str, Enum):
    NDA = "nda"
    MASTER_SERVICE_AGREEMENT = "master_service_agreement"
    SERVICE_AGREEMENT = "service_agreement"
    VENDOR_AGREEMENT = "vendor_agreement"
    SUPPLIER_AGREEMENT = "supplier_agreement"
    SOFTWARE_LICENSE = "software_license"
    SOFTWARE_AS_A_SERVICE = "software_as_a_service"
    DATA_PROCESSING_AGREEMENT = "data_processing_agreement"
    PRIVACY_AGREEMENT = "privacy_agreement"
    EMPLOYMENT_AGREEMENT = "employment_agreement"
    CONSULTING_AGREEMENT = "consulting_agreement"
    CONTRACTOR_AGREEMENT = "contractor_agreement"
    LEASE_AGREEMENT = "lease_agreement"
    PURCHASE_AGREEMENT = "purchase_agreement"
    SALES_AGREEMENT = "sales_agreement"
    PARTNERSHIP_AGREEMENT = "partnership_agreement"
    JOINT_VENTURE_AGREEMENT = "joint_venture_agreement"
    LICENSE_AGREEMENT = "license_agreement"
    DISTRIBUTION_AGREEMENT = "distribution_agreement"
    FRANCHISE_AGREEMENT = "franchise_agreement"
    LOAN_AGREEMENT = "loan_agreement"
    FINANCING_AGREEMENT = "financing_agreement"
    INSURANCE_AGREEMENT = "insurance_agreement"
    AMENDMENT = "amendment"
    ADDENDUM = "addendum"
    STATEMENT_OF_WORK = "statement_of_work"
    TERMS_AND_CONDITIONS = "terms_and_conditions"
    PRIVACY_POLICY = "privacy_policy"
    CORPORATE_POLICY = "corporate_policy"
    COMPLIANCE_DOCUMENT = "compliance_document"
    REGULATORY_DOCUMENT = "regulatory_document"
    COURT_DOCUMENT = "court_document"
    LEGAL_NOTICE = "legal_notice"
    POWER_OF_ATTORNEY = "power_of_attorney"
    MEMORANDUM = "memorandum"
    OTHER = "other"


@dataclass(frozen=True)
class DocumentClassification:
    document_type: DocumentType
    confidence: float
    model_name: str
    model_version: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentClassificationResult:
    document_id: Optional[str]
    primary: DocumentClassification
    alternatives: List[DocumentClassification] = field(
        default_factory=list
    )


class DocumentClassifier:
    """
    Document-level legal document classifier.

    Responsibilities:
        - identify the document category
        - normalize model labels
        - provide confidence and evidence
        - provide deterministic fallback classification
        - support batch classification

    This classifier does NOT:
        - determine legal validity
        - determine enforceability
        - assess legal risk
        - provide legal advice
        - determine contractual obligations
    """

    LABEL_ALIASES: Dict[str, DocumentType] = {
        "nda": DocumentType.NDA,
        "non_disclosure_agreement": DocumentType.NDA,
        "confidentiality_agreement": DocumentType.NDA,

        "master_service_agreement": DocumentType.MASTER_SERVICE_AGREEMENT,
        "msa": DocumentType.MASTER_SERVICE_AGREEMENT,

        "service_agreement": DocumentType.SERVICE_AGREEMENT,
        "services_agreement": DocumentType.SERVICE_AGREEMENT,

        "vendor_agreement": DocumentType.VENDOR_AGREEMENT,
        "supplier_agreement": DocumentType.SUPPLIER_AGREEMENT,

        "software_license": DocumentType.SOFTWARE_LICENSE,
        "software_license_agreement": DocumentType.SOFTWARE_LICENSE,

        "saas": DocumentType.SOFTWARE_AS_A_SERVICE,
        "saas_agreement": DocumentType.SOFTWARE_AS_A_SERVICE,
        "software_as_a_service": DocumentType.SOFTWARE_AS_A_SERVICE,

        "data_processing_agreement": DocumentType.DATA_PROCESSING_AGREEMENT,
        "dpa": DocumentType.DATA_PROCESSING_AGREEMENT,

        "privacy_agreement": DocumentType.PRIVACY_AGREEMENT,
        "privacy_policy": DocumentType.PRIVACY_POLICY,

        "employment_agreement": DocumentType.EMPLOYMENT_AGREEMENT,
        "employment_contract": DocumentType.EMPLOYMENT_AGREEMENT,

        "consulting_agreement": DocumentType.CONSULTING_AGREEMENT,
        "consultancy_agreement": DocumentType.CONSULTING_AGREEMENT,

        "contractor_agreement": DocumentType.CONTRACTOR_AGREEMENT,

        "lease": DocumentType.LEASE_AGREEMENT,
        "lease_agreement": DocumentType.LEASE_AGREEMENT,

        "purchase_agreement": DocumentType.PURCHASE_AGREEMENT,
        "sales_agreement": DocumentType.SALES_AGREEMENT,

        "partnership_agreement": DocumentType.PARTNERSHIP_AGREEMENT,
        "joint_venture_agreement": DocumentType.JOINT_VENTURE_AGREEMENT,

        "license_agreement": DocumentType.LICENSE_AGREEMENT,
        "licensing_agreement": DocumentType.LICENSE_AGREEMENT,

        "distribution_agreement": DocumentType.DISTRIBUTION_AGREEMENT,
        "franchise_agreement": DocumentType.FRANCHISE_AGREEMENT,

        "loan_agreement": DocumentType.LOAN_AGREEMENT,
        "financing_agreement": DocumentType.FINANCING_AGREEMENT,

        "insurance_agreement": DocumentType.INSURANCE_AGREEMENT,

        "amendment": DocumentType.AMENDMENT,
        "contract_amendment": DocumentType.AMENDMENT,

        "addendum": DocumentType.ADDENDUM,

        "statement_of_work": DocumentType.STATEMENT_OF_WORK,
        "sow": DocumentType.STATEMENT_OF_WORK,

        "terms_and_conditions": DocumentType.TERMS_AND_CONDITIONS,
        "terms_of_service": DocumentType.TERMS_AND_CONDITIONS,

        "corporate_policy": DocumentType.CORPORATE_POLICY,
        "compliance_document": DocumentType.COMPLIANCE_DOCUMENT,
        "regulatory_document": DocumentType.REGULATORY_DOCUMENT,

        "court_document": DocumentType.COURT_DOCUMENT,
        "legal_notice": DocumentType.LEGAL_NOTICE,
        "notice": DocumentType.LEGAL_NOTICE,

        "power_of_attorney": DocumentType.POWER_OF_ATTORNEY,
        "memorandum": DocumentType.MEMORANDUM,
    }

    KEYWORDS: Dict[DocumentType, Sequence[str]] = {
        DocumentType.NDA: (
            "non-disclosure agreement",
            "nondisclosure agreement",
            "confidentiality agreement",
            "confidential information",
            "receiving party",
            "disclosing party",
        ),
        DocumentType.MASTER_SERVICE_AGREEMENT: (
            "master services agreement",
            "master service agreement",
            "msa",
            "statement of work",
        ),
        DocumentType.SERVICE_AGREEMENT: (
            "service agreement",
            "services agreement",
            "services provided",
            "service provider",
        ),
        DocumentType.VENDOR_AGREEMENT: (
            "vendor agreement",
            "vendor",
            "supplier",
            "purchase order",
        ),
        DocumentType.SUPPLIER_AGREEMENT: (
            "supplier agreement",
            "supplier",
            "supply of goods",
            "supply agreement",
        ),
        DocumentType.SOFTWARE_LICENSE: (
            "software license agreement",
            "software licence agreement",
            "license grant",
            "licensed software",
        ),
        DocumentType.SOFTWARE_AS_A_SERVICE: (
            "software as a service",
            "saas",
            "subscription service",
            "cloud service",
        ),
        DocumentType.DATA_PROCESSING_AGREEMENT: (
            "data processing agreement",
            "data processor",
            "data controller",
            "processing of personal data",
        ),
        DocumentType.PRIVACY_AGREEMENT: (
            "privacy agreement",
            "personal data",
            "privacy obligations",
        ),
        DocumentType.EMPLOYMENT_AGREEMENT: (
            "employment agreement",
            "employment contract",
            "employee",
            "employer",
            "salary",
            "compensation",
        ),
        DocumentType.CONSULTING_AGREEMENT: (
            "consulting agreement",
            "consultant",
            "consulting services",
        ),
        DocumentType.CONTRACTOR_AGREEMENT: (
            "contractor agreement",
            "independent contractor",
            "contractor services",
        ),
        DocumentType.LEASE_AGREEMENT: (
            "lease agreement",
            "lease term",
            "landlord",
            "tenant",
            "premises",
            "rent",
        ),
        DocumentType.PURCHASE_AGREEMENT: (
            "purchase agreement",
            "purchase price",
            "buyer",
            "seller",
            "purchase of goods",
        ),
        DocumentType.SALES_AGREEMENT: (
            "sales agreement",
            "sale of goods",
            "sales price",
            "seller",
            "buyer",
        ),
        DocumentType.PARTNERSHIP_AGREEMENT: (
            "partnership agreement",
            "partners",
            "partnership interest",
            "profit sharing",
        ),
        DocumentType.JOINT_VENTURE_AGREEMENT: (
            "joint venture agreement",
            "joint venture",
            "venture parties",
        ),
        DocumentType.LICENSE_AGREEMENT: (
            "license agreement",
            "licensor",
            "licensee",
            "license rights",
        ),
        DocumentType.DISTRIBUTION_AGREEMENT: (
            "distribution agreement",
            "distributor",
            "distribution rights",
            "territory",
        ),
        DocumentType.FRANCHISE_AGREEMENT: (
            "franchise agreement",
            "franchisor",
            "franchisee",
            "franchise",
        ),
        DocumentType.LOAN_AGREEMENT: (
            "loan agreement",
            "borrower",
            "lender",
            "principal amount",
            "interest rate",
        ),
        DocumentType.FINANCING_AGREEMENT: (
            "financing agreement",
            "financing",
            "financed amount",
            "credit facility",
        ),
        DocumentType.INSURANCE_AGREEMENT: (
            "insurance agreement",
            "insured",
            "insurer",
            "policy coverage",
            "premium",
        ),
        DocumentType.AMENDMENT: (
            "amendment",
            "amendment agreement",
            "amends the agreement",
            "modification to agreement",
        ),
        DocumentType.ADDENDUM: (
            "addendum",
            "supplement to agreement",
            "additional terms",
        ),
        DocumentType.STATEMENT_OF_WORK: (
            "statement of work",
            "scope of work",
            "deliverables",
            "milestones",
            "sow",
        ),
        DocumentType.TERMS_AND_CONDITIONS: (
            "terms and conditions",
            "terms of service",
            "terms of use",
            "user terms",
        ),
        DocumentType.PRIVACY_POLICY: (
            "privacy policy",
            "privacy notice",
            "information we collect",
            "personal information",
        ),
        DocumentType.CORPORATE_POLICY: (
            "corporate policy",
            "company policy",
            "policy statement",
            "employees must",
        ),
        DocumentType.COMPLIANCE_DOCUMENT: (
            "compliance",
            "compliance requirements",
            "regulatory compliance",
            "compliance framework",
        ),
        DocumentType.REGULATORY_DOCUMENT: (
            "regulatory",
            "regulator",
            "regulatory requirements",
            "regulatory authority",
        ),
        DocumentType.COURT_DOCUMENT: (
            "court",
            "plaintiff",
            "defendant",
            "case number",
            "motion",
            "complaint",
        ),
        DocumentType.LEGAL_NOTICE: (
            "legal notice",
            "notice to",
            "formal notice",
            "notice of breach",
        ),
        DocumentType.POWER_OF_ATTORNEY: (
            "power of attorney",
            "attorney-in-fact",
            "appoints an agent",
        ),
        DocumentType.MEMORANDUM: (
            "memorandum",
            "memorandum of understanding",
            "mou",
            "understanding between",
        ),
    }

    def __init__(
        self,
        model: Optional[Any] = None,
        *,
        model_name: str = "legal-document-classifier",
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
        document_id: Optional[str] = None,
        top_k: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentClassificationResult:
        """
        Classify a document using the configured model or fallback.
        """

        normalized = self._normalize(text)

        if not normalized:
            raise ValueError("Document text cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        metadata = metadata or {}

        if self.model is not None:
            classifications = self._predict_model(
                normalized,
                top_k=top_k,
                metadata=metadata,
            )
        elif self.use_keyword_fallback:
            classifications = self._predict_keywords(
                normalized,
                top_k=top_k,
                metadata=metadata,
            )
        else:
            classifications = [
                self._create_classification(
                    DocumentType.OTHER,
                    0.0,
                    [],
                    metadata,
                )
            ]

        return DocumentClassificationResult(
            document_id=document_id,
            primary=classifications[0],
            alternatives=classifications[1:],
        )

    def classify_batch(
        self,
        texts: Sequence[str],
        *,
        top_k: int = 3,
    ) -> List[DocumentClassificationResult]:
        return [
            self.classify(
                text,
                top_k=top_k,
            )
            for text in texts
        ]

    def classify_with_model_output(
        self,
        text: str,
        model_outputs: Sequence[Dict[str, Any]],
        *,
        document_id: Optional[str] = None,
        top_k: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DocumentClassificationResult:
        """
        Convert Hugging Face-style outputs into the application's
        controlled DocumentType taxonomy.

        Example:

            [
                {"label": "NDA", "score": 0.96},
                {"label": "SERVICE_AGREEMENT", "score": 0.02},
            ]
        """

        normalized = self._normalize(text)
        metadata = metadata or {}

        classifications: List[DocumentClassification] = []

        for output in model_outputs:
            label = str(output.get("label", ""))
            score = self._clamp(
                float(output.get("score", 0.0))
            )

            document_type = self._map_label(label)

            if document_type is None:
                continue

            classifications.append(
                self._create_classification(
                    document_type=document_type,
                    confidence=score,
                    evidence=self._extract_evidence(
                        normalized,
                        document_type,
                    ),
                    metadata=metadata,
                )
            )

        classifications.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        if not classifications:
            classifications = [
                self._create_classification(
                    DocumentType.OTHER,
                    0.0,
                    [],
                    metadata,
                )
            ]

        return DocumentClassificationResult(
            document_id=document_id,
            primary=classifications[0],
            alternatives=classifications[1:top_k],
        )

    def _predict_model(
        self,
        text: str,
        *,
        top_k: int,
        metadata: Dict[str, Any],
    ) -> List[DocumentClassification]:
        """
        Supports a callable model or Hugging Face pipeline-like
        classifier.
        """

        output = self.model(
            text,
            top_k=top_k,
        )

        if isinstance(output, dict):
            output = [output]

        return self._convert_model_output(
            text=text,
            outputs=output,
            top_k=top_k,
            metadata=metadata,
        )

    def _convert_model_output(
        self,
        *,
        text: str,
        outputs: Sequence[Dict[str, Any]],
        top_k: int,
        metadata: Dict[str, Any],
    ) -> List[DocumentClassification]:
        classifications: List[DocumentClassification] = []

        for output in outputs:
            label = str(output.get("label", ""))
            document_type = self._map_label(label)

            if document_type is None:
                continue

            score = self._clamp(
                float(output.get("score", 0.0))
            )

            classifications.append(
                self._create_classification(
                    document_type=document_type,
                    confidence=score,
                    evidence=self._extract_evidence(
                        text,
                        document_type,
                    ),
                    metadata=metadata,
                )
            )

        classifications.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        if not classifications:
            classifications.append(
                self._create_classification(
                    DocumentType.OTHER,
                    0.0,
                    [],
                    metadata,
                )
            )

        return classifications[:top_k]

    def _predict_keywords(
        self,
        text: str,
        *,
        top_k: int,
        metadata: Dict[str, Any],
    ) -> List[DocumentClassification]:
        """
        Deterministic baseline used when no trained classifier is
        available.
        """

        results: List[DocumentClassification] = []

        for document_type, keywords in self.KEYWORDS.items():
            matched = [
                keyword
                for keyword in keywords
                if self._contains_term(text, keyword)
            ]

            if not matched:
                continue

            # Evidence count contributes to the baseline score,
            # but confidence is deliberately capped.
            score = min(
                0.45 + (0.08 * len(matched)),
                0.90,
            )

            results.append(
                self._create_classification(
                    document_type=document_type,
                    confidence=score,
                    evidence=matched,
                    metadata={
                        **metadata,
                        "classification_method": "keyword_fallback",
                    },
                )
            )

        if not results:
            return [
                self._create_classification(
                    DocumentType.OTHER,
                    0.20,
                    [],
                    {
                        **metadata,
                        "classification_method": "fallback",
                    },
                )
            ]

        results.sort(
            key=lambda item: item.confidence,
            reverse=True,
        )

        # Do not overstate classification confidence.
        if results[0].confidence < self.confidence_threshold:
            results.insert(
                0,
                self._create_classification(
                    DocumentType.OTHER,
                    1.0 - results[0].confidence,
                    [],
                    {
                        **metadata,
                        "classification_method":
                            "low_confidence_fallback",
                    },
                ),
            )

        return results[:top_k]

    def _extract_evidence(
        self,
        text: str,
        document_type: DocumentType,
    ) -> List[str]:
        keywords = self.KEYWORDS.get(
            document_type,
            (),
        )

        return [
            keyword
            for keyword in keywords
            if self._contains_term(text, keyword)
        ][:10]

    def _map_label(
        self,
        label: str,
    ) -> Optional[DocumentType]:
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
        document_type: DocumentType,
        confidence: float,
        evidence: List[str],
        metadata: Dict[str, Any],
    ) -> DocumentClassification:
        return DocumentClassification(
            document_type=document_type,
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
        return bool(
            re.search(
                rf"(?<!\w){re.escape(term.lower())}(?!\w)",
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