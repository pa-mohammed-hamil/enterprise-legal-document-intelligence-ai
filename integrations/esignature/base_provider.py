"""
Base E-Signature Provider

Abstract base class for e-signature providers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SignatureStatus(str, Enum):
    """Signature request status."""
    DRAFT = "draft"
    SENT = "sent"
    DELIVERED = "delivered"
    SIGNED = "signed"
    COMPLETED = "completed"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    VOIDED = "voided"


class SignatureType(str, Enum):
    """Type of signature required."""
    ELECTRONIC = "electronic"
    DIGITAL = "digital"
    WET_INK = "wet_ink"


class Signer(BaseModel):
    """Signer information."""
    name: str
    email: str
    role: str = "signer"
    order: int = 1
    authentication_method: Optional[str] = None
    phone_number: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None


class SignatureField(BaseModel):
    """Signature field placement."""
    page_number: int
    x_position: float
    y_position: float
    width: float = 150
    height: float = 50
    signer_email: str
    field_type: str = "signature"  # signature, initial, date, text
    required: bool = True
    tab_label: Optional[str] = None


class SignatureRequest(BaseModel):
    """E-signature request."""
    document_id: str
    document_name: str
    subject: str
    message: str
    signers: list[Signer]
    cc_recipients: list[str] = Field(default_factory=list)
    signature_fields: list[SignatureField] = Field(default_factory=list)
    expiration_days: int = 30
    reminder_frequency: Optional[int] = 7  # days
    allow_decline: bool = True
    require_all_signers: bool = True
    signing_order: bool = False


class SignatureResponse(BaseModel):
    """E-signature response."""
    request_id: str
    provider: str
    status: SignatureStatus
    document_id: str
    envelope_id: Optional[str] = None
    signing_url: Optional[str] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    signers_status: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SignatureWebhookEvent(BaseModel):
    """Webhook event from e-signature provider."""
    event_type: str
    envelope_id: str
    status: SignatureStatus
    timestamp: datetime
    signer_email: Optional[str] = None
    ip_address: Optional[str] = None
    reason: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseESignatureProvider(ABC):
    """
    Abstract base class for e-signature providers.
    
    All e-signature providers must implement these methods.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize provider with configuration.
        
        Args:
            config: Provider configuration including API credentials
        """
        self.config = config
        self.provider_name = "base"

    @abstractmethod
    async def create_signature_request(
        self, request: SignatureRequest, document_path: str
    ) -> SignatureResponse:
        """
        Create a new signature request.
        
        Args:
            request: Signature request details
            document_path: Path to document file
            
        Returns:
            Signature response with request details
        """
        pass

    @abstractmethod
    async def get_signature_status(self, request_id: str) -> SignatureResponse:
        """
        Get status of a signature request.
        
        Args:
            request_id: Provider-specific request ID
            
        Returns:
            Current signature status
        """
        pass

    @abstractmethod
    async def download_signed_document(
        self, request_id: str, output_path: str
    ) -> str:
        """
        Download completed signed document.
        
        Args:
            request_id: Provider-specific request ID
            output_path: Path to save signed document
            
        Returns:
            Path to downloaded document
        """
        pass

    @abstractmethod
    async def cancel_signature_request(self, request_id: str, reason: str) -> bool:
        """
        Cancel a pending signature request.
        
        Args:
            request_id: Provider-specific request ID
            reason: Reason for cancellation
            
        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    async def send_reminder(self, request_id: str, signer_email: str) -> bool:
        """
        Send reminder to a signer.
        
        Args:
            request_id: Provider-specific request ID
            signer_email: Email of signer to remind
            
        Returns:
            True if reminder sent successfully
        """
        pass

    @abstractmethod
    async def get_audit_trail(self, request_id: str) -> dict[str, Any]:
        """
        Get audit trail for a signature request.
        
        Args:
            request_id: Provider-specific request ID
            
        Returns:
            Audit trail information
        """
        pass

    @abstractmethod
    async def verify_signature(self, request_id: str) -> dict[str, Any]:
        """
        Verify authenticity of signatures.
        
        Args:
            request_id: Provider-specific request ID
            
        Returns:
            Verification results
        """
        pass

    @abstractmethod
    def parse_webhook(self, payload: dict[str, Any]) -> SignatureWebhookEvent:
        """
        Parse webhook payload from provider.
        
        Args:
            payload: Raw webhook payload
            
        Returns:
            Parsed webhook event
        """
        pass

    @abstractmethod
    async def get_embedded_signing_url(
        self, request_id: str, signer_email: str, return_url: str
    ) -> str:
        """
        Get embedded signing URL for in-app signing.
        
        Args:
            request_id: Provider-specific request ID
            signer_email: Email of signer
            return_url: URL to redirect after signing
            
        Returns:
            Embedded signing URL
        """
        pass

    async def validate_configuration(self) -> bool:
        """
        Validate provider configuration and credentials.
        
        Returns:
            True if configuration is valid
        """
        return True

    def supports_feature(self, feature: str) -> bool:
        """
        Check if provider supports a specific feature.
        
        Args:
            feature: Feature name
            
        Returns:
            True if feature is supported
        """
        features = {
            "embedded_signing": False,
            "bulk_send": False,
            "templates": False,
            "advanced_fields": False,
            "webhooks": False,
            "sms_authentication": False,
            "id_verification": False,
            "payment_collection": False,
        }
        return features.get(feature, False)
