"""
E-Signature Service

Unified service for managing e-signatures across multiple providers.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.integrations.esignature.adobe_sign_provider import AdobeSignProvider
from backend.integrations.esignature.base_provider import (
    BaseESignatureProvider,
    SignatureRequest,
    SignatureResponse,
    SignatureStatus,
    SignatureWebhookEvent,
)
from backend.integrations.esignature.docusign_provider import DocuSignProvider
from backend.integrations.esignature.hellosign_provider import HelloSignProvider

logger = logging.getLogger(__name__)


class ESignatureService:
    """
    Unified e-signature service supporting multiple providers.
    
    Providers:
    - DocuSign
    - Adobe Sign
    - HelloSign (Dropbox Sign)
    
    Features:
    - Provider abstraction
    - Automatic failover
    - Unified audit trail
    - Compliance tracking
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize e-signature service.
        
        Args:
            config: Service configuration including provider settings
        """
        self.config = config
        self.default_provider = config.get("default_provider", "docusign")
        self.providers: dict[str, BaseESignatureProvider] = {}
        
        # Initialize configured providers
        self._init_providers()

    def _init_providers(self):
        """Initialize all configured providers."""
        provider_configs = self.config.get("providers", {})
        
        for provider_name, provider_config in provider_configs.items():
            if not provider_config.get("enabled", False):
                continue
            
            try:
                if provider_name == "docusign":
                    self.providers["docusign"] = DocuSignProvider(provider_config)
                elif provider_name == "adobe_sign":
                    self.providers["adobe_sign"] = AdobeSignProvider(provider_config)
                elif provider_name == "hellosign":
                    self.providers["hellosign"] = HelloSignProvider(provider_config)
                else:
                    logger.warning(f"Unknown provider: {provider_name}")
                
                logger.info(f"Initialized e-signature provider: {provider_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {provider_name}: {e}")

    def get_provider(self, provider_name: Optional[str] = None) -> BaseESignatureProvider:
        """
        Get e-signature provider.
        
        Args:
            provider_name: Provider name (defaults to default_provider)
            
        Returns:
            E-signature provider instance
        """
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            raise ValueError(f"Provider not configured: {provider_name}")
        
        return self.providers[provider_name]

    async def create_signature_request(
        self,
        request: SignatureRequest,
        document_path: str,
        provider: Optional[str] = None,
    ) -> SignatureResponse:
        """
        Create signature request.
        
        Args:
            request: Signature request details
            document_path: Path to document
            provider: Provider name (optional)
            
        Returns:
            Signature response
        """
        provider_instance = self.get_provider(provider)
        
        try:
            response = await provider_instance.create_signature_request(
                request, document_path
            )
            
            # Store request in database
            await self._store_signature_request(response, request)
            
            logger.info(
                f"Created signature request {response.request_id} "
                f"via {provider_instance.provider_name}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to create signature request: {e}")
            raise

    async def get_signature_status(
        self, request_id: str, provider: Optional[str] = None
    ) -> SignatureResponse:
        """
        Get signature request status.
        
        Args:
            request_id: Request ID
            provider: Provider name (optional)
            
        Returns:
            Current status
        """
        provider_instance = self.get_provider(provider)
        
        try:
            status = await provider_instance.get_signature_status(request_id)
            
            # Update status in database
            await self._update_signature_status(request_id, status)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get signature status: {e}")
            raise

    async def download_signed_document(
        self,
        request_id: str,
        output_path: str,
        provider: Optional[str] = None,
    ) -> str:
        """
        Download signed document.
        
        Args:
            request_id: Request ID
            output_path: Output file path
            provider: Provider name (optional)
            
        Returns:
            Path to downloaded document
        """
        provider_instance = self.get_provider(provider)
        
        try:
            path = await provider_instance.download_signed_document(
                request_id, output_path
            )
            
            logger.info(f"Downloaded signed document: {path}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to download document: {e}")
            raise

    async def cancel_signature_request(
        self,
        request_id: str,
        reason: str,
        provider: Optional[str] = None,
    ) -> bool:
        """
        Cancel signature request.
        
        Args:
            request_id: Request ID
            reason: Cancellation reason
            provider: Provider name (optional)
            
        Returns:
            True if cancelled successfully
        """
        provider_instance = self.get_provider(provider)
        
        try:
            success = await provider_instance.cancel_signature_request(
                request_id, reason
            )
            
            if success:
                await self._update_signature_status(
                    request_id,
                    SignatureResponse(
                        request_id=request_id,
                        provider=provider_instance.provider_name,
                        status=SignatureStatus.CANCELLED,
                        document_id="",
                        created_at=datetime.utcnow(),
                    ),
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cancel signature request: {e}")
            return False

    async def send_reminder(
        self,
        request_id: str,
        signer_email: str,
        provider: Optional[str] = None,
    ) -> bool:
        """
        Send reminder to signer.
        
        Args:
            request_id: Request ID
            signer_email: Signer email
            provider: Provider name (optional)
            
        Returns:
            True if sent successfully
        """
        provider_instance = self.get_provider(provider)
        
        try:
            return await provider_instance.send_reminder(request_id, signer_email)
            
        except Exception as e:
            logger.error(f"Failed to send reminder: {e}")
            return False

    async def get_audit_trail(
        self, request_id: str, provider: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Get audit trail.
        
        Args:
            request_id: Request ID
            provider: Provider name (optional)
            
        Returns:
            Audit trail data
        """
        provider_instance = self.get_provider(provider)
        
        try:
            return await provider_instance.get_audit_trail(request_id)
            
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            return {}

    async def verify_signature(
        self, request_id: str, provider: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Verify signatures.
        
        Args:
            request_id: Request ID
            provider: Provider name (optional)
            
        Returns:
            Verification results
        """
        provider_instance = self.get_provider(provider)
        
        try:
            return await provider_instance.verify_signature(request_id)
            
        except Exception as e:
            logger.error(f"Failed to verify signature: {e}")
            return {}

    async def handle_webhook(
        self, provider: str, payload: dict[str, Any]
    ) -> SignatureWebhookEvent:
        """
        Handle webhook from provider.
        
        Args:
            provider: Provider name
            payload: Webhook payload
            
        Returns:
            Parsed webhook event
        """
        provider_instance = self.get_provider(provider)
        
        try:
            event = provider_instance.parse_webhook(payload)
            
            # Update status in database
            await self._handle_webhook_event(event)
            
            logger.info(f"Processed webhook event: {event.event_type}")
            return event
            
        except Exception as e:
            logger.error(f"Failed to handle webhook: {e}")
            raise

    async def get_embedded_signing_url(
        self,
        request_id: str,
        signer_email: str,
        return_url: str,
        provider: Optional[str] = None,
    ) -> str:
        """
        Get embedded signing URL.
        
        Args:
            request_id: Request ID
            signer_email: Signer email
            return_url: Return URL after signing
            provider: Provider name (optional)
            
        Returns:
            Embedded signing URL
        """
        provider_instance = self.get_provider(provider)
        
        try:
            url = await provider_instance.get_embedded_signing_url(
                request_id, signer_email, return_url
            )
            
            logger.info(f"Generated embedded signing URL for {signer_email}")
            return url
            
        except Exception as e:
            logger.error(f"Failed to get embedded signing URL: {e}")
            raise

    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

    def supports_feature(self, feature: str, provider: Optional[str] = None) -> bool:
        """
        Check if feature is supported.
        
        Args:
            feature: Feature name
            provider: Provider name (optional)
            
        Returns:
            True if supported
        """
        provider_instance = self.get_provider(provider)
        return provider_instance.supports_feature(feature)

    async def _store_signature_request(
        self, response: SignatureResponse, request: SignatureRequest
    ):
        """Store signature request in database."""
        # TODO: Implement database storage
        pass

    async def _update_signature_status(
        self, request_id: str, status: SignatureResponse
    ):
        """Update signature status in database."""
        # TODO: Implement database update
        pass

    async def _handle_webhook_event(self, event: SignatureWebhookEvent):
        """Handle webhook event."""
        # TODO: Implement webhook event handling
        # - Update database
        # - Trigger notifications
        # - Update document status
        pass
