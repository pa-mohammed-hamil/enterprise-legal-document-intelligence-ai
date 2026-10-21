"""
HelloSign (Dropbox Sign) E-Signature Provider

Integration with HelloSign/Dropbox Sign for electronic signatures.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .base_provider import (
    BaseESignatureProvider,
    SignatureRequest,
    SignatureResponse,
    SignatureStatus,
    SignatureWebhookEvent,
)

logger = logging.getLogger(__name__)


class HelloSignProvider(BaseESignatureProvider):
    """
    HelloSign (Dropbox Sign) e-signature provider.
    
    Features:
    - Simple, developer-friendly API
    - Part of Dropbox ecosystem
    - Affordable pricing
    - Easy integration
    - Mobile-optimized signing
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize HelloSign provider."""
        super().__init__(config)
        self.provider_name = "hellosign"
        
        self.api_key = config.get("api_key")
        self.client_id = config.get("client_id")
        self.base_url = "https://api.hellosign.com/v3"
        
        self.auth = (self.api_key, "")

    async def create_signature_request(
        self, request: SignatureRequest, document_path: str
    ) -> SignatureResponse:
        """Create HelloSign signature request."""
        try:
            async with httpx.AsyncClient() as client:
                # Prepare signers
                signers = []
                for idx, signer in enumerate(request.signers):
                    signers.append({
                        f"signers[{idx}][name]": signer.name,
                        f"signers[{idx}][email_address]": signer.email,
                        f"signers[{idx}][order]": signer.order,
                    })
                
                # Flatten signers dict
                signer_data = {}
                for signer in signers:
                    signer_data.update(signer)
                
                # Prepare request data
                data = {
                    "title": request.document_name,
                    "subject": request.subject,
                    "message": request.message,
                    **signer_data,
                }
                
                # Add test mode if configured
                if self.config.get("test_mode", False):
                    data["test_mode"] = 1
                
                # Prepare files
                files = {
                    "file[0]": open(document_path, "rb"),
                }
                
                url = f"{self.base_url}/signature_request/send"
                response = await client.post(
                    url,
                    auth=self.auth,
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                result = response.json()
                
                signature_request = result["signature_request"]
                signature_request_id = signature_request["signature_request_id"]
                
                response_data = SignatureResponse(
                    request_id=signature_request_id,
                    provider="hellosign",
                    status=SignatureStatus.SENT,
                    document_id=request.document_id,
                    envelope_id=signature_request_id,
                    created_at=datetime.fromtimestamp(signature_request["created_at"]),
                    signers_status={
                        sig["signer_email_address"]: sig["status_code"]
                        for sig in signature_request["signatures"]
                    },
                )
                
                logger.info(f"HelloSign request created: {signature_request_id}")
                return response_data
                
        except Exception as e:
            logger.error(f"HelloSign API error: {e}")
            raise RuntimeError(f"Failed to create HelloSign request: {e}")

    async def get_signature_status(self, request_id: str) -> SignatureResponse:
        """Get HelloSign signature request status."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/signature_request/{request_id}"
                response = await client.get(url, auth=self.auth)
                response.raise_for_status()
                result = response.json()
                
                signature_request = result["signature_request"]
                
                # Map HelloSign status to our status
                status_map = {
                    "awaiting_signature": SignatureStatus.SENT,
                    "signed": SignatureStatus.SIGNED,
                    "complete": SignatureStatus.COMPLETED,
                    "declined": SignatureStatus.DECLINED,
                    "cancelled": SignatureStatus.CANCELLED,
                }
                
                status = status_map.get(
                    signature_request.get("is_complete") and "complete" or "awaiting_signature",
                    SignatureStatus.SENT,
                )
                
                signers_status = {
                    sig["signer_email_address"]: sig["status_code"]
                    for sig in signature_request["signatures"]
                }
                
                return SignatureResponse(
                    request_id=request_id,
                    provider="hellosign",
                    status=status,
                    document_id="",
                    envelope_id=request_id,
                    created_at=datetime.fromtimestamp(signature_request["created_at"]),
                    signers_status=signers_status,
                )
                
        except Exception as e:
            logger.error(f"HelloSign API error: {e}")
            raise RuntimeError(f"Failed to get signature status: {e}")

    async def download_signed_document(
        self, request_id: str, output_path: str
    ) -> str:
        """Download signed document from HelloSign."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/signature_request/files/{request_id}"
                response = await client.get(url, auth=self.auth)
                response.raise_for_status()
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded signed document: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"HelloSign API error: {e}")
            raise RuntimeError(f"Failed to download document: {e}")

    async def cancel_signature_request(self, request_id: str, reason: str) -> bool:
        """Cancel HelloSign signature request."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/signature_request/cancel/{request_id}"
                response = await client.post(url, auth=self.auth)
                response.raise_for_status()
                
                logger.info(f"Cancelled signature request: {request_id}")
                return True
                
        except Exception as e:
            logger.error(f"HelloSign API error: {e}")
            return False

    async def send_reminder(self, request_id: str, signer_email: str) -> bool:
        """Send reminder to signer."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/signature_request/remind/{request_id}"
                data = {"email_address": signer_email}
                
                response = await client.post(url, auth=self.auth, data=data)
                response.raise_for_status()
                
                logger.info(f"Sent reminder to {signer_email}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to send reminder: {e}")
            return False

    async def get_audit_trail(self, request_id: str) -> dict[str, Any]:
        """Get HelloSign audit trail."""
        try:
            # HelloSign doesn't have a separate audit trail endpoint
            # Audit information is included in the final PDF
            status = await self.get_signature_status(request_id)
            
            return {
                "signature_request_id": request_id,
                "status": status.status,
                "signers": status.signers_status,
                "note": "Audit trail included in signed PDF",
            }
            
        except Exception as e:
            logger.error(f"HelloSign API error: {e}")
            return {}

    async def verify_signature(self, request_id: str) -> dict[str, Any]:
        """Verify HelloSign signatures."""
        try:
            status = await self.get_signature_status(request_id)
            
            signatures = []
            for email, status_code in status.signers_status.items():
                signatures.append({
                    "signer_email": email,
                    "status": status_code,
                    "verified": status_code == "signed",
                })
            
            return {
                "signature_request_id": request_id,
                "status": status.status,
                "completed": status.status == SignatureStatus.COMPLETED,
                "signatures": signatures,
            }
            
        except Exception as e:
            logger.error(f"HelloSign API error: {e}")
            return {}

    def parse_webhook(self, payload: dict[str, Any]) -> SignatureWebhookEvent:
        """Parse HelloSign webhook."""
        event = payload.get("event", {})
        event_type = event.get("event_type")
        
        signature_request = event.get("signature_request", {})
        signature_request_id = signature_request.get("signature_request_id")
        
        status_map = {
            "signature_request_sent": SignatureStatus.SENT,
            "signature_request_viewed": SignatureStatus.DELIVERED,
            "signature_request_signed": SignatureStatus.SIGNED,
            "signature_request_all_signed": SignatureStatus.COMPLETED,
            "signature_request_declined": SignatureStatus.DECLINED,
            "signature_request_cancelled": SignatureStatus.CANCELLED,
        }
        
        status = status_map.get(event_type, SignatureStatus.SENT)
        
        return SignatureWebhookEvent(
            event_type=event_type,
            envelope_id=signature_request_id,
            status=status,
            timestamp=datetime.fromtimestamp(event.get("event_time", 0)),
            metadata=payload,
        )

    async def get_embedded_signing_url(
        self, request_id: str, signer_email: str, return_url: str
    ) -> str:
        """Get embedded signing URL."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/embedded/sign_url/{request_id}"
                response = await client.get(url, auth=self.auth)
                response.raise_for_status()
                result = response.json()
                
                embedded = result.get("embedded", {})
                return embedded.get("sign_url", "")
                
        except Exception as e:
            logger.error(f"HelloSign API error: {e}")
            raise RuntimeError(f"Failed to get embedded signing URL: {e}")

    def supports_feature(self, feature: str) -> bool:
        """Check if HelloSign supports feature."""
        features = {
            "embedded_signing": True,
            "bulk_send": False,
            "templates": True,
            "advanced_fields": False,
            "webhooks": True,
            "sms_authentication": False,
            "id_verification": False,
            "payment_collection": False,
        }
        return features.get(feature, False)
