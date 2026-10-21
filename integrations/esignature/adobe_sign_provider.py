"""
Adobe Sign E-Signature Provider

Integration with Adobe Sign (formerly EchoSign) for electronic signatures.
"""

import base64
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


class AdobeSignProvider(BaseESignatureProvider):
    """
    Adobe Sign e-signature provider.
    
    Features:
    - Part of Adobe Document Cloud
    - Integration with Adobe PDF tools
    - Advanced workflow automation
    - Enterprise-grade security
    - Global compliance
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize Adobe Sign provider."""
        super().__init__(config)
        self.provider_name = "adobe_sign"
        
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.na1.adobesign.com/api/rest/v6")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def create_signature_request(
        self, request: SignatureRequest, document_path: str
    ) -> SignatureResponse:
        """Create Adobe Sign agreement."""
        try:
            async with httpx.AsyncClient() as client:
                # Step 1: Upload document
                with open(document_path, "rb") as f:
                    document_content = f.read()
                
                upload_url = f"{self.base_url}/transientDocuments"
                files = {
                    "File": (request.document_name, document_content, "application/pdf")
                }
                
                upload_response = await client.post(
                    upload_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                )
                upload_response.raise_for_status()
                transient_document_id = upload_response.json()["transientDocumentId"]
                
                # Step 2: Create agreement
                agreement_data = {
                    "fileInfos": [
                        {
                            "transientDocumentId": transient_document_id,
                        }
                    ],
                    "name": request.document_name,
                    "participantSetsInfo": [
                        {
                            "order": signer.order,
                            "role": "SIGNER",
                            "memberInfos": [
                                {
                                    "email": signer.email,
                                    "name": signer.name,
                                }
                            ],
                        }
                        for signer in request.signers
                    ],
                    "signatureType": "ESIGN",
                    "state": "IN_PROCESS",
                    "emailOption": {
                        "sendOptions": {
                            "completionEmails": "ALL",
                            "inFlightEmails": "ALL",
                            "initEmails": "ALL",
                        }
                    },
                    "message": request.message,
                    "externalId": {
                        "id": request.document_id,
                    },
                }
                
                if request.expiration_days:
                    agreement_data["daysUntilSigningDeadline"] = request.expiration_days
                
                create_url = f"{self.base_url}/agreements"
                create_response = await client.post(
                    create_url,
                    headers=self.headers,
                    json=agreement_data,
                )
                create_response.raise_for_status()
                result = create_response.json()
                
                agreement_id = result["id"]
                
                response = SignatureResponse(
                    request_id=agreement_id,
                    provider="adobe_sign",
                    status=SignatureStatus.SENT,
                    document_id=request.document_id,
                    envelope_id=agreement_id,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=request.expiration_days) if request.expiration_days else None,
                    signers_status={s.email: "sent" for s in request.signers},
                )
                
                logger.info(f"Adobe Sign agreement created: {agreement_id}")
                return response
                
        except Exception as e:
            logger.error(f"Adobe Sign API error: {e}")
            raise RuntimeError(f"Failed to create Adobe Sign agreement: {e}")

    async def get_signature_status(self, request_id: str) -> SignatureResponse:
        """Get Adobe Sign agreement status."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/agreements/{request_id}"
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                agreement = response.json()
                
                # Map Adobe Sign status to our status
                status_map = {
                    "OUT_FOR_SIGNATURE": SignatureStatus.SENT,
                    "SIGNED": SignatureStatus.SIGNED,
                    "APPROVED": SignatureStatus.COMPLETED,
                    "DELIVERED": SignatureStatus.DELIVERED,
                    "EXPIRED": SignatureStatus.EXPIRED,
                    "CANCELLED": SignatureStatus.CANCELLED,
                    "DECLINED": SignatureStatus.DECLINED,
                }
                
                status = status_map.get(
                    agreement.get("status"), SignatureStatus.SENT
                )
                
                # Get participants
                participants_url = f"{self.base_url}/agreements/{request_id}/members"
                participants_response = await client.get(
                    participants_url, headers=self.headers
                )
                participants_response.raise_for_status()
                participants = participants_response.json()
                
                signers_status = {}
                for participant_set in participants.get("participantSets", []):
                    for member in participant_set.get("memberInfos", []):
                        signers_status[member["email"]] = member.get("status", "")
                
                return SignatureResponse(
                    request_id=request_id,
                    provider="adobe_sign",
                    status=status,
                    document_id="",
                    envelope_id=request_id,
                    created_at=datetime.fromisoformat(agreement.get("createdDate", "").replace("Z", "+00:00")),
                    signers_status=signers_status,
                )
                
        except Exception as e:
            logger.error(f"Adobe Sign API error: {e}")
            raise RuntimeError(f"Failed to get agreement status: {e}")

    async def download_signed_document(
        self, request_id: str, output_path: str
    ) -> str:
        """Download signed document from Adobe Sign."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/agreements/{request_id}/combinedDocument"
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded signed document: {output_path}")
                return output_path
                
        except Exception as e:
            logger.error(f"Adobe Sign API error: {e}")
            raise RuntimeError(f"Failed to download document: {e}")

    async def cancel_signature_request(self, request_id: str, reason: str) -> bool:
        """Cancel Adobe Sign agreement."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/agreements/{request_id}/state"
                data = {
                    "state": "CANCELLED",
                    "agreementCancellationInfo": {
                        "comment": reason,
                        "notifyOthers": True,
                    },
                }
                
                response = await client.put(url, headers=self.headers, json=data)
                response.raise_for_status()
                
                logger.info(f"Cancelled agreement: {request_id}")
                return True
                
        except Exception as e:
            logger.error(f"Adobe Sign API error: {e}")
            return False

    async def send_reminder(self, request_id: str, signer_email: str) -> bool:
        """Send reminder to signer."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/agreements/{request_id}/reminders"
                data = {
                    "recipientParticipantIds": [signer_email],
                    "agreementId": request_id,
                }
                
                response = await client.post(url, headers=self.headers, json=data)
                response.raise_for_status()
                
                logger.info(f"Sent reminder to {signer_email}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to send reminder: {e}")
            return False

    async def get_audit_trail(self, request_id: str) -> dict[str, Any]:
        """Get Adobe Sign audit trail."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/agreements/{request_id}/auditTrail"
                response = await client.get(url, headers=self.headers)
                response.raise_for_status()
                
                audit_data = response.content
                
                return {
                    "agreement_id": request_id,
                    "audit_trail": base64.b64encode(audit_data).decode(),
                }
                
        except Exception as e:
            logger.error(f"Adobe Sign API error: {e}")
            return {}

    async def verify_signature(self, request_id: str) -> dict[str, Any]:
        """Verify Adobe Sign signatures."""
        try:
            status = await self.get_signature_status(request_id)
            
            return {
                "agreement_id": request_id,
                "status": status.status,
                "completed": status.status == SignatureStatus.COMPLETED,
                "signatures": status.signers_status,
                "verified": status.status == SignatureStatus.COMPLETED,
            }
            
        except Exception as e:
            logger.error(f"Adobe Sign API error: {e}")
            return {}

    def parse_webhook(self, payload: dict[str, Any]) -> SignatureWebhookEvent:
        """Parse Adobe Sign webhook."""
        event = payload.get("event")
        agreement = payload.get("agreement", {})
        
        status_map = {
            "AGREEMENT_CREATED": SignatureStatus.DRAFT,
            "AGREEMENT_ACTION_DELEGATED": SignatureStatus.SENT,
            "AGREEMENT_RECALLED": SignatureStatus.CANCELLED,
            "AGREEMENT_REJECTED": SignatureStatus.DECLINED,
            "AGREEMENT_EXPIRED": SignatureStatus.EXPIRED,
            "AGREEMENT_ACTION_COMPLETED": SignatureStatus.COMPLETED,
        }
        
        status = status_map.get(event, SignatureStatus.SENT)
        
        return SignatureWebhookEvent(
            event_type=event,
            envelope_id=agreement.get("id"),
            status=status,
            timestamp=datetime.fromisoformat(
                payload.get("eventDate", "").replace("Z", "+00:00")
            ),
            metadata=payload,
        )

    async def get_embedded_signing_url(
        self, request_id: str, signer_email: str, return_url: str
    ) -> str:
        """Get embedded signing URL."""
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/agreements/{request_id}/views"
                data = {
                    "name": "ALL",
                    "commonViewConfiguration": {
                        "autoLoginUser": True,
                        "noChrome": False,
                    },
                }
                
                response = await client.post(url, headers=self.headers, json=data)
                response.raise_for_status()
                result = response.json()
                
                return result.get("url", "")
                
        except Exception as e:
            logger.error(f"Adobe Sign API error: {e}")
            raise RuntimeError(f"Failed to get embedded signing URL: {e}")

    def supports_feature(self, feature: str) -> bool:
        """Check if Adobe Sign supports feature."""
        features = {
            "embedded_signing": True,
            "bulk_send": True,
            "templates": True,
            "advanced_fields": True,
            "webhooks": True,
            "sms_authentication": True,
            "id_verification": True,
            "payment_collection": False,
        }
        return features.get(feature, False)
