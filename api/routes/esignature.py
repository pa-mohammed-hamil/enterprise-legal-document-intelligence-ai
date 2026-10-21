"""
E-Signature API Routes

FastAPI routes for electronic signature management.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.app.dependencies import get_current_user, get_esignature_service
from backend.integrations.esignature.base_provider import (
    Signer,
    SignatureField,
    SignatureRequest,
    SignatureStatus,
)
from backend.integrations.esignature.esignature_service import ESignatureService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/esignature", tags=["esignature"])


# Request Models
class CreateSignatureRequestModel(BaseModel):
    """Create signature request model."""
    document_id: str
    document_name: str
    subject: str
    message: str
    signers: list[dict[str, Any]]
    cc_recipients: list[str] = Field(default_factory=list)
    signature_fields: list[dict[str, Any]] = Field(default_factory=list)
    expiration_days: int = 30
    reminder_frequency: int = 7
    provider: Optional[str] = None


class CancelSignatureRequestModel(BaseModel):
    """Cancel signature request model."""
    reason: str


class SendReminderModel(BaseModel):
    """Send reminder model."""
    signer_email: str


class EmbeddedSigningModel(BaseModel):
    """Embedded signing model."""
    signer_email: str
    return_url: str


# Response Models
class SignatureResponseModel(BaseModel):
    """Signature response model."""
    request_id: str
    provider: str
    status: str
    document_id: str
    envelope_id: Optional[str]
    signing_url: Optional[str]
    created_at: str
    expires_at: Optional[str]
    signers_status: dict[str, Any]


@router.post("/requests", response_model=SignatureResponseModel)
async def create_signature_request(
    data: CreateSignatureRequestModel,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """
    Create a new e-signature request.
    
    Uploads a document and sends it for signature.
    """
    try:
        # Save uploaded file
        upload_dir = Path("uploads/esignature")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / f"{data.document_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse signers
        signers = [
            Signer(
                name=s["name"],
                email=s["email"],
                role=s.get("role", "signer"),
                order=s.get("order", 1),
            )
            for s in data.signers
        ]
        
        # Parse signature fields
        signature_fields = [
            SignatureField(
                page_number=f["page_number"],
                x_position=f["x_position"],
                y_position=f["y_position"],
                signer_email=f["signer_email"],
                field_type=f.get("field_type", "signature"),
            )
            for f in data.signature_fields
        ]
        
        # Create signature request
        request = SignatureRequest(
            document_id=data.document_id,
            document_name=data.document_name,
            subject=data.subject,
            message=data.message,
            signers=signers,
            cc_recipients=data.cc_recipients,
            signature_fields=signature_fields,
            expiration_days=data.expiration_days,
            reminder_frequency=data.reminder_frequency,
        )
        
        response = await service.create_signature_request(
            request, str(file_path), provider=data.provider
        )
        
        return SignatureResponseModel(
            request_id=response.request_id,
            provider=response.provider,
            status=response.status.value,
            document_id=response.document_id,
            envelope_id=response.envelope_id,
            signing_url=response.signing_url,
            created_at=response.created_at.isoformat(),
            expires_at=response.expires_at.isoformat() if response.expires_at else None,
            signers_status=response.signers_status,
        )
        
    except Exception as e:
        logger.error(f"Failed to create signature request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create signature request: {str(e)}",
        )


@router.get("/requests/{request_id}", response_model=SignatureResponseModel)
async def get_signature_status(
    request_id: str,
    provider: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Get signature request status."""
    try:
        response = await service.get_signature_status(request_id, provider)
        
        return SignatureResponseModel(
            request_id=response.request_id,
            provider=response.provider,
            status=response.status.value,
            document_id=response.document_id,
            envelope_id=response.envelope_id,
            signing_url=response.signing_url,
            created_at=response.created_at.isoformat(),
            expires_at=response.expires_at.isoformat() if response.expires_at else None,
            signers_status=response.signers_status,
        )
        
    except Exception as e:
        logger.error(f"Failed to get signature status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get signature status: {str(e)}",
        )


@router.post("/requests/{request_id}/cancel")
async def cancel_signature_request(
    request_id: str,
    data: CancelSignatureRequestModel,
    provider: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Cancel a pending signature request."""
    try:
        success = await service.cancel_signature_request(
            request_id, data.reason, provider
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel signature request",
            )
        
        return {"success": True, "message": "Signature request cancelled"}
        
    except Exception as e:
        logger.error(f"Failed to cancel signature request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel signature request: {str(e)}",
        )


@router.post("/requests/{request_id}/remind")
async def send_reminder(
    request_id: str,
    data: SendReminderModel,
    provider: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Send reminder to a signer."""
    try:
        success = await service.send_reminder(
            request_id, data.signer_email, provider
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send reminder",
            )
        
        return {"success": True, "message": "Reminder sent"}
        
    except Exception as e:
        logger.error(f"Failed to send reminder: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send reminder: {str(e)}",
        )


@router.get("/requests/{request_id}/download")
async def download_signed_document(
    request_id: str,
    provider: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Download signed document."""
    try:
        # Check if completed
        status_response = await service.get_signature_status(request_id, provider)
        
        if status_response.status != SignatureStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document not yet signed by all parties",
            )
        
        # Download document
        output_dir = Path("downloads/esignature")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{request_id}_signed.pdf"
        
        path = await service.download_signed_document(
            request_id, str(output_path), provider
        )
        
        return {"success": True, "download_path": path}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download document: {str(e)}",
        )


@router.get("/requests/{request_id}/audit-trail")
async def get_audit_trail(
    request_id: str,
    provider: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Get audit trail for signature request."""
    try:
        audit_trail = await service.get_audit_trail(request_id, provider)
        return audit_trail
        
    except Exception as e:
        logger.error(f"Failed to get audit trail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audit trail: {str(e)}",
        )


@router.get("/requests/{request_id}/verify")
async def verify_signature(
    request_id: str,
    provider: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Verify signatures on document."""
    try:
        verification = await service.verify_signature(request_id, provider)
        return verification
        
    except Exception as e:
        logger.error(f"Failed to verify signature: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify signature: {str(e)}",
        )


@router.post("/requests/{request_id}/embedded-signing")
async def get_embedded_signing_url(
    request_id: str,
    data: EmbeddedSigningModel,
    provider: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Get embedded signing URL for in-app signing."""
    try:
        url = await service.get_embedded_signing_url(
            request_id, data.signer_email, data.return_url, provider
        )
        
        return {"signing_url": url}
        
    except Exception as e:
        logger.error(f"Failed to get embedded signing URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embedded signing URL: {str(e)}",
        )


@router.get("/providers")
async def get_providers(
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Get list of available e-signature providers."""
    providers = service.get_available_providers()
    
    return {
        "providers": providers,
        "default_provider": service.default_provider,
    }


@router.get("/providers/{provider}/features")
async def get_provider_features(
    provider: str,
    current_user: dict = Depends(get_current_user),
    service: ESignatureService = Depends(get_esignature_service),
):
    """Get features supported by provider."""
    features = [
        "embedded_signing",
        "bulk_send",
        "templates",
        "advanced_fields",
        "webhooks",
        "sms_authentication",
        "id_verification",
        "payment_collection",
    ]
    
    supported_features = {
        feature: service.supports_feature(feature, provider)
        for feature in features
    }
    
    return {
        "provider": provider,
        "features": supported_features,
    }


@router.post("/webhooks/{provider}")
async def handle_webhook(
    provider: str,
    payload: dict[str, Any],
    service: ESignatureService = Depends(get_esignature_service),
):
    """
    Handle webhook from e-signature provider.
    
    Note: This endpoint should be configured in the provider's webhook settings.
    """
    try:
        event = await service.handle_webhook(provider, payload)
        
        logger.info(
            f"Processed webhook from {provider}: "
            f"{event.event_type} for {event.envelope_id}"
        )
        
        return {"success": True, "event_type": event.event_type}
        
    except Exception as e:
        logger.error(f"Failed to handle webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle webhook: {str(e)}",
        )
