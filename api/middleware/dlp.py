"""
Data Leakage Prevention Middleware

Middleware for intercepting and preventing data leakage through API responses.
"""

import json
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.security.data_leakage_prevention import (
    DataLeakagePreventor,
    LeakageChannel,
    SensitivityLevel
)

logger = logging.getLogger(__name__)


class DLPMiddleware(BaseHTTPMiddleware):
    """
    Middleware to prevent data leakage through API responses.
    
    Features:
    - Scans API responses for sensitive data
    - Redacts sensitive information
    - Logs potential leakage attempts
    - Enforces access controls
    """

    def __init__(self, app: ASGIApp, config: dict = None):
        super().__init__(app)
        self.dlp = DataLeakagePreventor(config)
        self.enabled = config.get("dlp_enabled", True) if config else True
        self.scan_responses = config.get("scan_responses", True) if config else True
        self.auto_redact = config.get("auto_redact", True) if config else True

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Intercept and scan responses for sensitive data."""
        
        if not self.enabled:
            return await call_next(request)
        
        # Get user info from request
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Process request
        response = await call_next(request)
        
        # Scan response if configured
        if self.scan_responses and self._should_scan_response(request, response):
            response = await self._scan_and_protect_response(
                request,
                response,
                user_id
            )
        
        return response

    async def _scan_and_protect_response(
        self,
        request: Request,
        response: Response,
        user_id: str
    ) -> Response:
        """Scan and protect response content."""
        
        # Read response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk
        
        try:
            # Try to parse as JSON
            content = response_body.decode("utf-8")
            
            # Scan for sensitive data
            detection = self.dlp.scan_content(
                content,
                context={
                    "channel": LeakageChannel.API_RESPONSE.value,
                    "user_id": user_id,
                    "endpoint": str(request.url.path),
                    "method": request.method
                }
            )
            
            # Log detection
            if detection.detected:
                logger.warning(
                    f"Sensitive data in API response: "
                    f"endpoint={request.url.path}, "
                    f"user={user_id}, "
                    f"sensitivity={detection.sensitivity_level.value}, "
                    f"risk_score={detection.risk_score}"
                )
            
            # Auto-redact if configured and high risk
            if self.auto_redact and detection.risk_score > 0.7:
                if detection.redacted_content:
                    content = detection.redacted_content
                    logger.info(f"Auto-redacted API response for user {user_id}")
            
            # Add security headers
            response.headers["X-Content-Classification"] = detection.sensitivity_level.value
            response.headers["X-DLP-Scanned"] = "true"
            
            if detection.detected:
                response.headers["X-DLP-Matches"] = str(len(detection.matches))
            
            # Create new response
            return Response(
                content=content.encode("utf-8"),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        except Exception as e:
            logger.error(f"Error in DLP middleware: {e}")
            # Return original response on error
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )

    def _should_scan_response(self, request: Request, response: Response) -> bool:
        """Determine if response should be scanned."""
        # Skip non-200 responses
        if response.status_code != 200:
            return False
        
        # Skip health checks and static content
        skip_paths = ["/health", "/metrics", "/static", "/favicon.ico"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return False
        
        # Only scan text responses
        content_type = response.headers.get("content-type", "")
        if not any(t in content_type for t in ["application/json", "text/", "application/xml"]):
            return False
        
        return True


class DLPRequestValidator:
    """Validate requests for potential data leakage."""

    def __init__(self):
        self.dlp = DataLeakagePreventor()

    async def validate_export_request(
        self,
        content: str,
        user_id: str,
        format: str,
        destination: str = None
    ) -> tuple[bool, list[str]]:
        """
        Validate export request for data leakage.
        
        Returns:
            Tuple of (allowed, warnings)
        """
        allowed, _, warnings = self.dlp.prevent_leakage(
            content=content,
            channel=LeakageChannel.EXPORT,
            user_id=user_id,
            destination=destination
        )
        
        return allowed, warnings

    async def validate_share_request(
        self,
        content: str,
        user_id: str,
        recipient_email: str
    ) -> tuple[bool, list[str]]:
        """
        Validate document sharing request.
        
        Returns:
            Tuple of (allowed, warnings)
        """
        # Check recipient domain
        domain = recipient_email.split("@")[-1] if "@" in recipient_email else ""
        
        allowed, _, warnings = self.dlp.prevent_leakage(
            content=content,
            channel=LeakageChannel.EXTERNAL_SHARE,
            user_id=user_id,
            destination=domain
        )
        
        return allowed, warnings

    async def validate_email_request(
        self,
        content: str,
        user_id: str,
        to_addresses: list[str],
        attachments: list = None
    ) -> tuple[bool, list[str]]:
        """
        Validate email sending request.
        
        Returns:
            Tuple of (allowed, warnings)
        """
        warnings = []
        
        # Check email content
        allowed, modified_content, content_warnings = self.dlp.prevent_leakage(
            content=content,
            channel=LeakageChannel.EMAIL,
            user_id=user_id,
            destination=",".join(to_addresses)
        )
        
        warnings.extend(content_warnings)
        
        if not allowed:
            return False, warnings
        
        # Check attachments if present
        if attachments:
            for attachment in attachments:
                attachment_content = attachment.get("content", "")
                att_allowed, _, att_warnings = self.dlp.prevent_leakage(
                    content=attachment_content,
                    channel=LeakageChannel.EMAIL,
                    user_id=user_id,
                    destination=",".join(to_addresses)
                )
                
                if not att_allowed:
                    return False, [f"Attachment blocked: {att_warnings}"]
                
                warnings.extend(att_warnings)
        
        return True, warnings
