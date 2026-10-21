"""
E-Signature Integration Module

Provides electronic signature capabilities through multiple providers.
"""

from backend.integrations.esignature.base_provider import BaseESignatureProvider
from backend.integrations.esignature.docusign_provider import DocuSignProvider
from backend.integrations.esignature.adobe_sign_provider import AdobeSignProvider
from backend.integrations.esignature.hellosign_provider import HelloSignProvider
from backend.integrations.esignature.esignature_service import ESignatureService

__all__ = [
    "BaseESignatureProvider",
    "DocuSignProvider",
    "AdobeSignProvider",
    "HelloSignProvider",
    "ESignatureService",
]
