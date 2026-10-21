"""
Application version metadata.

This module provides a single source of truth for the backend
application version and build information.
"""

from __future__ import annotations

from typing import Final


# ============================================================================
# APPLICATION VERSION
# ============================================================================

__version__: Final[str] = "1.0.0"

VERSION: Final[str] = __version__


# ============================================================================
# APPLICATION METADATA
# ============================================================================

APP_NAME: Final[str] = (
    "Enterprise Legal Document Intelligence AI"
)

APP_DESCRIPTION: Final[str] = (
    "Enterprise AI platform for legal document intelligence, "
    "contract lifecycle management, clause analysis, legal RAG, "
    "risk detection, expiry tracking, renewal intelligence, "
    "and human-in-the-loop legal review."
)

API_VERSION: Final[str] = "v1"


# ============================================================================
# BUILD METADATA
# ============================================================================

BUILD_VERSION: Final[str] = "1.0.0"

BUILD_ENVIRONMENT: Final[str] = "development"

BUILD_COMMIT: Final[str] = "unknown"

BUILD_TIMESTAMP: Final[str] = "unknown"


# ============================================================================
# PROJECT METADATA
# ============================================================================

PROJECT_NAME: Final[str] = (
    "enterprise-legal-document-intelligence-ai"
)

PROJECT_AUTHOR: Final[str] = "Enterprise Legal AI Team"

PROJECT_LICENSE: Final[str] = "MIT"


# ============================================================================
# VERSION HELPERS
# ============================================================================

def get_version() -> str:
    """
    Return the current application version.
    """

    return __version__


def get_version_info() -> dict[str, str]:
    """
    Return structured application version information.
    """

    return {
        "name": APP_NAME,
        "version": __version__,
        "api_version": API_VERSION,
        "build_version": BUILD_VERSION,
        "environment": BUILD_ENVIRONMENT,
        "commit": BUILD_COMMIT,
        "build_timestamp": BUILD_TIMESTAMP,
    }


__all__ = [
    "__version__",
    "VERSION",
    "APP_NAME",
    "APP_DESCRIPTION",
    "API_VERSION",
    "BUILD_VERSION",
    "BUILD_ENVIRONMENT",
    "BUILD_COMMIT",
    "BUILD_TIMESTAMP",
    "PROJECT_NAME",
    "PROJECT_AUTHOR",
    "PROJECT_LICENSE",
    "get_version",
    "get_version_info",
]