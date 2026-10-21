"""
Seed the Enterprise Legal Document Intelligence AI database.

Purpose:
    - Create development/demo organizations
    - Create development/demo users
    - Create roles and permissions
    - Create representative legal documents
    - Create contracts, clauses, obligations, risks, renewals
    - Create review tasks and notifications
    - Create audit records
    - Create model-run records
    - Support deterministic, idempotent seeding

IMPORTANT:
    This script is intended for local development, testing, and demos.
    Do not run against production databases unless explicitly intended.

Usage:
    python scripts/seed_database.py

    python scripts/seed_database.py --reset

    python scripts/seed_database.py --minimal

    python scripts/seed_database.py --dry-run

    python scripts/seed_database.py --tenant-id demo-tenant

Environment:
    DATABASE_URL
    DEFAULT_TENANT_ID
    DEFAULT_ORGANIZATION_ID
    SEED_ADMIN_EMAIL
    SEED_ADMIN_PASSWORD
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(
        PROJECT_ROOT / ".env",
        override=False,
    )
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    ),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "legal-database-seeder"
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TENANT_ID = os.getenv(
    "DEFAULT_TENANT_ID",
    "demo-tenant",
)

DEFAULT_ORGANIZATION_ID = os.getenv(
    "DEFAULT_ORGANIZATION_ID",
    "demo-organization",
)

DEFAULT_ADMIN_EMAIL = os.getenv(
    "SEED_ADMIN_EMAIL",
    "admin@legal-ai.local",
)

DEFAULT_ADMIN_PASSWORD = os.getenv(
    "SEED_ADMIN_PASSWORD",
    "ChangeMe-LocalOnly-123!",
)

DEFAULT_USER_PASSWORD = os.getenv(
    "SEED_USER_PASSWORD",
    "ChangeMe-LocalOnly-123!",
)

# ---------------------------------------------------------------------------
# Seed identifiers
# ---------------------------------------------------------------------------

SEED_NAMESPACE = uuid.UUID(
    "8f9d1d5a-1a48-4b32-a8bb-9c6f9f0c0e71"
)


def deterministic_uuid(
    name: str,
) -> str:
    return str(
        uuid.uuid5(
            SEED_NAMESPACE,
            name,
        )
    )


TENANT_ID = deterministic_uuid(
    f"tenant:{DEFAULT_TENANT_ID}"
)

ORGANIZATION_ID = deterministic_uuid(
    f"organization:{DEFAULT_ORGANIZATION_ID}"
)

ADMIN_USER_ID = deterministic_uuid(
    "user:admin@legal-ai.local"
)

LEGAL_USER_ID = deterministic_uuid(
    "user:legal@legal-ai.local"
)

REVIEWER_USER_ID = deterministic_uuid(
    "user:reviewer@legal-ai.local"
)

ANALYST_USER_ID = deterministic_uuid(
    "user:analyst@legal-ai.local"
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SeedStats:
    tenants: int = 0
    organizations: int = 0
    roles: int = 0
    users: int = 0
    documents: int = 0
    document_versions: int = 0
    chunks: int = 0
    contracts: int = 0
    clauses: int = 0
    obligations: int = 0
    risks: int = 0
    renewals: int = 0
    reviews: int = 0
    notifications: int = 0
    audit_logs: int = 0
    model_runs: int = 0


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def hash_password(
    password: str,
) -> str:
    """
    Hash a local seed password.

    Prefer the application's password hashing utility
    when available. PBKDF2 is used only as a fallback.
    """

    try:
        from backend.core.security.password import (
            hash_password as application_hash_password,
        )

        return application_hash_password(
            password
        )

    except Exception:
        pass

    salt = os.urandom(16)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        310_000,
    )

    return (
        "pbkdf2_sha256$310000$"
        f"{salt.hex()}${derived.hex()}"
    )


def safe_repr(
    value: Any,
) -> str:
    return repr(value)


# ---------------------------------------------------------------------------
# Database imports
# ---------------------------------------------------------------------------


def load_database_components() -> tuple[
    Any,
    Any,
    dict[str, Any],
]:
    """
    Load the project's SQLAlchemy session and models.

    Model imports are intentionally defensive so the seed
    script remains easy to adapt when individual model
    filenames differ.
    """

    try:
        database_module = importlib.import_module(
            "backend.db.database"
        )
    except ImportError:
        database_module = importlib.import_module(
            "backend.db.session"
        )

    session_factory = (
        getattr(
            database_module,
            "SessionLocal",
            None,
        )
        or getattr(
            database_module,
            "sessionmaker",
            None,
        )
    )

    if session_factory is None:
        raise RuntimeError(
            "Unable to locate SessionLocal/sessionmaker "
            "in backend.db.database or backend.db.session."
        )

    model_module = importlib.import_module(
        "backend.db.models"
    )

    models: dict[str, Any] = {}

    model_names = (
        "Tenant",
        "Organization",
        "Role",
        "User",
        "Document",
        "DocumentVersion",
        "DocumentChunk",
        "Contract",
        "Clause",
        "Obligation",
        "Risk",
        "Renewal",
        "Review",
        "Notification",
        "AuditLog",
        "ModelRun",
    )

    for name in model_names:
        model = getattr(
            model_module,
            name,
            None,
        )

        if model is not None:
            models[name] = model

    return (
        session_factory,
        database_module,
        models,
    )


# ---------------------------------------------------------------------------
# Generic model helpers
# ---------------------------------------------------------------------------


def model_columns(
    model: Any,
) -> set[str]:
    try:
        return {
            column.name
            for column in model.__table__.columns
        }
    except Exception:
        return set()


def build_model_kwargs(
    model: Any,
    values: dict[str, Any],
) -> dict[str, Any]:
    """
    Only pass fields that actually exist on the model.

    This makes the seed script resilient to additional
    optional columns being added to the schema.
    """

    columns = model_columns(
        model
    )

    if not columns:
        return values

    return {
        key: value
        for key, value in values.items()
        if key in columns
    }


def create_model(
    session: Any,
    model: Any,
    values: dict[str, Any],
) -> Any:
    kwargs = build_model_kwargs(
        model,
        values,
    )

    instance = model(
        **kwargs
    )

    session.add(
        instance
    )

    return instance


def query_by_id(
    session: Any,
    model: Any,
    object_id: str,
) -> Any | None:
    try:
        return session.get(
            model,
            object_id,
        )
    except Exception:
        return None


def get_existing_by_email(
    session: Any,
    model: Any,
    email: str,
) -> Any | None:
    if model is None:
        return None

    try:
        return (
            session.query(model)
            .filter(
                model.email == email
            )
            .first()
        )
    except Exception:
        return None


def get_existing_by_name(
    session: Any,
    model: Any,
    name: str,
) -> Any | None:
    if model is None:
        return None

    try:
        return (
            session.query(model)
            .filter(
                model.name == name
            )
            .first()
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tenant
# ---------------------------------------------------------------------------


def seed_tenant(
    session: Any,
    models: dict[str, Any],
    stats: SeedStats,
) -> Any | None:
    model = models.get(
        "Tenant"
    )

    if model is None:
        logger.warning(
            "Tenant model not found; skipping."
        )
        return None

    existing = query_by_id(
        session,
        model,
        TENANT_ID,
    )

    if existing:
        return existing

    tenant = create_model(
        session,
        model,
        {
            "id": TENANT_ID,
            "name": (
                "Enterprise Legal AI Demo Tenant"
            ),
            "slug": DEFAULT_TENANT_ID,
            "status": "active",
            "is_active": True,
            "metadata": {
                "environment": "local",
                "seeded": True,
            },
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
    )

    stats.tenants += 1

    return tenant


# ---------------------------------------------------------------------------
# Organization
# ---------------------------------------------------------------------------


def seed_organization(
    session: Any,
    models: dict[str, Any],
    stats: SeedStats,
) -> Any | None:
    model = models.get(
        "Organization"
    )

    if model is None:
        logger.warning(
            "Organization model not found; skipping."
        )
        return None

    existing = query_by_id(
        session,
        model,
        ORGANIZATION_ID,
    )

    if existing:
        return existing

    organization = create_model(
        session,
        model,
        {
            "id": ORGANIZATION_ID,
            "tenant_id": TENANT_ID,
            "name": (
                "Enterprise Legal AI Demo Organization"
            ),
            "slug": DEFAULT_ORGANIZATION_ID,
            "status": "active",
            "is_active": True,
            "metadata": {
                "environment": "local",
                "seeded": True,
            },
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
    )

    stats.organizations += 1

    return organization


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------


ROLE_DEFINITIONS = (
    (
        "administrator",
        "Full local development administrator",
        [
            "users:read",
            "users:write",
            "documents:read",
            "documents:write",
            "contracts:read",
            "contracts:write",
            "clauses:read",
            "clauses:write",
            "risk:read",
            "risk:write",
            "reviews:read",
            "reviews:write",
            "audit:read",
            "models:read",
            "models:run",
        ],
    ),
    (
        "legal_counsel",
        "Legal counsel and contract analyst",
        [
            "documents:read",
            "documents:write",
            "contracts:read",
            "contracts:write",
            "clauses:read",
            "clauses:write",
            "risk:read",
            "risk:write",
            "reviews:read",
            "reviews:write",
        ],
    ),
    (
        "reviewer",
        "Human legal reviewer",
        [
            "documents:read",
            "contracts:read",
            "clauses:read",
            "risk:read",
            "reviews:read",
            "reviews:write",
        ],
    ),
    (
        "analyst",
        "Read-only legal intelligence analyst",
        [
            "documents:read",
            "contracts:read",
            "clauses:read",
            "risk:read",
            "analytics:read",
        ],
    ),
)


def seed_roles(
    session: Any,
    models: dict[str, Any],
    stats: SeedStats,
) -> dict[str, Any]:
    model = models.get(
        "Role"
    )

    roles: dict[str, Any] = {}

    if model is None:
        logger.warning(
            "Role model not found; skipping."
        )
        return roles

    for name, description, permissions in ROLE_DEFINITIONS:
        existing = get_existing_by_name(
            session,
            model,
            name,
        )

        if existing:
            roles[name] = existing
            continue

        role_id = deterministic_uuid(
            f"role:{name}"
        )

        role = create_model(
            session,
            model,
            {
                "id": role_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "name": name,
                "description": description,
                "permissions": permissions,
                "is_active": True,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        roles[name] = role
        stats.roles += 1

    return roles


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


USER_DEFINITIONS = (
    (
        ADMIN_USER_ID,
        "admin@legal-ai.local",
        "System",
        "Administrator",
        "administrator",
        DEFAULT_ADMIN_PASSWORD,
    ),
    (
        LEGAL_USER_ID,
        "legal@legal-ai.local",
        "Alex",
        "Counsel",
        "legal_counsel",
        DEFAULT_USER_PASSWORD,
    ),
    (
        REVIEWER_USER_ID,
        "reviewer@legal-ai.local",
        "Jordan",
        "Reviewer",
        "reviewer",
        DEFAULT_USER_PASSWORD,
    ),
    (
        ANALYST_USER_ID,
        "analyst@legal-ai.local",
        "Taylor",
        "Analyst",
        "analyst",
        DEFAULT_USER_PASSWORD,
    ),
)


def seed_users(
    session: Any,
    models: dict[str, Any],
    roles: dict[str, Any],
    stats: SeedStats,
) -> dict[str, Any]:
    model = models.get(
        "User"
    )

    users: dict[str, Any] = {}

    if model is None:
        logger.warning(
            "User model not found; skipping."
        )
        return users

    for (
        user_id,
        email,
        first_name,
        last_name,
        role_name,
        password,
    ) in USER_DEFINITIONS:
        existing = get_existing_by_email(
            session,
            model,
            email,
        )

        if existing:
            users[email] = existing
            continue

        role = roles.get(
            role_name
        )

        user = create_model(
            session,
            model,
            {
                "id": user_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "role_id": (
                    getattr(
                        role,
                        "id",
                        None,
                    )
                    if role
                    else None
                ),
                "email": email,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": (
                    f"{first_name} "
                    f"{last_name}"
                ),
                "hashed_password": hash_password(
                    password
                ),
                "password_hash": hash_password(
                    password
                ),
                "is_active": True,
                "is_verified": True,
                "status": "active",
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        users[email] = user
        stats.users += 1

    return users


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


DOCUMENT_DEFINITIONS = (
    {
        "key": "msa",
        "filename": "Acme_Master_Service_Agreement.pdf",
        "title": "Master Service Agreement",
        "document_type": "contract",
        "status": "active",
        "description": (
            "Master agreement governing software and "
            "professional services."
        ),
    },
    {
        "key": "nda",
        "filename": "Acme_Mutual_NDA.pdf",
        "title": "Mutual Non-Disclosure Agreement",
        "document_type": "nda",
        "status": "active",
        "description": (
            "Mutual confidentiality agreement."
        ),
    },
    {
        "key": "employment",
        "filename": "Senior_Engineer_Employment_Agreement.docx",
        "title": "Senior Engineer Employment Agreement",
        "document_type": "employment",
        "status": "active",
        "description": (
            "Employment agreement with compensation, "
            "confidentiality and termination terms."
        ),
    },
    {
        "key": "dpa",
        "filename": "Data_Processing_Addendum.pdf",
        "title": "Data Processing Addendum",
        "document_type": "privacy",
        "status": "active",
        "description": (
            "Data protection and processor obligations."
        ),
    },
)


def seed_documents(
    session: Any,
    models: dict[str, Any],
    stats: SeedStats,
) -> dict[str, Any]:
    model = models.get(
        "Document"
    )

    documents: dict[str, Any] = {}

    if model is None:
        logger.warning(
            "Document model not found; skipping."
        )
        return documents

    for definition in DOCUMENT_DEFINITIONS:
        document_id = deterministic_uuid(
            f"document:{definition['key']}"
        )

        existing = query_by_id(
            session,
            model,
            document_id,
        )

        if existing:
            documents[
                definition["key"]
            ] = existing
            continue

        document = create_model(
            session,
            model,
            {
                "id": document_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "name": definition["filename"],
                "filename": definition["filename"],
                "original_filename": definition[
                    "filename"
                ],
                "title": definition["title"],
                "document_type": definition[
                    "document_type"
                ],
                "status": definition["status"],
                "description": definition[
                    "description"
                ],
                "source": "seed",
                "mime_type": (
                    "application/pdf"
                    if definition["filename"].endswith(
                        ".pdf"
                    )
                    else (
                        "application/"
                        "vnd.openxmlformats-officedocument."
                        "wordprocessingml.document"
                    )
                ),
                "metadata": {
                    "seeded": True,
                    "demo": True,
                    "classification": (
                        definition[
                            "document_type"
                        ]
                    ),
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        documents[
            definition["key"]
        ] = document

        stats.documents += 1

    return documents


# ---------------------------------------------------------------------------
# Document versions
# ---------------------------------------------------------------------------


def seed_document_versions(
    session: Any,
    models: dict[str, Any],
    documents: dict[str, Any],
    stats: SeedStats,
) -> dict[str, Any]:
    model = models.get(
        "DocumentVersion"
    )

    versions: dict[str, Any] = {}

    if model is None:
        logger.warning(
            "DocumentVersion model not found; skipping."
        )
        return versions

    for key, document in documents.items():
        document_id = getattr(
            document,
            "id",
            None,
        )

        if not document_id:
            continue

        version_id = deterministic_uuid(
            f"version:{key}:1"
        )

        existing = query_by_id(
            session,
            model,
            version_id,
        )

        if existing:
            versions[key] = existing
            continue

        version = create_model(
            session,
            model,
            {
                "id": version_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "document_id": document_id,
                "version_number": 1,
                "version": 1,
                "status": "current",
                "is_current": True,
                "filename": getattr(
                    document,
                    "filename",
                    None,
                ),
                "storage_path": (
                    f"seed/{key}/"
                    f"version-1"
                ),
                "checksum": hashlib.sha256(
                    key.encode()
                ).hexdigest(),
                "metadata": {
                    "seeded": True,
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        versions[key] = version
        stats.document_versions += 1

    return versions


# ---------------------------------------------------------------------------
# Document chunks
# ---------------------------------------------------------------------------


CHUNK_CONTENT = {
    "msa": (
        "The parties agree to provide software services "
        "under the terms of this Master Service Agreement. "
        "Either party may terminate for material breach "
        "following the applicable cure period."
    ),
    "nda": (
        "Each party shall protect confidential information "
        "received from the other party and shall use such "
        "information solely for the permitted business purpose."
    ),
    "employment": (
        "The employee shall maintain confidentiality and "
        "comply with company information security policies. "
        "Termination rights are governed by the employment terms."
    ),
    "dpa": (
        "The processor shall implement appropriate technical "
        "and organizational measures and shall notify the "
        "controller of qualifying personal data incidents."
    ),
}


def seed_chunks(
    session: Any,
    models: dict[str, Any],
    versions: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "DocumentChunk"
    )

    if model is None:
        logger.warning(
            "DocumentChunk model not found; skipping."
        )
        return

    for key, version in versions.items():
        version_id = getattr(
            version,
            "id",
            None,
        )

        if not version_id:
            continue

        content = CHUNK_CONTENT.get(
            key,
            "Seed legal document content.",
        )

        chunk_id = deterministic_uuid(
            f"chunk:{key}:1"
        )

        existing = query_by_id(
            session,
            model,
            chunk_id,
        )

        if existing:
            continue

        create_model(
            session,
            model,
            {
                "id": chunk_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "document_version_id": version_id,
                "chunk_index": 0,
                "content": content,
                "text": content,
                "token_count": len(
                    content.split()
                ),
                "metadata": {
                    "seeded": True,
                    "page": 1,
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        stats.chunks += 1


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


def seed_contracts(
    session: Any,
    models: dict[str, Any],
    documents: dict[str, Any],
    stats: SeedStats,
) -> dict[str, Any]:
    model = models.get(
        "Contract"
    )

    contracts: dict[str, Any] = {}

    if model is None:
        logger.warning(
            "Contract model not found; skipping."
        )
        return contracts

    contract_definitions = (
        (
            "msa",
            "Acme Software Master Services Agreement",
            "Acme Corporation",
            "Enterprise Legal AI Ltd.",
            365,
        ),
        (
            "nda",
            "Acme Mutual Confidentiality Agreement",
            "Acme Corporation",
            "Enterprise Legal AI Ltd.",
            730,
        ),
        (
            "employment",
            "Senior Engineer Employment Agreement",
            "Enterprise Legal AI Ltd.",
            "Demo Employee",
            365,
        ),
        (
            "dpa",
            "Data Processing Addendum",
            "Acme Corporation",
            "Enterprise Legal AI Ltd.",
            365,
        ),
    )

    for (
        key,
        title,
        counterparty,
        internal_party,
        duration_days,
    ) in contract_definitions:
        contract_id = deterministic_uuid(
            f"contract:{key}"
        )

        existing = query_by_id(
            session,
            model,
            contract_id,
        )

        if existing:
            contracts[key] = existing
            continue

        document = documents.get(
            key
        )

        contract = create_model(
            session,
            model,
            {
                "id": contract_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "document_id": getattr(
                    document,
                    "id",
                    None,
                ),
                "title": title,
                "name": title,
                "contract_type": (
                    "service"
                    if key == "msa"
                    else key
                ),
                "status": "active",
                "counterparty": counterparty,
                "party_name": counterparty,
                "internal_party": internal_party,
                "effective_date": (
                    utc_now().date()
                    - timedelta(days=30)
                ),
                "expiration_date": (
                    utc_now().date()
                    + timedelta(days=duration_days)
                ),
                "governing_law": (
                    "State of New York"
                    if key == "msa"
                    else "Applicable Jurisdiction"
                ),
                "metadata": {
                    "seeded": True,
                    "demo": True,
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        contracts[key] = contract
        stats.contracts += 1

    return contracts


# ---------------------------------------------------------------------------
# Clauses
# ---------------------------------------------------------------------------


CLAUSE_DEFINITIONS = (
    (
        "msa",
        "termination",
        "Termination for Cause",
        "Either party may terminate this agreement for material breach after thirty days written notice and opportunity to cure.",
        "medium",
    ),
    (
        "msa",
        "limitation_of_liability",
        "Limitation of Liability",
        "Except for specified excluded claims, aggregate liability shall not exceed fees paid during the applicable twelve-month period.",
        "high",
    ),
    (
        "msa",
        "confidentiality",
        "Confidentiality",
        "Each party shall protect confidential information and use it only for permitted purposes.",
        "medium",
    ),
    (
        "nda",
        "confidentiality",
        "Mutual Confidentiality",
        "Recipients must maintain confidential information using reasonable security measures.",
        "high",
    ),
    (
        "employment",
        "termination",
        "Termination Notice",
        "Either party may terminate employment subject to applicable notice requirements.",
        "medium",
    ),
    (
        "dpa",
        "data_protection",
        "Security Measures",
        "The processor shall implement appropriate technical and organizational safeguards.",
        "high",
    ),
)


def seed_clauses(
    session: Any,
    models: dict[str, Any],
    contracts: dict[str, Any],
    stats: SeedStats,
) -> dict[str, Any]:
    model = models.get(
        "Clause"
    )

    clauses: dict[str, Any] = {}

    if model is None:
        logger.warning(
            "Clause model not found; skipping."
        )
        return clauses

    for (
        contract_key,
        clause_type,
        title,
        text,
        severity,
    ) in CLAUSE_DEFINITIONS:
        contract = contracts.get(
            contract_key
        )

        if contract is None:
            continue

        clause_id = deterministic_uuid(
            f"clause:{contract_key}:{clause_type}"
        )

        existing = query_by_id(
            session,
            model,
            clause_id,
        )

        if existing:
            clauses[
                f"{contract_key}:{clause_type}"
            ] = existing
            continue

        clause = create_model(
            session,
            model,
            {
                "id": clause_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "contract_id": getattr(
                    contract,
                    "id",
                    None,
                ),
                "clause_type": clause_type,
                "type": clause_type,
                "title": title,
                "text": text,
                "content": text,
                "severity": severity,
                "confidence": 0.96,
                "page_number": 1,
                "metadata": {
                    "seeded": True,
                    "ai_extracted": True,
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        clauses[
            f"{contract_key}:{clause_type}"
        ] = clause

        stats.clauses += 1

    return clauses


# ---------------------------------------------------------------------------
# Obligations
# ---------------------------------------------------------------------------


def seed_obligations(
    session: Any,
    models: dict[str, Any],
    contracts: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "Obligation"
    )

    if model is None:
        logger.warning(
            "Obligation model not found; skipping."
        )
        return

    definitions = (
        (
            "msa",
            "Provide contracted services",
            "service_provider",
            30,
        ),
        (
            "msa",
            "Maintain confidentiality",
            "both_parties",
            90,
        ),
        (
            "nda",
            "Protect confidential information",
            "recipient",
            30,
        ),
        (
            "dpa",
            "Report qualifying security incidents",
            "processor",
            15,
        ),
    )

    for (
        contract_key,
        description,
        responsible_party,
        days,
    ) in definitions:
        contract = contracts.get(
            contract_key
        )

        if contract is None:
            continue

        obligation_id = deterministic_uuid(
            f"obligation:{contract_key}:{description}"
        )

        if query_by_id(
            session,
            model,
            obligation_id,
        ):
            continue

        create_model(
            session,
            model,
            {
                "id": obligation_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "contract_id": getattr(
                    contract,
                    "id",
                    None,
                ),
                "description": description,
                "title": description,
                "responsible_party": responsible_party,
                "status": "open",
                "priority": (
                    "high"
                    if days <= 30
                    else "medium"
                ),
                "due_date": (
                    utc_now().date()
                    + timedelta(days=days)
                ),
                "confidence": 0.94,
                "metadata": {
                    "seeded": True,
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        stats.obligations += 1


# ---------------------------------------------------------------------------
# Risks
# ---------------------------------------------------------------------------


def seed_risks(
    session: Any,
    models: dict[str, Any],
    contracts: dict[str, Any],
    clauses: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "Risk"
    )

    if model is None:
        logger.warning(
            "Risk model not found; skipping."
        )
        return

    definitions = (
        (
            "msa",
            "limitation_of_liability",
            "Liability exposure",
            "high",
            0.88,
        ),
        (
            "msa",
            "termination",
            "Termination ambiguity",
            "medium",
            0.81,
        ),
        (
            "dpa",
            "data_protection",
            "Data protection compliance exposure",
            "high",
            0.91,
        ),
    )

    for (
        contract_key,
        clause_type,
        title,
        severity,
        confidence,
    ) in definitions:
        contract = contracts.get(
            contract_key
        )

        clause = clauses.get(
            f"{contract_key}:{clause_type}"
        )

        if contract is None:
            continue

        risk_id = deterministic_uuid(
            f"risk:{contract_key}:{clause_type}"
        )

        if query_by_id(
            session,
            model,
            risk_id,
        ):
            continue

        create_model(
            session,
            model,
            {
                "id": risk_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "contract_id": getattr(
                    contract,
                    "id",
                    None,
                ),
                "clause_id": getattr(
                    clause,
                    "id",
                    None,
                ),
                "title": title,
                "description": (
                    f"Potential {title.lower()} "
                    "identified during AI analysis."
                ),
                "risk_type": clause_type,
                "severity": severity,
                "level": severity,
                "score": (
                    0.85
                    if severity == "high"
                    else 0.62
                ),
                "confidence": confidence,
                "status": "open",
                "recommendation": (
                    "Review clause with qualified "
                    "legal counsel."
                ),
                "metadata": {
                    "seeded": True,
                    "source": "demo-ai-analysis",
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        stats.risks += 1


# ---------------------------------------------------------------------------
# Renewals
# ---------------------------------------------------------------------------


def seed_renewals(
    session: Any,
    models: dict[str, Any],
    contracts: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "Renewal"
    )

    if model is None:
        logger.warning(
            "Renewal model not found; skipping."
        )
        return

    for key, contract in contracts.items():
        renewal_id = deterministic_uuid(
            f"renewal:{key}"
        )

        if query_by_id(
            session,
            model,
            renewal_id,
        ):
            continue

        expiration_date = (
            getattr(
                contract,
                "expiration_date",
                None,
            )
        )

        if expiration_date is None:
            expiration_date = (
                utc_now().date()
                + timedelta(days=180)
            )

        renewal_date = (
            expiration_date
            - timedelta(days=60)
        )

        create_model(
            session,
            model,
            {
                "id": renewal_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "contract_id": getattr(
                    contract,
                    "id",
                    None,
                ),
                "renewal_date": renewal_date,
                "expiration_date": expiration_date,
                "status": "upcoming",
                "auto_renew": True,
                "notice_period_days": 60,
                "priority": (
                    "high"
                    if (
                        expiration_date
                        - utc_now().date()
                    ).days
                    <= 90
                    else "medium"
                ),
                "metadata": {
                    "seeded": True,
                },
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        stats.renewals += 1


# ---------------------------------------------------------------------------
# Reviews
# ---------------------------------------------------------------------------


def seed_reviews(
    session: Any,
    models: dict[str, Any],
    contracts: dict[str, Any],
    users: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "Review"
    )

    if model is None:
        logger.warning(
            "Review model not found; skipping."
        )
        return

    reviewer = users.get(
        "reviewer@legal-ai.local"
    )

    for key in (
        "msa",
        "dpa",
    ):
        contract = contracts.get(
            key
        )

        if contract is None:
            continue

        review_id = deterministic_uuid(
            f"review:{key}"
        )

        if query_by_id(
            session,
            model,
            review_id,
        ):
            continue

        create_model(
            session,
            model,
            {
                "id": review_id,
                "tenant_id": TENANT_ID,
                "organization_id": ORGANIZATION_ID,
                "contract_id": getattr(
                    contract,
                    "id",
                    None,
                ),
                "reviewer_id": getattr(
                    reviewer,
                    "id",
                    None,
                ),
                "status": "pending",
                "priority": (
                    "high"
                    if key == "dpa"
                    else "medium"
                ),
                "reason": (
                    "AI identified clauses requiring "
                    "human legal review."
                ),
                "summary": (
                    "Review AI-generated risk findings, "
                    "citations, and recommendations."
                ),
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )

        stats.reviews += 1


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


def seed_notifications(
    session: Any,
    models: dict[str, Any],
    users: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "Notification"
    )

    if model is None:
        logger.warning(
            "Notification model not found; skipping."
        )
        return

    admin = users.get(
        "admin@legal-ai.local"
    )

    notification_id = deterministic_uuid(
        "notification:seed:renewal"
    )

    if query_by_id(
        session,
        model,
        notification_id,
    ):
        return

    create_model(
        session,
        model,
        {
            "id": notification_id,
            "tenant_id": TENANT_ID,
            "organization_id": ORGANIZATION_ID,
            "user_id": getattr(
                admin,
                "id",
                None,
            ),
            "type": "renewal",
            "title": (
                "Upcoming contract renewals"
            ),
            "message": (
                "Demo contracts contain renewal "
                "deadlines that require review."
            ),
            "severity": "info",
            "status": "unread",
            "read": False,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        },
    )

    stats.notifications += 1


# ---------------------------------------------------------------------------
# Audit logs
# ---------------------------------------------------------------------------


def seed_audit_logs(
    session: Any,
    models: dict[str, Any],
    users: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "AuditLog"
    )

    if model is None:
        logger.warning(
            "AuditLog model not found; skipping."
        )
        return

    admin = users.get(
        "admin@legal-ai.local"
    )

    audit_id = deterministic_uuid(
        "audit:seed:database"
    )

    if query_by_id(
        session,
        model,
        audit_id,
    ):
        return

    create_model(
        session,
        model,
        {
            "id": audit_id,
            "tenant_id": TENANT_ID,
            "organization_id": ORGANIZATION_ID,
            "user_id": getattr(
                admin,
                "id",
                None,
            ),
            "action": "database_seed",
            "event_type": "system",
            "resource_type": "seed",
            "resource_id": ORGANIZATION_ID,
            "status": "success",
            "details": {
                "environment": "local",
                "seeded": True,
            },
            "ip_address": "127.0.0.1",
            "user_agent": "seed_database.py",
            "created_at": utc_now(),
        },
    )

    stats.audit_logs += 1


# ---------------------------------------------------------------------------
# Model runs
# ---------------------------------------------------------------------------


def seed_model_runs(
    session: Any,
    models: dict[str, Any],
    stats: SeedStats,
) -> None:
    model = models.get(
        "ModelRun"
    )

    if model is None:
        logger.warning(
            "ModelRun model not found; skipping."
        )
        return

    model_run_id = deterministic_uuid(
        "model-run:legal-risk-demo"
    )

    if query_by_id(
        session,
        model,
        model_run_id,
    ):
        return

    create_model(
        session,
        model,
        {
            "id": model_run_id,
            "tenant_id": TENANT_ID,
            "organization_id": ORGANIZATION_ID,
            "model_name": (
                "legal-risk-demo-model"
            ),
            "model_version": "1.0.0",
            "task": "risk_classification",
            "status": "completed",
            "provider": "local",
            "input_tokens": 1200,
            "output_tokens": 420,
            "latency_ms": 842.4,
            "cost": 0.0,
            "metrics": {
                "accuracy": 0.94,
                "f1": 0.92,
                "groundedness": 0.96,
                "citation_recall": 0.91,
            },
            "metadata": {
                "seeded": True,
                "environment": "local",
            },
            "started_at": utc_now()
            - timedelta(
                seconds=3
            ),
            "completed_at": utc_now(),
            "created_at": utc_now(),
        },
    )

    stats.model_runs += 1


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def reset_seed_data(
    session: Any,
    models: dict[str, Any],
) -> None:
    """
    Remove records created by this script.

    Deletion order is child-first to avoid foreign-key
    violations where relationships are enforced.
    """

    deletion_order = (
        "ModelRun",
        "AuditLog",
        "Notification",
        "Review",
        "Renewal",
        "Risk",
        "Obligation",
        "Clause",
        "Contract",
        "DocumentChunk",
        "DocumentVersion",
        "Document",
        "User",
        "Role",
        "Organization",
        "Tenant",
    )

    for model_name in deletion_order:
        model = models.get(
            model_name
        )

        if model is None:
            continue

        try:
            # Prefer tenant-scoped deletion where possible.
            query = session.query(
                model
            )

            columns = model_columns(
                model
            )

            if (
                "tenant_id" in columns
                and model_name != "Tenant"
            ):
                query = query.filter(
                    model.tenant_id
                    == TENANT_ID
                )

            elif model_name == "Tenant":
                query = query.filter(
                    model.id
                    == TENANT_ID
                )

            deleted = query.delete(
                synchronize_session=False
            )

            logger.info(
                "Reset %s: deleted %d rows.",
                model_name,
                deleted,
            )

        except Exception:
            logger.warning(
                "Unable to reset %s.",
                model_name,
                exc_info=True,
            )

    session.commit()


# ---------------------------------------------------------------------------
# Seed orchestration
# ---------------------------------------------------------------------------


def seed_database(
    session: Any,
    models: dict[str, Any],
    minimal: bool = False,
) -> SeedStats:
    stats = SeedStats()

    seed_tenant(
        session,
        models,
        stats,
    )

    seed_organization(
        session,
        models,
        stats,
    )

    roles = seed_roles(
        session,
        models,
        stats,
    )

    users = seed_users(
        session,
        models,
        roles,
        stats,
    )

    documents = seed_documents(
        session,
        models,
        stats,
    )

    versions = seed_document_versions(
        session,
        models,
        documents,
        stats,
    )

    seed_chunks(
        session,
        models,
        versions,
        stats,
    )

    contracts = seed_contracts(
        session,
        models,
        documents,
        stats,
    )

    clauses = seed_clauses(
        session,
        models,
        contracts,
        stats,
    )

    seed_obligations(
        session,
        models,
        contracts,
        stats,
    )

    seed_risks(
        session,
        models,
        contracts,
        clauses,
        stats,
    )

    seed_renewals(
        session,
        models,
        contracts,
        stats,
    )

    if not minimal:
        seed_reviews(
            session,
            models,
            contracts,
            users,
            stats,
        )

        seed_notifications(
            session,
            models,
            users,
            stats,
        )

        seed_audit_logs(
            session,
            models,
            users,
            stats,
        )

        seed_model_runs(
            session,
            models,
            stats,
        )

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seed local development data for the "
            "Enterprise Legal Document Intelligence AI."
        )
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Delete existing seed tenant data "
            "before reseeding."
        ),
    )

    parser.add_argument(
        "--minimal",
        action="store_true",
        help=(
            "Create only core tenant, user, "
            "document, contract and analysis data."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate imports/configuration without "
            "writing database records."
        ),
    )

    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT_ID,
        help="Seed tenant slug.",
    )

    parser.add_argument(
        "--organization-id",
        default=DEFAULT_ORGANIZATION_ID,
        help="Seed organization slug.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    logger.info(
        "Starting database seed."
    )

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "Tenant: %s",
        args.tenant_id,
    )

    logger.info(
        "Organization: %s",
        args.organization_id,
    )

    if args.dry_run:
        logger.info(
            "Dry-run mode enabled."
        )

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        logger.warning(
            "DATABASE_URL is not configured."
        )

    try:
        (
            session_factory,
            _database_module,
            models,
        ) = load_database_components()

    except Exception:
        logger.exception(
            "Unable to load database components."
        )
        return 1

    logger.info(
        "Loaded %d database models.",
        len(models),
    )

    if args.dry_run:
        logger.info(
            "Dry-run completed successfully."
        )
        return 0

    session = None

    try:
        session = session_factory()

        if args.reset:
            logger.warning(
                "Resetting existing seed data."
            )

            reset_seed_data(
                session,
                models,
            )

        stats = seed_database(
            session=session,
            models=models,
            minimal=args.minimal,
        )

        session.commit()

        logger.info(
            "Database seed committed successfully."
        )

    except Exception:
        if session is not None:
            try:
                session.rollback()
            except Exception:
                logger.debug(
                    "Session rollback failed.",
                    exc_info=True,
                )

        logger.exception(
            "Database seed failed."
        )

        return 1

    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                logger.debug(
                    "Session close failed.",
                    exc_info=True,
                )

    print()
    print("=" * 72)
    print("DATABASE SEED COMPLETE")
    print("=" * 72)
    print(
        f"Tenants:             {stats.tenants}"
    )
    print(
        f"Organizations:       {stats.organizations}"
    )
    print(
        f"Roles:               {stats.roles}"
    )
    print(
        f"Users:               {stats.users}"
    )
    print(
        f"Documents:           {stats.documents}"
    )
    print(
        f"Document versions:   {stats.document_versions}"
    )
    print(
        f"Chunks:              {stats.chunks}"
    )
    print(
        f"Contracts:           {stats.contracts}"
    )
    print(
        f"Clauses:             {stats.clauses}"
    )
    print(
        f"Obligations:         {stats.obligations}"
    )
    print(
        f"Risks:               {stats.risks}"
    )
    print(
        f"Renewals:            {stats.renewals}"
    )
    print(
        f"Reviews:             {stats.reviews}"
    )
    print(
        f"Notifications:       {stats.notifications}"
    )
    print(
        f"Audit logs:          {stats.audit_logs}"
    )
    print(
        f"Model runs:          {stats.model_runs}"
    )
    print("=" * 72)
    print()
    print(
        "Local demo users:"
    )
    print(
        f"  Admin:    {DEFAULT_ADMIN_EMAIL}"
    )
    print(
        "  Counsel:  legal@legal-ai.local"
    )
    print(
        "  Reviewer: reviewer@legal-ai.local"
    )
    print(
        "  Analyst:  analyst@legal-ai.local"
    )
    print()
    print(
        "WARNING: Seed passwords are for local/demo "
        "use only. Change them before any shared "
        "or production deployment."
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )