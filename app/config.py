
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central application configuration.

    Values are loaded from environment variables and .env.
    Environment variables take precedence over .env values.
    """

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    APP_NAME: str = "Enterprise Legal Document Intelligence AI"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    API_PREFIX: str = "/api/v1"

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------

    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "null",  # For file:// protocol
            "*",  # Allow all origins in development
        ]
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, value):
        if isinstance(value, str):
            return [
                origin.strip()
                for origin in value.split(",")
                if origin.strip()
            ]

        return value

    # ------------------------------------------------------------------
    # Database - PostgreSQL
    # ------------------------------------------------------------------

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "legal_ai"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    DATABASE_URL: str | None = None

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """
        SQLAlchemy async PostgreSQL connection string.
        """

        if self.DATABASE_URL:
            database_url = self.DATABASE_URL

            if database_url.startswith("postgresql://"):
                database_url = database_url.replace(
                    "postgresql://",
                    "postgresql+asyncpg://",
                    1,
                )

            return database_url

        return (
            "postgresql+asyncpg://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"
            f"/{self.POSTGRES_DB}"
        )

    @property
    def SYNC_DATABASE_URL(self) -> str:
        """
        SQLAlchemy synchronous PostgreSQL connection string.
        """

        if self.DATABASE_URL:
            database_url = self.DATABASE_URL

            if "+asyncpg" in database_url:
                database_url = database_url.replace(
                    "+asyncpg",
                    "",
                    1,
                )

            return database_url

        return (
            "postgresql://"
            f"{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}"
            f"/{self.POSTGRES_DB}"
        )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    REDIS_URL: str | None = None

    @property
    def REDIS_CONNECTION_URL(self) -> str:
        if self.REDIS_URL:
            return self.REDIS_URL

        if self.REDIS_PASSWORD:
            return (
                f"redis://:{self.REDIS_PASSWORD}"
                f"@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
            )

        return (
            f"redis://{self.REDIS_HOST}:"
            f"{self.REDIS_PORT}/{self.REDIS_DB}"
        )

    # ------------------------------------------------------------------
    # Qdrant Vector Database
    # ------------------------------------------------------------------

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    QDRANT_COLLECTION: str = "legal_documents"

    @property
    def QDRANT_CONNECTION_URL(self) -> str:
        if self.QDRANT_URL:
            return self.QDRANT_URL

        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    # ------------------------------------------------------------------
    # Object Storage
    # ------------------------------------------------------------------

    STORAGE_PROVIDER: str = "local"

    STORAGE_BUCKET: str = "legal-documents"

    STORAGE_ENDPOINT: str | None = None
    STORAGE_ACCESS_KEY: str | None = None
    STORAGE_SECRET_KEY: str | None = None
    STORAGE_REGION: str = "us-east-1"

    LOCAL_STORAGE_PATH: str = "./storage/documents"

    # ------------------------------------------------------------------
    # Hugging Face
    # ------------------------------------------------------------------

    HF_API_TOKEN: str | None = None

    HF_ORGANIZATION: str | None = None

    HF_DOCUMENT_CLASSIFIER: str = (
        "distilbert-base-uncased"
    )

    HF_LEGAL_NER_MODEL: str = (
        "dslim/bert-base-NER"
    )

    HF_CLAUSE_CLASSIFIER: str = (
        "distilbert-base-uncased"
    )

    HF_RISK_CLASSIFIER: str = (
        "distilbert-base-uncased"
    )

    HF_EMBEDDING_MODEL: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    HF_RERANKER_MODEL: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    HF_CACHE_DIR: str = "./.cache/huggingface"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    LLM_PROVIDER: str = "openai"

    LLM_MODEL: str = "gpt-5"

    LLM_API_KEY: str | None = None

    LLM_BASE_URL: str | None = None

    LLM_TEMPERATURE: float = 0.0

    LLM_MAX_TOKENS: int = 4096

    LLM_TIMEOUT: int = 120

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------

    RAG_TOP_K: int = 20

    RAG_FINAL_TOP_K: int = 8

    RAG_RERANK_TOP_K: int = 12

    RAG_CHUNK_SIZE: int = 800

    RAG_CHUNK_OVERLAP: int = 120

    RAG_PARENT_CHUNK_SIZE: int = 2400

    RAG_ENABLE_HYBRID_SEARCH: bool = True

    RAG_ENABLE_RERANKING: bool = True

    RAG_ENABLE_MULTI_QUERY: bool = True

    RAG_ENABLE_PERMISSION_FILTERING: bool = True

    RAG_ENABLE_CITATIONS: bool = True

    RAG_ENABLE_HALLUCINATION_CHECK: bool = True

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    BM25_K1: float = 1.5
    BM25_B: float = 0.75

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    EMBEDDING_DIMENSION: int = 384

    EMBEDDING_BATCH_SIZE: int = 32

    EMBEDDING_CACHE_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Document Processing
    # ------------------------------------------------------------------

    MAX_UPLOAD_SIZE_MB: int = 50

    ALLOWED_DOCUMENT_EXTENSIONS: List[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".docx",
            ".doc",
            ".txt",
            ".png",
            ".jpg",
            ".jpeg",
        ]
    )

    OCR_ENABLED: bool = True

    OCR_LANGUAGE: str = "eng"

    EXTRACT_TABLES: bool = True

    EXTRACT_METADATA: bool = True

    EXTRACT_PAGE_NUMBERS: bool = True

    # ------------------------------------------------------------------
    # JWT / Authentication
    # ------------------------------------------------------------------

    JWT_SECRET_KEY: str = (
        "CHANGE_THIS_IN_PRODUCTION"
    )

    JWT_ALGORITHM: str = "HS256"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ------------------------------------------------------------------
    # Password Security
    # ------------------------------------------------------------------

    PASSWORD_MIN_LENGTH: int = 12

    PASSWORD_HASH_ALGORITHM: str = "bcrypt"

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    ENABLE_SECURITY_HEADERS: bool = True

    ENABLE_RATE_LIMITING: bool = True

    RATE_LIMIT_REQUESTS: int = 100

    RATE_LIMIT_WINDOW_SECONDS: int = 60

    ENABLE_PROMPT_INJECTION_PROTECTION: bool = True

    ENABLE_PII_DETECTION: bool = True

    ENABLE_OUTPUT_GUARDRAILS: bool = True

    ENABLE_CONTENT_FILTER: bool = True

    ENABLE_TENANT_ISOLATION: bool = True

    # ------------------------------------------------------------------
    # Legal AI Safety
    # ------------------------------------------------------------------

    LEGAL_DISCLAIMER_ENABLED: bool = True

    REQUIRE_HUMAN_REVIEW_FOR_HIGH_RISK: bool = True

    HIGH_RISK_THRESHOLD: float = 0.80

    REQUIRE_CITATIONS: bool = True

    BLOCK_UNGROUNDED_ANSWERS: bool = True

    MAX_CITATIONS_PER_RESPONSE: int = 10

    # ------------------------------------------------------------------
    # Multi-Tenant Architecture
    # ------------------------------------------------------------------

    MULTI_TENANT_ENABLED: bool = True

    DEFAULT_ORGANIZATION_ROLE: str = "member"

    # ------------------------------------------------------------------
    # Celery / Background Tasks
    # ------------------------------------------------------------------

    CELERY_BROKER_URL: str | None = None

    CELERY_RESULT_BACKEND: str | None = None

    @property
    def CELERY_BROKER_CONNECTION(self) -> str:
        return (
            self.CELERY_BROKER_URL
            or self.REDIS_CONNECTION_URL
        )

    @property
    def CELERY_RESULT_CONNECTION(self) -> str:
        return (
            self.CELERY_RESULT_BACKEND
            or self.REDIS_CONNECTION_URL
        )

    # ------------------------------------------------------------------
    # Email / Notifications
    # ------------------------------------------------------------------

    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587

    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None

    SMTP_FROM_EMAIL: str = "noreply@example.com"

    SMTP_USE_TLS: bool = True

    NOTIFICATIONS_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Expiry / Renewal Management
    # ------------------------------------------------------------------

    EXPIRY_WARNING_DAYS: List[int] = Field(
        default_factory=lambda: [
            90,
            60,
            30,
            14,
            7,
            1,
        ]
    )

    AUTO_RENEWAL_ALERT_ENABLED: bool = True

    MISSED_DEADLINE_ALERT_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    AUDIT_LOG_ENABLED: bool = True

    AUDIT_RETENTION_DAYS: int = 3650

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    LOG_LEVEL: str = "INFO"

    LOG_FORMAT: str = "json"

    ENABLE_TELEMETRY: bool = True

    OTEL_ENABLED: bool = False

    OTEL_EXPORTER_ENDPOINT: str | None = None

    SERVICE_NAME: str = "enterprise-legal-document-intelligence-ai"

    # ------------------------------------------------------------------
    # Prometheus
    # ------------------------------------------------------------------

    PROMETHEUS_ENABLED: bool = True

    PROMETHEUS_PORT: int = 9090

    # ------------------------------------------------------------------
    # Model Evaluation
    # ------------------------------------------------------------------

    EVALUATION_ENABLED: bool = True

    EVALUATION_DATASET_PATH: str = (
        "./ml/datasets/evaluation"
    )

    RETRIEVAL_RECALL_K: List[int] = Field(
        default_factory=lambda: [1, 3, 5, 10]
    )

    # ------------------------------------------------------------------
    # Model Serving
    # ------------------------------------------------------------------

    MODEL_DEVICE: str = "cpu"

    MODEL_BATCH_SIZE: int = 8

    MODEL_MAX_CONCURRENCY: int = 4

    MODEL_TIMEOUT_SECONDS: int = 120

    # ------------------------------------------------------------------
    # Frontend
    # ------------------------------------------------------------------

    FRONTEND_URL: str = "http://localhost:5173"

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached application settings instance.
    """

    return Settings()


settings = get_settings()