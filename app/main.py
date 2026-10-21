from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.core.logging import configure_logging
from backend.core.telemetry import setup_telemetry
from backend.api.routes import (
    auth,
    users,
    organizations,
    documents,
    search,
    chat,
    risk,
    expiry,
    renewals,
    reviews,
    notifications,
    audit,
    models,
    health,
)


configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application startup/shutdown lifecycle.

    Startup:
        - Initialize telemetry
        - Prepare application resources
        - Load required model/configuration state

    Shutdown:
        - Release application resources
        - Flush telemetry/logging
    """

    setup_telemetry(app)

    # Add future startup initialization here:
    # await initialize_database()
    # await initialize_vector_store()
    # await initialize_model_registry()
    # await initialize_cache()

    yield

    # Add future shutdown cleanup here:
    # await close_database()
    # await close_vector_store()
    # await close_cache()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Enterprise Legal Document Intelligence & Contract Lifecycle AI. "
        "Provides document ingestion, legal RAG, clause extraction, "
        "risk detection, expiry intelligence, contract analysis, "
        "agentic workflows, and human-in-the-loop review."
    ),
    docs_url="/docs" if settings.DEBUG else "/docs",
    redoc_url="/redoc" if settings.DEBUG else "/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=False,  # Must be False when allow_origins is ["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Prevent internal exception details from being exposed to clients.
    """

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "request_id": getattr(
                request.state,
                "request_id",
                None,
            ),
        },
    )


# ---------------------------------------------------------------------------
# API Router
# ---------------------------------------------------------------------------

API_PREFIX = "/api/v1"

app.include_router(
    auth.router,
    prefix=API_PREFIX,
    tags=["Authentication"],
)

app.include_router(
    users.router,
    prefix=API_PREFIX,
    tags=["Users"],
)

app.include_router(
    organizations.router,
    prefix=API_PREFIX,
    tags=["Organizations"],
)

app.include_router(
    documents.router,
    prefix=API_PREFIX,
    tags=["Documents"],
)

# Documents router enabled - contract, clause, and analytics routes remain disabled
# due to import issues - will be re-enabled after fixing schema dependencies

app.include_router(
    search.router,
    prefix=API_PREFIX,
    tags=["Search"],
)

app.include_router(
    chat.router,
    prefix=API_PREFIX,
    tags=["Legal AI Chat"],
)

app.include_router(
    risk.router,
    prefix=API_PREFIX,
    tags=["Risk Intelligence"],
)

app.include_router(
    expiry.router,
    prefix=API_PREFIX,
    tags=["Expiry Management"],
)

app.include_router(
    renewals.router,
    prefix=API_PREFIX,
    tags=["Renewals"],
)

app.include_router(
    reviews.router,
    prefix=API_PREFIX,
    tags=["Human Review"],
)

app.include_router(
    notifications.router,
    prefix=API_PREFIX,
    tags=["Notifications"],
)

app.include_router(
    audit.router,
    prefix=API_PREFIX,
    tags=["Audit"],
)

app.include_router(
    models.router,
    prefix=API_PREFIX,
    tags=["AI Models"],
)

app.include_router(
    health.router,
    prefix=API_PREFIX,
    tags=["Health"],
)


# ---------------------------------------------------------------------------
# Root endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
    summary="API information",
)
async def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "environment": settings.ENVIRONMENT,
        "api": API_PREFIX,
        "docs": "/docs",
    }


@app.get(
    "/health",
    tags=["System"],
    summary="Basic application health check",
)
async def health_check() -> dict:
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )