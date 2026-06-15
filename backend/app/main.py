"""
ForgeMind FastAPI Application.

Entry point for the ForgeMind Dataset Intelligence API.
Initializes the application, database, and includes all routers.
"""

import logging
import sys
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.datasets import router as datasets_router
from app.core.config import get_settings
from app.infrastructure.database import init_db

# ============================================
# Structured Logging Setup
# ============================================
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

settings = get_settings()


# ============================================
# Application Lifespan
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info(
        "Starting ForgeMind API",
        env=settings.app_env,
        debug=settings.app_debug,
    )

    # Initialize database tables
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down ForgeMind API")


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="ForgeMind",
    description=(
        "AI-powered platform that transforms user-provided datasets "
        "into domain-specific Small Language Models (SLMs). "
        "Phase 1: Dataset Intelligence Layer."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Global Exception Handler
# ============================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler for unhandled errors."""
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal error occurred. Please try again later.",
        },
    )


# ============================================
# Routers
# ============================================
app.include_router(datasets_router, prefix="/api/v1")

# Phase 2: Augmentation & Training Dataset routers
from app.api.augmentation import router as augmentation_router
from app.api.training_datasets import router as training_datasets_router

app.include_router(augmentation_router, prefix="/api/v1")
app.include_router(training_datasets_router, prefix="/api/v1")


# ============================================
# Health Check
# ============================================
@app.get(
    "/health",
    tags=["system"],
    summary="Health check",
    description="Returns the health status of the API and its dependencies.",
)
async def health_check():
    """System health check endpoint."""
    health = {
        "status": "healthy",
        "version": "1.0.0",
        "services": {},
    }

    # Check PostgreSQL
    try:
        from sqlalchemy import text
        from app.infrastructure.database import engine

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health["services"]["postgres"] = "healthy"
    except Exception as e:
        health["services"]["postgres"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"

    # Check Redis
    try:
        from app.events.redis_publisher import get_event_publisher

        publisher = get_event_publisher()
        if publisher.ping():
            health["services"]["redis"] = "healthy"
        else:
            health["services"]["redis"] = "unhealthy"
            health["status"] = "degraded"
    except Exception as e:
        health["services"]["redis"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"

    # Check MinIO
    try:
        from app.storage.minio_client import get_minio_client

        minio = get_minio_client()
        minio.ensure_bucket()
        health["services"]["minio"] = "healthy"
    except Exception as e:
        health["services"]["minio"] = f"unhealthy: {str(e)}"
        health["status"] = "degraded"

    status_code = (
        status.HTTP_200_OK
        if health["status"] == "healthy"
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(content=health, status_code=status_code)
