"""SourceFidelity API – FastAPI entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.log_safety import configure_sensitive_transport_logging

configure_sensitive_transport_logging()

from app.routers import health, check, status, report, sources

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Self-hosted academic citation checking API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cross-origin API access is an explicit deployment allowlist. Moodle deep
# links are ordinary navigation and do not require permissive CORS.
allowed_origins = [
    value.strip()
    for value in settings.CORS_ALLOWED_ORIGINS.split(",")
    if value.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(check.router, prefix="/check", tags=["check"])
app.include_router(status.router, prefix="/status", tags=["status"])
app.include_router(report.router, prefix="/report", tags=["report"])
app.include_router(sources.router)  # prefix "/sources" set in the router


@app.get("/")
async def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION}
