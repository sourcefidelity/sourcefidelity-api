"""SourceFidelity API – FastAPI entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, check, status, report, sources

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Self-hosted academic citation checking API",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS – allow Moodle plugin and other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
