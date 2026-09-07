"""Liveness and dependency readiness endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from redis import Redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.services.file_safety import check_clamd_health
from app.services.storage.backend import StorageBackend, get_storage_backend

router = APIRouter()


@router.get("/health")
def health_check():
    """Process liveness only; it deliberately does not claim dependencies work."""
    return {"status": "ok"}


@router.get("/health/ready")
def readiness_check(
    session: Session = Depends(get_db),
    backend: StorageBackend = Depends(get_storage_backend),
):
    """Fail closed unless the dependencies needed by an ordinary job respond."""
    client = None
    try:
        session.execute(text("SELECT 1")).scalar_one()
        backend.list_keys("__sourcefidelity_readiness_probe_no_match__/")
        client = Redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        if not client.ping():
            raise RuntimeError("redis_unavailable")
        if settings.MALWARE_SCAN_REQUIRED:
            check_clamd_health()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Service dependencies are unavailable") from exc
    finally:
        if client is not None:
            client.close()
    return {"status": "ready"}
