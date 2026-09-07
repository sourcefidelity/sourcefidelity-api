"""Durable, text-free upload intents in the existing object store.

An intent precedes source bytes and outlives a database rollback or process
exit. Only this reserved namespace is enumerated; historical files are never
inferred to be orphans merely because no database record was found.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import uuid

from sqlalchemy import select, text

from app.models.source_repository import ContentObjectRecord
from app.services.upload_completion import upload_confirmed, check_due, next_check


INTENT_PREFIX = "source-upload-intents/v1/"
INTENT_LEASE_SECONDS = 3600
CURSOR_KEY = "source-upload-recovery/v1/cursor.json"
_PENDING = "source_upload_intents_v1"


def persist_upload_intent(session, backend, *, storage_key, license_class, digest, kind,
                          generation=None, retirement=False):
    identifier = uuid.uuid4().hex
    key = f"{INTENT_PREFIX}{identifier}.json"
    now = datetime.now(timezone.utc)
    payload = {
        "version": 1, "id": identifier, "storage_key": storage_key,
        "license_class": license_class, "content_sha256": digest,
        "kind": kind.value, "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=INTENT_LEASE_SECONDS)).isoformat(),
        "upload_confirmed": False,
    }
    if generation is not None or retirement:
        payload.update(version=2, generation=generation,
                       purpose="retirement" if retirement else "upload")
    # Remember before PUT: an acknowledgement failure may still leave a journal.
    session.info.setdefault(_PENDING, {})[key] = backend
    backend.upload(json.dumps(payload, sort_keys=True).encode(), key)
    return key


def persist_retirement_intent(session, backend, *, storage_key, license_class, digest, kind):
    """Record an old deletion target before rotating the database locator.

    This intent initiates no upload to the old target. Earlier uncertain PUTs
    retain their own upload watches; retirement never settles those watches.
    """
    from app.services.source_repository import _object_key
    legacy = _object_key(license_class, digest, kind)
    generation = None if storage_key == legacy else storage_key.rsplit("/", 1)[-1].split(".")[0]
    expected = _object_key(license_class, digest, kind, generation)
    if storage_key != expected or (generation is not None and not re.fullmatch(r"[0-9a-f]{32}", generation)):
        raise ValueError("Invalid retirement target")
    return persist_upload_intent(session, backend, storage_key=storage_key,
        license_class=license_class, digest=digest, kind=kind,
        generation=generation, retirement=True)


def confirm_upload_intent(backend, key, receipt):
    """Persist acknowledgement, never infer completion from current existence."""
    if not upload_confirmed(receipt):
        return
    payload, _expires = _load_intent(backend, key)
    payload["upload_confirmed"] = True
    # A lost acknowledgement here can only leave conservative pending state
    # or a valid completion record for an already confirmed content PUT.
    backend.upload(json.dumps(payload, sort_keys=True).encode(), key)


def _load_intent(backend, key):
    from app.services.retrieval.base import RepresentationKind
    from app.services.source_repository import LICENSE_CLASSES, _object_key
    if not re.fullmatch(re.escape(INTENT_PREFIX) + r"[0-9a-f]{32}\.json", key):
        raise ValueError("Invalid upload intent key")
    content = backend.download(key)
    if len(content) > 4096:
        raise ValueError("Oversized upload intent")
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("version") not in {1, 2}:
        raise ValueError("Invalid upload intent version")
    if key != f"{INTENT_PREFIX}{payload.get('id')}.json":
        raise ValueError("Upload intent identity mismatch")
    license_class = payload.get("license_class")
    digest = payload.get("content_sha256")
    if license_class not in LICENSE_CLASSES or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid upload intent binding")
    generation = None
    if payload["version"] == 2:
        generation = payload.get("generation")
        if payload.get("purpose") not in {"upload", "retirement"}:
            raise ValueError("Invalid upload intent purpose")
        if generation is None:
            if payload["purpose"] != "retirement":
                raise ValueError("Missing upload generation")
        elif not isinstance(generation, str) or not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("Invalid upload generation")
    elif "purpose" in payload or "generation" in payload:
        raise ValueError("Invalid legacy intent fields")
    expected = _object_key(license_class, digest, RepresentationKind(payload["kind"]), generation)
    if payload.get("storage_key") != expected:
        raise ValueError("Upload intent target mismatch")
    created = datetime.fromisoformat(payload["created_at"])
    expires = datetime.fromisoformat(payload["expires_at"])
    if created.tzinfo is None or expires.tzinfo is None or (expires - created).total_seconds() != INTENT_LEASE_SECONDS:
        raise ValueError("Invalid upload intent lease")
    return payload, expires


def _try_content_lock(session, payload):
    if session.get_bind().dialect.name != "postgresql":
        return True  # SQLite unit tests do not establish concurrency acceptance.
    seed = f"content-object:{payload['license_class']}:{payload['content_sha256']}"
    lock_id = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big", signed=True)
    return bool(session.scalar(text("SELECT pg_try_advisory_xact_lock(:id)"), {"id": lock_id}))


def _cleanup_one(session, backend, key, *, now, immediate=False):
    try:
        payload, expires = _load_intent(backend, key)
    except FileNotFoundError:
        return "absent"
    except (ValueError, TypeError, KeyError):
        return "invalid"
    if not immediate and expires > now and payload.get("cleanup_status") != "upload_completion_unresolved":
        return "pending"
    if not immediate and not check_due(payload.get("cleanup_next_check_at"), now):
        return "pending"
    if not _try_content_lock(session, payload):
        return "busy"
    # Refresh after acquiring ownership: admission may have confirmed the PUT.
    payload, _expires = _load_intent(backend, key)
    if not immediate and not check_due(payload.get("cleanup_next_check_at"), now):
        return "pending"
    referenced = session.scalar(select(ContentObjectRecord.id).where(
        ContentObjectRecord.storage_key == payload["storage_key"]
    ).limit(1))
    if referenced is None:
        if not backend.delete(payload["storage_key"]) or backend.exists(payload["storage_key"]):
            return "pending"
    if payload.get("upload_confirmed") is not True and not (
        payload["version"] == 2 and payload["purpose"] == "retirement"
    ):
        # Even a new durable owner cannot settle this older uncertain PUT:
        # it could finish after that owner's last reference is later deleted.
        payload["cleanup_status"] = "upload_completion_unresolved"
        payload["cleanup_next_check_at"] = next_check(datetime.fromisoformat(payload["created_at"]), now)
        backend.upload(json.dumps(payload, sort_keys=True).encode(), key)
        return "pending"
    # Completion plus confirmed absence (or a durable owner) permits retirement.
    if not backend.delete(key) or backend.exists(key):
        return "pending"
    return "protected" if referenced is not None else "removed"


def settle_session_upload_intents(session):
    """Best-effort immediate cleanup after commit/rollback; never mask its result."""
    pending = session.info.pop(_PENDING, {})
    removed = 0
    for key, backend in pending.items():
        try:
            outcome = _cleanup_one(session, backend, key, now=datetime.now(timezone.utc), immediate=True)
            session.commit()
            removed += outcome == "removed"
        except Exception:
            session.rollback()
            # The persisted intent, not session memory, owns the next retry.
    return removed


def cleanup_source_upload_intents(session, backend, *, now=None, batch_size=100):
    """Retry one bounded batch of recorded intents under admission locks."""
    current = now or datetime.now(timezone.utc)
    counts = {name: 0 for name in ("inspected", "removed", "protected", "pending", "busy", "invalid", "absent", "unavailable")}
    limit = max(1, min(batch_size, 1000))
    start_after = ""
    try:
        encoded_cursor = backend.download(CURSOR_KEY)
        cursor = json.loads(encoded_cursor) if len(encoded_cursor) <= 512 else {}
        candidate = cursor.get("start_after", "") if isinstance(cursor, dict) else ""
        if isinstance(candidate, str) and candidate.startswith(INTENT_PREFIX) and len(candidate) <= 200:
            start_after = candidate
    except FileNotFoundError:
        pass
    except (ValueError, TypeError):
        pass  # A cursor controls enumeration only, never deletion authority.
    def page(after):
        method = getattr(backend, "list_keys_page", None)
        if method is not None:
            return method(INTENT_PREFIX, start_after=after, limit=limit)
        return [key for key in sorted(backend.list_keys(INTENT_PREFIX)) if key > after][:limit]
    keys = page(start_after)
    if not keys and start_after:
        keys = page("")
    for key in keys:
        counts["inspected"] += 1
        try:
            outcome = _cleanup_one(session, backend, key, now=current)
            session.commit()
        except Exception:
            session.rollback()
            outcome = "unavailable"
        counts[outcome] += 1
    # Advance past busy/invalid/failed intents too, so one blocked batch cannot
    # hide later cleanup work. Concurrent passes may repeat work safely.
    if len(keys) == limit:
        backend.upload(json.dumps({"start_after": keys[-1]}).encode(), CURSOR_KEY)
    elif start_after:
        backend.delete(CURSOR_KEY)
    return counts
