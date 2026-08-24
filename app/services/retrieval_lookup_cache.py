"""Bounded Redis cache for canonical abstract and lookup freshness state."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import re

import redis

from app.config import settings
from app.services.retrieval.base import RetrievalResult
from app.services.source_type import normalize_source_kind

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_doi(value: str) -> str:
    normalized = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
    return normalized


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _url_hash(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LookupCacheRecord:
    outcome: str
    stored_at: datetime
    refresh_after: datetime
    policy_signature: str
    student_url_hash: str | None
    result: RetrievalResult | None = None

    def is_fresh(
        self,
        *,
        policy_signature: str,
        student_url: str | None,
        now: datetime | None = None,
    ) -> bool:
        current_url_hash = _url_hash(student_url)
        return bool(
            self.policy_signature == policy_signature
            and (not current_url_hash or current_url_hash == self.student_url_hash)
            and (now or _utcnow()) < self.refresh_after
        )


class RetrievalLookupCache:
    """Store reusable abstract evidence without treating it as full text.

    Keys contain only hashes. Values contain a bounded, validated subset of
    canonical bibliographic fields; raw provider responses and source text are
    deliberately excluded.
    """

    def __init__(self, client=None) -> None:
        self._client = client
        self._available = settings.RETRIEVAL_LOOKUP_CACHE_ENABLED
        self._warned = False

    def _redis(self):
        if not self._available:
            return None
        if self._client is None:
            self._client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=0.25,
                socket_timeout=0.25,
            )
        return self._client

    def _disable(self, exc: Exception) -> None:
        self._available = False
        if not self._warned:
            logger.warning(
                "Retrieval lookup cache unavailable; continuing without it (%s)",
                type(exc).__name__,
            )
            self._warned = True

    @staticmethod
    def _key(
        *,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        source_kind: str | None = None,
    ) -> str | None:
        if doi and _normalize_doi(doi):
            identity = f"doi:{_normalize_doi(doi)}"
        elif title and _normalize_text(title):
            identity = "|".join(
                (
                    "title",
                    _normalize_text(title),
                    _normalize_text(author),
                    year or "",
                    normalize_source_kind(source_kind),
                )
            )
        else:
            return None
        scope_hash = hashlib.sha256(
            settings.SOURCE_REPOSITORY_SCOPE_ID.encode("utf-8")
        ).hexdigest()[:16]
        identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"sourcefidelity:lookup:v{_SCHEMA_VERSION}:{scope_hash}:{identity_hash}"

    def get(
        self,
        *,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        source_kind: str | None = None,
    ) -> LookupCacheRecord | None:
        key = self._key(
            doi=doi,
            title=title,
            author=author,
            year=year,
            source_kind=source_kind,
        )
        client = self._redis()
        if key is None or client is None:
            return None
        try:
            raw = client.get(key)
            if not raw:
                return None
            payload = json.loads(raw)
            if payload.get("schema_version") != _SCHEMA_VERSION:
                return None
            result = None
            if payload.get("outcome") == "abstract" and payload.get("abstract"):
                result = RetrievalResult(
                    source_name=payload.get("source_name") or "lookup_cache",
                    success=True,
                    abstract=payload["abstract"],
                    doi=payload.get("doi"),
                    title=payload.get("title"),
                    year=payload.get("year"),
                    authors=list(payload.get("authors") or []),
                    metadata={
                        "from_lookup_cache": True,
                        "cached_at": payload["stored_at"],
                    },
                )
            return LookupCacheRecord(
                outcome=payload["outcome"],
                stored_at=datetime.fromisoformat(payload["stored_at"]),
                refresh_after=datetime.fromisoformat(payload["refresh_after"]),
                policy_signature=payload["policy_signature"],
                student_url_hash=payload.get("student_url_hash"),
                result=result,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        except redis.RedisError as exc:
            self._disable(exc)
            return None

    def put_abstract(
        self,
        result: RetrievalResult,
        *,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        student_url: str | None,
        policy_signature: str,
        source_kind: str | None = None,
    ) -> None:
        if not result.abstract:
            return
        self._put(
            outcome="abstract",
            result=result,
            doi=doi,
            title=title,
            author=author,
            year=year,
            student_url=student_url,
            policy_signature=policy_signature,
            source_kind=source_kind,
            refresh_after=_utcnow()
            + timedelta(days=settings.RETRIEVAL_ABSTRACT_REFRESH_DAYS),
        )

    def put_miss(
        self,
        *,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        student_url: str | None,
        policy_signature: str,
        source_kind: str | None = None,
    ) -> None:
        self._put(
            outcome="miss",
            result=None,
            doi=doi,
            title=title,
            author=author,
            year=year,
            student_url=student_url,
            policy_signature=policy_signature,
            source_kind=source_kind,
            refresh_after=_utcnow()
            + timedelta(hours=settings.RETRIEVAL_NEGATIVE_REFRESH_HOURS),
        )

    def _put(
        self,
        *,
        outcome: str,
        result: RetrievalResult | None,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        student_url: str | None,
        policy_signature: str,
        refresh_after: datetime,
        source_kind: str | None = None,
    ) -> None:
        key = self._key(
            doi=doi,
            title=title,
            author=author,
            year=year,
            source_kind=source_kind,
        )
        client = self._redis()
        if key is None or client is None:
            return
        now = _utcnow()
        abstract = (result.abstract if result else None) or None
        if abstract:
            abstract = re.sub(r"\x00", "", abstract)[: settings.MAX_TEXT_LENGTH_CHARS]
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "outcome": outcome,
            "stored_at": now.isoformat(),
            "refresh_after": refresh_after.isoformat(),
            "policy_signature": policy_signature,
            "student_url_hash": _url_hash(student_url),
            "source_name": result.source_name if result else None,
            "abstract": abstract,
            "doi": (result.doi if result else None) or doi,
            "title": (result.title if result else None) or title,
            "year": (result.year if result else None) or year,
            "authors": list(result.authors if result else []),
            "source_kind": normalize_source_kind(source_kind),
        }
        ttl = max(1, settings.RETRIEVAL_LOOKUP_CACHE_RETENTION_DAYS * 86400)
        try:
            client.setex(key, ttl, json.dumps(payload, ensure_ascii=False))
        except redis.RedisError as exc:
            self._disable(exc)

    def delete(
        self,
        *,
        doi: str | None,
        title: str | None,
        author: str | None,
        year: str | None,
        source_kind: str | None = None,
    ) -> None:
        key = self._key(
            doi=doi,
            title=title,
            author=author,
            year=year,
            source_kind=source_kind,
        )
        client = self._redis()
        if key is None or client is None:
            return
        try:
            client.delete(key)
        except redis.RedisError as exc:
            self._disable(exc)
