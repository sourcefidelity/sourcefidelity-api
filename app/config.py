"""Application configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "SourceFidelity API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql://sourcefidelity:sourcefidelity@localhost:5432/sourcefidelity"

    # Redis (Celery broker)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # S3 / MinIO
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: str = "sourcefidelity"
    S3_SECRET_KEY: str = "sourcefidelity123"
    S3_BUCKET: str = "sourcefidelity-texts"
    S3_REGION: str = "us-east-1"

    # LLM (OpenAI-compatible)
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL: str = "deepseek-v4-flash"  # Default: DeepSeek V4 Flash (cheapest, JSON mode supported)
    LLM_BASE_URL: Optional[str] = None  # e.g., https://api.deepseek.com/v1
    LLM_BATCH_SIZE: int = 10  # References per LLM call
    LLM_MAX_RETRIES: int = 2  # Retries for JSON parse failures
    LLM_TEMPERATURE: float = 0.0  # Deterministic output
    LLM_MAX_TOKENS: int = 8192  # Max tokens per response (DeepSeek max output)

    # LLM Provider options (configure via LLM_BASE_URL + LLM_MODEL)
    # DeepSeek: LLM_BASE_URL="https://api.deepseek.com/v1", LLM_MODEL="deepseek-chat"
    # OpenAI: LLM_BASE_URL=None (default), LLM_MODEL="gpt-4o-mini"
    # Ollama: LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL="llama3.1"

    # Cache
    CACHE_ENABLED: bool = True  # Enable DOI/title-hash caching

    # OpenAlex
    OPENALEX_EMAIL: Optional[str] = None  # Recommended for polite pool
    OPENALEX_API_KEY: Optional[str] = None

    # File processing limits
    MAX_FILE_SIZE_MB: int = 50
    MAX_TEXT_LENGTH_CHARS: int = 100_000  # Truncation guard

    # ── Source Repository (Phase 3.5) ────────────────────────
    SOURCE_REPOSITORY_ENABLED: bool = False

    # Storage Backend: "s3" | "seafile"
    STORAGE_BACKEND: str = "s3"

    # Campus access content retention (days). 0 = keep indefinitely.
    CAMPUS_ACCESS_TTL_DAYS: int = 90

    # Retrieval Sources (comma-separated priority list).
    # Order matters: sources are tried left-to-right, stopping at the first
    # that returns full text (or abstract if no PDF found). Reorder based on
    # your institution's access and the empirical hit-rate data from test runs.
    #
    # Current rationale:
    #   1. Elsevier — largest publisher; only source for Elsevier full text (PII issue)
    #   2. OpenAlex — fast, broad coverage, provides OA flag + abstract; resolves ~60% of articles
    #   3. CORE — unique OA PDFs from 10K+ repositories not in OpenAlex
    #   4. Semantic Scholar — good OA coverage but throttled at 1 req/sec
    #   5. Crossref — metadata + abstract only (no full text); reliable DOI resolution
    #   6. Gutenberg — public-domain Western classics (~70K works)
    #   7. Wikisource — multilingual public-domain texts (83 languages)
    RETRIEVAL_SOURCES: str = "elsevier,openalex,core,semantic_scholar,crossref,gutenberg,wikisource"

    # CORE API
    CORE_API_KEY: str | None = None

    # Semantic Scholar API
    S2_API_KEY: str | None = None

    # Elsevier API (Article Retrieval — OA full text + metadata/abstract for paywalled)
    # API key alone: OA articles + metadata/abstract for all.
    # Insttoken (optional, via institutional email to apisupport@elsevier.com):
    #   unlocks paywalled full text if your institution subscribes.
    ELSEVIER_API_KEY: str | None = None
    ELSEVIER_INST_TOKEN: str | None = None

    # Crossref polite email (uses OPENALEX_EMAIL as fallback)
    CROSSREF_EMAIL: str | None = None

    # Student URL download limits (R9)
    STUDENT_URL_MAX_SIZE_MB: int = 50
    STUDENT_URL_TIMEOUT_SECONDS: int = 30

    # ── Completeness / Strictness (Phase 3.5 hardening) ──────
    # Strictness for instructor uploads:
    #   lenient  = accept flagged items with a WARNING (default; individuals)
    #   standard = accept but hold for review (institutions)
    #   strict   = reject flagged items at upload (HTTP 422)
    STRICTNESS_MODE: str = "lenient"

    # Completeness detection toggle
    COMPLETENESS_CHECK_ENABLED: bool = True

    # Flag incompleteness if logical pages < this fraction of expected.
    COMPLETENESS_MIN_PAGE_RATIO: float = 0.70

    # Google Books API key (optional; for page-count lookup). Free key.
    GOOGLE_BOOKS_API_KEY: str | None = None

    # Whether to persist paywalled (campus-access) PDFs to S3.
    # When True: paywalled PDFs downloaded via campus IP are cached to S3
    #   for reuse across checks (efficient, but creates a persistent copy
    #   of copyrighted content on the institution's server).
    # When False (default): paywalled PDFs are verified in-memory and
    #   discarded immediately (like website content). No copyrighted
    #   paywalled content persists in S3. Re-checking the same article
    #   requires re-downloading.
    # Institutions should set this based on their interpretation of publisher
    # terms and their own copyright policy.
    CACHE_PAYWALLED_PDFS: bool = False

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
