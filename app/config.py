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
    # Current rationale (revised Aug 12 after the baseline run):
    #   1. OpenAlex — does the bulk of the work (87 of 121 hits in the baseline).
    #      Fast, broad, now codebase-unified with Unpaywall (so it also covers OA
    #      full text that a separate Unpaywall adapter would have found).
    #   2. Crossref — reliable DOI resolution, metadata + abstract only.
    #   3. CORE — unique OA PDFs from 10K+ repositories. Now fully serialized
    #      (per-key concurrency limit stalls parallel requests) so it's slower
    #      per call; placed after the fast metadata sources. v3 `q=doi:` /
    #      default-field title query (Aug 12 fix).
    #   4. Elsevier — only source for Elsevier full text (PII-based URL).
    #   5. Semantic Scholar — re-added (DOI-only OA): openAccessPdf sometimes
    #      has PDFs OpenAlex's best_oa_location misses. Throttled ~1 req/s.
    #   6. Wikisource — multilingual public-domain texts; only tried for refs
    #      whose year is old enough to be public domain (see source code).
    #   7. Gutenberg — public-domain Western classics; same PD year gate.
    #      Volunteer-run and slow (timeout 45s); last resort for old works.
    #   8. web_search (optional, last) — searches Google/Bing for PDFs of sources
    #      the academic-DB chain couldn't find. Only active when SEARCH_PROVIDER
    #      is configured. Add to the list to enable: "...,gutenberg,web_search"
    RETRIEVAL_SOURCES: str = "openalex,crossref,core,elsevier,semantic_scholar,wikisource,gutenberg"

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

    # Web search provider for PDF fallback retrieval (after academic-DB chain fails).
    # Pluggable: "google" (Custom Search), "bing" (Web Search), or None (disabled).
    # When set, the retrieval chain searches the web for source titles + "filetype:pdf"
    # and downloads/validates any PDFs found (author homepages, repositories, OA copies).
    # Source-access neutrality applies (§3.5): the app verifies against whatever it finds,
    # does NOT access Sci-Hub or pirated copies. Legitimate OA / author-homepage / institutional-repository PDFs only.
    SEARCH_PROVIDER: str | None = None  # "google"|"bing"|"searxng"|"brave"|"duckduckgo"|"tavily"|"exa"|None
    GOOGLE_SEARCH_API_KEY: str | None = None
    GOOGLE_SEARCH_CSE_ID: str | None = None  # Custom Search Engine ID
    BING_SEARCH_API_KEY: str | None = None
    # SearXNG (self-hosted meta-search — recommended for institutions)
    SEARXNG_URL: str | None = None  # e.g., "http://localhost:8080"
    # Brave Search (commercial API, free tier 2000/month)
    BRAVE_SEARCH_API_KEY: str | None = None
    # Tavily (AI-focused search, free tier 1000/month)
    TAVILY_API_KEY: str | None = None
    # Exa (neural/semantic search, free tier — good for academic content)
    EXA_API_KEY: str | None = None

    # DOI Resolver / Campus Proxy (institutional deployment).
    # When set, the app constructs {DOI_RESOLVER_URL}{doi} to access papers
    # through the institution's library proxy/subscription. This dramatically
    # improves full-text retrieval for paywalled content — the proxy handles
    # authentication, the app gets the PDF.
    # Examples:
    #   EZproxy:   "https://proxy.university.edu:2048/login?url=https://doi.org/"
    #   OpenURL:   "https://resolver.university.edu/sfx?rft_id=info:doi/"
    #   Custom:    "https://library.university.edu/fulltext/"
    # The DOI is appended directly: {DOI_RESOLVER_URL}10.1234/foo
    DOI_RESOLVER_URL: str | None = None

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
