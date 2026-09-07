"""Application configuration using pydantic-settings."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings
from typing import Literal, Optional


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "SourceFidelity API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    # Report delivery fails closed unless a deployment authentication adapter
    # is explicitly enabled. The personal bearer is a prototype/operator
    # credential; Institutional deployments replace it with their identity/LMS
    # adapter while producing the same scoped principal contract.
    REPORT_AUTH_MODE: Literal[
        "disabled", "personal_local", "personal_bearer", "institutional_adapter"
    ] = "disabled"
    REPORT_PERSONAL_ACCESS_TOKEN: SecretStr | None = None
    REPORT_SESSION_TTL_SECONDS: int = 900
    REPORT_SESSION_COOKIE_SECURE: bool = True
    API_BIND_HOST: Literal["127.0.0.1", "::1", "localhost"] = "127.0.0.1"
    # Comma-separated exact origins. Empty disables cross-origin browser API
    # access; Moodle deep links do not require CORS.
    CORS_ALLOWED_ORIGINS: str = ""
    CORS_ALLOW_CREDENTIALS: bool = False

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
    # Optional second-provider credential for bounded, development-only model
    # comparisons. Runtime LLM routing continues to use LLM_* above.
    OPENAI_API_KEY: Optional[str] = None
    # Development-only OpenRouter supplemental opinion. This never enters the
    # three-arm formal denominator or runtime routing. Calls remain fail-closed
    # unless the local gate is deliberately enabled.
    OPENROUTER_ENABLED: bool = False
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "z-ai/glm-5.3-flash"
    OPENROUTER_DATA_COLLECTION: Literal["deny"] = "deny"
    OPENROUTER_ZDR: bool = True
    OPENROUTER_REQUIRE_PARAMETERS: bool = True
    # Match the time-bounded launch discount. Requests fail rather than
    # silently paying the doubled standard rates after the promotion ends.
    OPENROUTER_MAX_PROMPT_USD_PER_MILLION: float = 0.075
    OPENROUTER_MAX_COMPLETION_USD_PER_MILLION: float = 0.25

    # Local specialist relationship signal (Phase 3.8). The runtime is an
    # optional install because PyTorch/model weights materially enlarge the
    # base API image. Deployments prefetch a pinned model, then enable it.
    RELATIONSHIP_SIGNAL_BACKEND: Literal["disabled", "deberta_nli"] = "disabled"
    RELATIONSHIP_MODEL_NAME: str = (
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )
    RELATIONSHIP_MODEL_REVISION: str = (
        "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
    )
    RELATIONSHIP_MODEL_LOCAL_FILES_ONLY: bool = True
    RELATIONSHIP_MODEL_DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    RELATIONSHIP_MODEL_BATCH_SIZE: int = 4

    # Optional local semantic rescue for citations whose bounded relevance gate
    # rejected every protected lexical/locator candidate. The same pinned NLI
    # runtime is used only to rank a BM25-prefiltered source subset; it does not
    # make a relationship decision and never replaces the protected union.
    EVIDENCE_RETRIEVAL_SEMANTIC_BACKEND: Literal[
        "disabled", "deberta_nli"
    ] = "disabled"
    EVIDENCE_RETRIEVAL_SEMANTIC_PREFILTER: int = 32
    EVIDENCE_RETRIEVAL_SEMANTIC_MAX_ADDITIONS: int = 4
    EVIDENCE_RETRIEVAL_SEMANTIC_BATCH_SIZE: int = 8

    # Bounded structured judgment is shadow-only until a fixed-corpus
    # calibration justifies allowing its validated outputs to change verdicts.
    VERIFICATION_JUDGMENT_MODE: Literal["shadow", "adjudicate"] = "shadow"
    VERIFICATION_JUDGMENT_MAX_PASSAGES: int = 3
    VERIFICATION_JUDGMENT_MAX_INPUT_TOKENS: int = 4_000
    VERIFICATION_JUDGMENT_MAX_OUTPUT_TOKENS: int = 1_600
    # A committed lease lets the cleanup worker recover transient source,
    # extracted-text, chunk, and embedding objects after a hard worker exit.
    VERIFICATION_RUN_LEASE_SECONDS: int = 900
    VERIFICATION_RUN_CLEANUP_INTERVAL_SECONDS: int = 300
    VERIFICATION_RUN_MAX_TRANSIENT_MB: int = 200
    # A paper can discover many sources serially. Keep each already-admitted
    # transient source alive for the bounded paper workflow without weakening
    # the shorter default lease used by individual verification operations.
    PAPER_SOURCE_RETRIEVAL_LEASE_SECONDS: int = 86_400
    # Student-paper uploads are temporary workflow inputs, not source-repository
    # objects. The locator is committed before upload and stale cleanup removes
    # an object after this recovery window even if a worker exits hard.
    PAPER_UPLOAD_LEASE_SECONDS: int = 86_400
    PAPER_UPLOAD_CLEANUP_INTERVAL_SECONDS: int = 300
    # A sanitized, separately hashed marking copy is retained with the report,
    # never with the comparison corpus. Formative copies expire independently.
    REPORT_PAPER_COPY_ENABLED: bool = True
    FORMATIVE_REPORT_PAPER_RETENTION_DAYS: int = 7
    # Word-processing submissions retain native semantic structure and receive
    # a distinct controlled PDF presentation derivative when this local
    # capability is installed. Compose enables it in the pinned app image;
    # source-only host development fails visibly rather than using an unknown
    # desktop office installation implicitly.
    DOCX_PRESENTATION_RENDERING_ENABLED: bool = False
    DOCX_PRESENTATION_RENDERER_EXECUTABLE: str = "soffice"
    DOCX_PRESENTATION_RENDER_TIMEOUT_SECONDS: int = 120
    DOCX_PRESENTATION_RENDER_MAX_MB: int = 100
    # Only temporary is operational. The remaining recognized values reserve
    # the policy boundary for a later, separately authorized student-paper
    # comparison corpus; selecting one now fails closed at upload.
    PAPER_RETENTION_MODE: Literal[
        "temporary", "assessment", "course", "institutional"
    ] = "temporary"
    # Remote citation/relevance judgment is an explicit deployment capability.
    # When false, the workflow still extracts, retrieves, persists candidates,
    # and reports visible not-assessed model stages without sending paper text.
    PAPER_LLM_PROCESSING_ENABLED: bool = False

    # LLM Provider options (configure via LLM_BASE_URL + LLM_MODEL)
    # DeepSeek: LLM_BASE_URL="https://api.deepseek.com/v1", LLM_MODEL="deepseek-chat"
    # OpenAI: LLM_BASE_URL=None (default), LLM_MODEL="gpt-4o-mini"
    # Ollama: LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL="llama3.1"

    # Cache
    CACHE_ENABLED: bool = True  # Enable DOI/title-hash caching

    # OpenAlex
    OPENALEX_EMAIL: Optional[str] = None  # Recommended for polite pool
    OPENALEX_API_KEY: Optional[str] = None
    # Experimental Boolean candidate generation for several title-only works.
    # Disabled by default: the fixed-corpus run did not reduce provider calls.
    OPENALEX_GROUPED_TITLE_PREFETCH_ENABLED: bool = False

    # File processing limits
    MAX_FILE_SIZE_MB: int = 50
    MAX_TEXT_LENGTH_CHARS: int = 100_000  # Truncation guard
    MAX_PAPER_TEXT_LENGTH_CHARS: int = 2_000_000
    MAX_PDF_PAGES: int = 5000
    MAX_PDF_OBJECTS: int = 250_000

    # Hostile-file inspection. A clamd socket is deliberately opt-in because
    # its TCP protocol is unauthenticated and must remain on a trusted network.
    CLAMD_UNIX_SOCKET: str | None = None
    CLAMD_HOST: str | None = None
    CLAMD_PORT: int = 3310
    CLAMD_TIMEOUT_SECONDS: float = 30.0
    MALWARE_SCAN_REQUIRED: bool = True

    # Pure-scan OCR is an explicit local capability. It remains disabled until
    # the deployment has Tesseract and enough isolated worker capacity. The
    # resolver never falls back to remote OCR.
    PURE_SCAN_OCR_ENABLED: bool = False
    PURE_SCAN_OCR_LANGUAGE: Literal["eng"] = "eng"
    PURE_SCAN_OCR_EXECUTABLE: str = "tesseract"
    PURE_SCAN_OCR_DPI: int = 250
    PURE_SCAN_OCR_PAGE_SEGMENTATION_MODE: Literal[6] = 6
    PURE_SCAN_OCR_MAX_PAGES: int = 200
    PURE_SCAN_OCR_MAX_PIXELS_PER_PAGE: int = 25_000_000
    PURE_SCAN_OCR_MAX_TOTAL_PIXELS: int = 500_000_000
    PURE_SCAN_OCR_PAGE_TIMEOUT_SECONDS: int = 45
    PURE_SCAN_OCR_TOTAL_TIMEOUT_SECONDS: int = 900
    PURE_SCAN_OCR_MAX_DERIVATIVE_MB: int = 20
    PURE_SCAN_OCR_MIN_MEAN_WORD_CONFIDENCE: float = 70.0

    # ── Source Repository (Phase 3.5) ────────────────────────
    SOURCE_REPOSITORY_ENABLED: bool = False
    # Personal/local prototype scope. Institutional deployments must replace
    # this fixed scope with the authenticated tenant/course authorization layer.
    SOURCE_REPOSITORY_SCOPE_ID: str = "personal-default"

    # Retention of representations downloaded successfully from ordinary
    # public HTTP(S) URLs when no machine-readable licence/OA assertion is
    # available. "store_scoped" records them as rights_unclassified without
    # making a copyright/OA claim; "explicit_license_only" keeps the older
    # conservative behavior and uses them only for the current run.
    PUBLIC_RETRIEVAL_RETENTION_POLICY: Literal[
        "store_scoped", "explicit_license_only"
    ] = "store_scoped"
    # Periodic physical cleanup is separate from immediate fail-closed lookup:
    # expired representations are unusable at once even if object deletion is
    # waiting for the next retryable cleanup pass.
    SOURCE_RETENTION_CLEANUP_INTERVAL_SECONDS: int = 3600

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
    #      Fast, broad, now codebase-unified with Unpaywall. DOI lookups can be
    #      grouped into configurable filter batches. Title-only work uses
    #      individual relevance-gated queries by default. Experimental grouped
    #      Boolean candidate generation is opt-in because the fixed-corpus
    #      measurement did not reduce calls.
    #   2. Crossref — reliable DOI resolution, metadata + abstract only.
    #   3. CORE — unique OA PDFs from 10K+ repositories. v3 DOI lookups are
    #      grouped into small Boolean queries and cached per run. Requests stay
    #      serialized because concurrent calls stalled in live testing; quota
    #      state and retry timing come from CORE's X-RateLimit-* headers.
    #   4. Elsevier — only source for Elsevier full text (PII-based URL).
    #   5. Semantic Scholar — DOI-only OA supplement. DOI records are prefetched
    #      in conservative five-item batches; title matching is disabled.
    #   6. Gutenberg — preferred public-domain complete-work source. Uses the
    #      official OPDS catalog and downloadable EPUB editions.
    #   7. Wikisource — multilingual supplement. It fails closed where the root
    #      page is only an index and complete subpage aggregation is unavailable.
    #   8. web_search (optional, last) — searches configured discovery providers
    #      the academic-DB chain couldn't find. Only active when SEARCH_PROVIDER
    #      is configured. Add to the list to enable: "...,gutenberg,web_search"
    RETRIEVAL_SOURCES: str = "openalex,crossref,core,elsevier,semantic_scholar,gutenberg,wikisource"
    # JSON object containing safe operational overrides for installed retrieval
    # adapters. Credentials remain in their dedicated secret settings.
    # Example: {"semantic_scholar":{"batch_size":5,"max_batches":5}}
    RETRIEVAL_PROVIDER_CONFIG: str = "{}"
    # Conservative routing heuristic only, not a copyright determination.
    # Deployments should set this according to their jurisdiction and policy.
    PUBLIC_DOMAIN_CUTOFF_YEAR: int = 1928
    # Non-secret provider cooldown state. Personal uses this local file;
    # Institutional deployments can point all workers at a shared mounted path.
    PROVIDER_HEALTH_STATE_PATH: str = ".sourcefidelity/provider_health.json"
    # One recovery event refreshes only this many completed papers at a time.
    # Remaining dependencies stay durably attached to their paper checkpoints.
    PROVIDER_RECOVERY_MAX_JOBS: int = 25
    PROVIDER_HEALTH_PROBE_INTERVAL_SECONDS: int = 60
    # Reusable canonical abstract/miss cache. Abstract evidence is reused while
    # full-text discovery is refreshed on this bounded schedule. Increment the
    # access revision after a subscription, proxy, or library-route change.
    RETRIEVAL_LOOKUP_CACHE_ENABLED: bool = True
    RETRIEVAL_ABSTRACT_REFRESH_DAYS: int = 30
    RETRIEVAL_NEGATIVE_REFRESH_HOURS: int = 24
    RETRIEVAL_LOOKUP_CACHE_RETENTION_DAYS: int = 180
    RETRIEVAL_ACCESS_REVISION: str = "1"

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
    # Pluggable primary provider. Bing Search APIs were retired in August 2025
    # and are intentionally unsupported.
    # When set, the retrieval chain searches the web for source titles + "filetype:pdf"
    # and downloads/validates any PDFs found (author homepages, repositories, OA copies).
    # Source-access neutrality applies (§3.5): the app verifies against whatever it finds,
    # does NOT access Sci-Hub or pirated copies. Legitimate OA / author-homepage / institutional-repository PDFs only.
    SEARCH_PROVIDER: str | None = None  # "google"|"searxng"|"brave"|"duckduckgo"|"tavily"|"exa"|None
    # Comma-separated paid/bounded fallbacks, tried only when the primary
    # provider returns no usable candidates. Exact queries are cached per run.
    SEARCH_ESCALATION_PROVIDERS: str = "tavily,exa"
    # Per-process request ceilings for escalation providers. These are safety
    # limits, not targets. Format: comma-separated provider:count pairs.
    SEARCH_ESCALATION_MAX_CALLS: str = "tavily:50,exa:25"
    GOOGLE_SEARCH_API_KEY: str | None = None
    GOOGLE_SEARCH_CSE_ID: str | None = None  # Custom Search Engine ID
    # SearXNG (self-hosted meta-search — recommended for institutions)
    SEARXNG_URL: str | None = None  # e.g., "http://localhost:8080"
    # Ordered semicolon-separated engine groups. Each group may contain a
    # comma-separated SearXNG engine list. The next group runs only when the
    # earlier group yields no results. These defaults were live-checked from
    # the project Compose deployment; operators may replace them for their
    # regional network without changing application code.
    SEARXNG_ENGINE_GROUPS: str = "google cse;google scholar;brave"
    # Optional retries apply only to upstream timeouts. CAPTCHA/access/rate
    # failures open the circuit immediately. Retries remain off until a
    # controlled source-admission experiment demonstrates benefit.
    SEARXNG_TIMEOUT_RETRIES: int = 0
    SEARXNG_TIMEOUT_CIRCUIT_THRESHOLD: int = 3
    SEARXNG_REQUEST_TIMEOUT_SECONDS: float = 15.0
    # Per-request ceiling forwarded to SearXNG for its upstream engines. This
    # remains at the current three-second baseline until a controlled corpus
    # experiment establishes that a larger value improves admitted sources.
    SEARXNG_ENGINE_TIMEOUT_SECONDS: float = 3.0
    # SearXNG requests also honor RETRIEVAL_PROVIDER_CONFIG's existing
    # min_interval_seconds (default 1s), shared across local worker processes.
    # Increasing the engine limit requires the instance max_request_timeout
    # to permit it; the Compose template permits at most 6s, not an automatic
    # six-second production default.
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
