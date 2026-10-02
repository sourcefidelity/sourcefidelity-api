"""Application configuration using pydantic-settings."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings
from typing import Literal, Optional


class Settings(BaseSettings):
    # Credentials are SecretStr so they never appear in repr(), str(),
    # tracebacks or logs; read the raw value with secret_value(). Connection
    # URLs may embed passwords, so they are excluded from repr as well.

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
    DATABASE_URL: str = Field(default="postgresql://sourcefidelity:sourcefidelity@localhost:5432/sourcefidelity", repr=False)

    # Redis (Celery broker)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", repr=False)

    # Celery
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", repr=False)
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", repr=False)

    # S3 / MinIO
    S3_ENDPOINT: str = "http://localhost:9000"
    S3_ACCESS_KEY: SecretStr = SecretStr("sourcefidelity")
    S3_SECRET_KEY: SecretStr = SecretStr("sourcefidelity123")
    S3_BUCKET: str = "sourcefidelity-texts"
    S3_REGION: str = "us-east-1"

    # LLM (OpenAI-compatible)
    LLM_API_KEY: SecretStr | None = None
    LLM_MODEL: str = "deepseek-v4-flash"  # Default: DeepSeek V4 Flash (cheapest, JSON mode supported)
    LLM_BASE_URL: Optional[str] = None  # e.g., https://api.deepseek.com/v1
    LLM_BATCH_SIZE: int = 10  # References per LLM call
    LLM_MAX_RETRIES: int = 2  # Retries for JSON parse failures
    LLM_TEMPERATURE: float = 0.0  # Deterministic output
    LLM_MAX_TOKENS: int = 8192  # Max tokens per response (DeepSeek max output)
    # Optional second-provider credential for bounded, development-only model
    # comparisons. Runtime LLM routing continues to use LLM_* above.
    OPENAI_API_KEY: SecretStr | None = None
    # Development-only OpenRouter supplemental opinion. This never enters the
    # three-arm formal denominator or runtime routing. Calls remain fail-closed
    # unless the local gate is deliberately enabled.
    OPENROUTER_ENABLED: bool = False
    OPENROUTER_API_KEY: SecretStr | None = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "z-ai/glm-5.3-flash"
    OPENROUTER_DATA_COLLECTION: Literal["deny"] = "deny"
    OPENROUTER_ZDR: bool = True
    OPENROUTER_REQUIRE_PARAMETERS: bool = True
    # Match the time-bounded launch discount. Requests fail rather than
    # silently paying the doubled standard rates after the promotion ends.
    OPENROUTER_MAX_PROMPT_USD_PER_MILLION: float = 0.075
    OPENROUTER_MAX_COMPLETION_USD_PER_MILLION: float = 0.25

    # Qwen API arm: Alibaba Cloud Model Studio, OpenAI-compatible endpoint.
    # A formal panel arm (ARCHITECTURE §7) with standing authorization in
    # AGENTS; development-only and fail-closed like OpenRouter, never runtime
    # routing. Terms verified 2026-09-24 against the provider's documentation:
    # API inputs are not used for training, but call data is stored by the
    # provider under applicable regulation and no zero-retention option exists,
    # so retention must be acknowledged deliberately rather than assumed away.
    # An API key is bound to the region of the base URL it was created in.
    QWEN_ENABLED: bool = False
    QWEN_API_KEY: SecretStr | None = None
    QWEN_BASE_URL: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen3.8-flash"
    QWEN_RETENTION_ACKNOWLEDGED: bool = False
    QWEN_TERMS_VERIFIED_ON: str = "2026-09-24"
    # Documented international-endpoint list price at verification; requests
    # are refused rather than silently paying more if the configured rate is
    # ever raised above the ceiling the owner accepted.
    QWEN_MAX_PROMPT_USD_PER_MILLION: float = 0.15
    QWEN_MAX_COMPLETION_USD_PER_MILLION: float = 0.47

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
    PAPER_EXPERIMENTAL_RELATIONSHIP_JUDGMENTS_ENABLED: bool = False
    # Input bound for one Judgment prompt (up to 128 real sentences per claim
    # since foundation v11; measured median about 7,800 and max about 11,100
    # tokens, STATE). Still a bounded excerpt, never a complete source file.
    JUDGMENT_MAX_INPUT_TOKENS: int = 16_000
    # Development only: the three Judgment arms answer from a deterministic
    # local stand-in instead of any provider (no network, no spend), so the
    # layout can be built and checked. Runs record it and never share a cache.
    JUDGMENT_FAKE_PANEL: bool = False
    # Owner decision 2026-09-27: one judge (GLM-5.3-Flash through Z.ai's own
    # API); DeepSeek keeps every other model task, including the notes. The
    # three-arm panel ("deepseek,glm,qwen") is kept as a post-prototype option.
    JUDGMENT_JUDGES: str = "zai_glm"
    # Owner decision 2026-09-30: a single judge answers each claim this many
    # times and the majority result is shown (1 restores one call per claim).
    JUDGMENT_SAMPLES: int = 3
    # Owner decision 2026-09-29: Judgment runs on every checked paper. The
    # route still fails closed without a key and a terms-verified date.
    ZAI_ENABLED: bool = True
    ZAI_API_KEY: SecretStr | None = None
    ZAI_BASE_URL: str = "https://api.z.ai/api/paas/v4"
    ZAI_MODEL: str = "glm-5.3-flash"
    # GLM-5.3-Flash cannot turn thinking off (Z.ai documentation, 2026-09-27);
    # the lowest effort keeps the answer inside the output allowance.
    ZAI_REASONING_EFFORT: str = "low"
    ZAI_TERMS_VERIFIED_ON: str | None = None
    ZAI_MAX_PROMPT_USD_PER_MILLION: float = 0.15
    ZAI_MAX_COMPLETION_USD_PER_MILLION: float = 0.50
    # Spend limits for Judgment runs, in USD. Unset until measured spend lets
    # the owner choose them (owner decision 14); spend is recorded either way.
    JUDGMENT_MAX_USD_PER_REPORT: float | None = None
    JUDGMENT_MAX_USD_PER_DAY: float | None = None
    # Wider-search sentence reserve (owner decision 13). "paper_retention":
    # kept with the verification report and deleted with it (Personal default);
    # "until_grades_released": Institutional, kept until marks are released for
    # the paper's assessment (assessment_marks.py: set by an instructor now, by
    # the planned Moodle signal later); a paper without an assessment keeps it
    # with the verification report;
    # "off": not built, so an agreed "no evidence" stays not judged.
    JUDGMENT_RESERVE_RETENTION: Literal["paper_retention", "until_grades_released", "off"] = "paper_retention"
    # Pace rate-limited providers (Semantic Scholar, CORE) across every worker
    # process through Redis; falls back to per-process pacing when unavailable.
    RETRIEVAL_SHARED_PACING_ENABLED: bool = True
    # The web fallback after Brave and Exa: "searxng" (default), "tavily" or "none".
    # Tavily is to be tested as SearXNG's replacement (owner decision 2026-09-25).
    SEARCH_WEB_FALLBACK_PROVIDER: Literal["searxng", "tavily", "none"] = "searxng"
    # Your Tavily plan's price per credit, for Technical details; unset means
    # "no price on record", never zero.
    TAVILY_USD_PER_CREDIT: float | None = None
    # Brave reports neither credits nor cost. Its published Search price was
    # $5.00 per 1,000 requests on 2026-09-25 (api-dashboard.search.brave.com,
    # before the $5 monthly free credit); recorded per request Brave served.
    BRAVE_USD_PER_REQUEST: float | None = 0.005
    # Bright Data bills successful SERP requests only, at a plan-dependent
    # price not shown publicly; unset means "no price on record", never zero.
    BRIGHTDATA_USD_PER_REQUEST: float | None = None
    PAPER_EXPERIMENTAL_JOINT_SELECTION_ENABLED: bool = False
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
    # Remote citation/relevance judgment, on by default (owner decision
    # 2026-09-29). When false, the workflow still extracts, retrieves, persists
    # candidates, and reports visible not-assessed model stages without sending
    # paper text.
    PAPER_LLM_PROCESSING_ENABLED: bool = True
    SOURCE_INSPECTION_ENABLED: bool = False
    SOURCE_INSPECTION_REMOTE_ALLOWED: bool = False
    SOURCE_INSPECTION_MAX_CALLS: int = 6

    # LLM Provider options (configure via LLM_BASE_URL + LLM_MODEL)
    # DeepSeek: LLM_BASE_URL="https://api.deepseek.com/v1", LLM_MODEL="deepseek-chat"
    # OpenAI: LLM_BASE_URL=None (default), LLM_MODEL="gpt-4o-mini"
    # Ollama: LLM_BASE_URL="http://localhost:11434/v1", LLM_MODEL="llama3.1"

    # Cache
    CACHE_ENABLED: bool = True  # Enable DOI/title-hash caching

    # OpenAlex
    OPENALEX_EMAIL: Optional[str] = None  # Recommended for polite pool
    OPENALEX_API_KEY: SecretStr | None = None
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
    # Page-level text check of PDF sources, with local OCR repair of damaged pages
    # (owner decision 2026-09-28). A source whose damaged pages cannot be repaired
    # is rejected. Off only for tests that do not exercise it.
    SOURCE_TEXT_QUALITY_CHECK_ENABLED: bool = True

    # ── Source Repository (Phase 3.5) ────────────────────────
    SOURCE_REPOSITORY_ENABLED: bool = False
    # A saved publisher page the operator already holds may enter intake
    # through the ordinary identity/coverage checks. Default off: it is a
    # separate provenance boundary from an application-fetched response.
    SUPPLIED_HTML_SOURCE_INTAKE_ENABLED: bool = False
    # retired 2026-09-29; remove after the .env line is deleted. Ignored: the
    # deployed app keeps no search-result links, and SEARCH_CANDIDATE_AUDIT_URLS
    # below is the only development audit of web-search candidates.
    DEVELOPMENT_RETAIN_DISCOVERY_LEAD_URLS: bool = False
    # Development audit only. Records each Exa, Tavily and SearXNG web-search
    # candidate's URL, query id, rank and disposition under `candidate_audit`
    # on the bounded-web discovery attempt. Must stay False in any deployed
    # profile. Brave is never recorded (Brave API terms §3.2(i)).
    SEARCH_CANDIDATE_AUDIT_URLS: bool = False
    # A student's own cited URL was deliberately chosen and deserves patience.
    # A speculative search hit is one of up to five tried per reference, and the
    # same 30s budget meant one reference could wait 150s on candidates that
    # never arrive. Measured 2026-09-21: 85% of web-search time was this wait.
    # This is a total deadline including body transfer, not a connect timeout,
    # so lowering it too far would drop large PDFs from slow servers — the
    # retrieval counts, not the wall time, are what must hold.
    #
    # Held at the student-URL value pending evidence. A 15.0s trial could not be
    # judged: across four Stardom runs the number of references retrieving
    # anything swung 12-17 and wall time 1,114-1,845s on configurations that
    # cannot plausibly explain a five-source difference, so run-to-run variance
    # exceeds the effect being measured. The split exists so this can be tuned
    # once per-fetch durations are recorded; until then, changing it would be
    # trading retrieval for wall time on the strength of noise.
    DISCOVERY_CANDIDATE_TIMEOUT_SECONDS: float = 30.0
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
    RETRIEVAL_SOURCES: str = "openalex,crossref,core,elsevier,semantic_scholar,datacite,open_library,eric,gutenberg,wikisource"
    # Europe PMC open-access repository copies, added after Crossref unless
    # RETRIEVAL_SOURCES names it (owner decision 2026-10-02).
    EUROPE_PMC_ENABLED: bool = True
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
    # Re-run the incomplete references of completed papers when a provider
    # that was blocking them recovers. On by default: a reference left
    # unfinished by a temporary outage is finished without resubmission,
    # and each re-run writes a new report version with the earlier one
    # preserved. Set false to stop all automatic re-runs; provider health
    # probing continues either way, since live searches rely on it.
    PROVIDER_RECOVERY_REFRESH_ENABLED: bool = True
    PROVIDER_RECOVERY_MAX_JOBS: int = 25
    PROVIDER_HEALTH_PROBE_INTERVAL_SECONDS: int = 60
    # Automatic retries of a completed paper's incomplete searches (owner
    # decision 2026-09-29). Each retry is a targeted refresh of the affected
    # references and writes a new report version. Retry n runs the n-th delay
    # after the incomplete search it retries; a reference gets one retry per
    # delay, at most. A retry whose due time passed more than the window ago
    # is not started, so incomplete searches older than the schedule are left
    # as they are. At most MAX_JOBS papers start a retry per scan.
    # PROVIDER_RECOVERY_REFRESH_ENABLED is the one switch for these retries
    # too (owner decision 2026-09-29).
    INCOMPLETE_SEARCH_RETRY_SCAN_SECONDS: int = 600
    INCOMPLETE_SEARCH_RETRY_DELAYS_HOURS: str = "1,24,72"
    INCOMPLETE_SEARCH_RETRY_WINDOW_HOURS: int = 24
    INCOMPLETE_SEARCH_RETRY_MAX_JOBS: int = 10
    # Reusable canonical abstract/miss cache. Abstract evidence is reused while
    # full-text discovery is refreshed on this bounded schedule. Increment the
    # access revision after a subscription, proxy, or library-route change.
    RETRIEVAL_LOOKUP_CACHE_ENABLED: bool = True
    RETRIEVAL_ABSTRACT_REFRESH_DAYS: int = 30
    RETRIEVAL_NEGATIVE_REFRESH_HOURS: int = 24
    RETRIEVAL_LOOKUP_CACHE_RETENTION_DAYS: int = 180
    RETRIEVAL_ACCESS_REVISION: str = "1"

    # CORE API
    CORE_API_KEY: SecretStr | None = None

    # Semantic Scholar API
    S2_API_KEY: SecretStr | None = None

    # Elsevier API (Article Retrieval — OA full text + metadata/abstract for paywalled)
    # API key alone: OA articles + metadata/abstract for all.
    # Insttoken (optional, via institutional email to apisupport@elsevier.com):
    #   unlocks paywalled full text if your institution subscribes.
    ELSEVIER_API_KEY: SecretStr | None = None
    ELSEVIER_INST_TOKEN: SecretStr | None = None

    # Crossref polite email (uses OPENALEX_EMAIL as fallback)
    CROSSREF_EMAIL: str | None = None

    # Web search provider for PDF fallback retrieval (after academic-DB chain fails).
    # Pluggable primary provider. Bing Search APIs were retired in August 2025
    # and are intentionally unsupported.
    # When set, the retrieval chain searches the web for source titles + "filetype:pdf"
    # and downloads/validates any PDFs found (author homepages, repositories, OA copies).
    # Source-access neutrality applies (§3.5): the app verifies against whatever it finds,
    # does NOT access Sci-Hub or pirated copies. Legitimate OA / author-homepage / institutional-repository PDFs only.
    # Which abstract-scope prompt the assessment uses. v6 clarifies only the
    # general-point exclusion, which was swallowing a statement about one
    # national industry supported by a source stating a different one.
    # Revert with ABSTRACT_SCOPE_POLICY_VERSION=abstract-topic-v5; stored
    # records keep the version they were written under and are read by it.
    # Default v5. v6 was measured on the case it was written for (Ryan &
    # Hearn, an Australian filmmaking study cited for a claim about
    # postclassical Hollywood distribution) and did not change the outcome:
    # the model names both scopes correctly and still answers
    # stated_scope_conflict=absent. It is kept selectable, but it costs 439
    # characters of abstract budget for no measured gain, so it is not the
    # default. Set ABSTRACT_SCOPE_POLICY_VERSION=abstract-topic-v6 to use it.
    ABSTRACT_SCOPE_POLICY_VERSION: Literal[
        "abstract-topic-v5", "abstract-topic-v6"
    ] = "abstract-topic-v5"
    SEARCH_PROVIDER: str | None = None  # "google"|"searxng"|"brave"|"brightdata"|"tavily"|"exa"|None
    # New resolutions use the versioned API-first policy. Legacy configuration
    # remains selectable; stored traces never inherit a new policy implicitly.
    SEARCH_POLICY_VERSION: Literal["configured-search-v1", "api-first-search-v2"] = "api-first-search-v2"
    # A self-hosted metasearch fallback behind the two paid APIs. Enabled
    # 2026-09-21 after measuring 11-20 results in 0.5-1.5s from this network,
    # against a 59-69s mean per paid search call, and after an Exa outage
    # removed six findings from a single paper.
    SEARCH_SEARXNG_FALLBACK_ENABLED: bool = True
    # Required API routes fail closed until account-specific retention rights
    # are confirmed. Credits alone are not a storage/evaluation permission.
    BRAVE_SEARCH_RETENTION_PERMITTED: bool = False
    # Ordinary Brave discovery uses operation-local data, not storage rights.
    BRAVE_SEARCH_TRANSIENT_ENABLED: bool = True
    # Owner decision 2026-09-29: after a reference's paid web search completed
    # without finding its source, later runs in the same authorization scope
    # reuse that dated outcome for this many days instead of paying for the
    # same search again (`search-reuse-memo-v1`). Free academic adapters still
    # run; an incomplete search never creates a memo; a source upload or
    # `force_search` bypasses it. 0 disables reuse.
    # Patchwriting detector (patchwriting-v4) run during each paper check while
    # every retrieved source's text is authorized (owner decision 2026-09-29).
    # Results are stored under the verification summary and shown in the report
    # (yellow Academic Practice marks and window lines).
    PATCHWRITING_AT_CHECK_ENABLED: bool = True
    SEARCH_REUSE_PAUSE_DAYS: int = Field(default=30, ge=0)
    # A re-run of a paper reuses an earlier run's completed result for each
    # unchanged reference within this many days (paper-search-reuse-v1,
    # owner decision 2026-10-01). 0 disables reuse.
    SEARCH_RERUN_REUSE_DAYS: int = Field(default=30, ge=0)
    # Comma-separated paid/bounded fallbacks, tried only when the primary
    # provider returns no usable candidates. Exact queries are cached per run.
    SEARCH_ESCALATION_PROVIDERS: str = "tavily,exa"
    # Per-process request ceilings for escalation providers. These are safety
    # limits, not targets. Format: comma-separated provider:count pairs.
    SEARCH_ESCALATION_MAX_CALLS: str = "tavily:50,exa:25"
    # Calls to each required API provider every reference keeps after the
    # job-wide ceiling above is spent, so a paper's later references are not
    # left with a skipped required search. Brave's ladder is at most three
    # queries (DOI, title, title PDF); Exa's two (no filetype operator).
    # provider:count pairs, bounded by the hard cap below.
    WEB_SEARCH_PER_REFERENCE_FLOOR: str = "brave:3,exa:2"
    # Absolute per-run ceiling on required API calls, including floor calls;
    # never below SEARCH_ESCALATION_MAX_CALLS. Worst case about $1.16 per
    # paper run at list prices (see WebSearchRetriever._policy_call_permitted).
    # Lowered from brave:200,exa:150 on 2026-09-29: the spend audit
    # (tmp/search_spend_audit_v1) found no run above brave 91 / exa 51 calls.
    WEB_SEARCH_HARD_MAX_CALLS: str = "brave:120,exa:80"
    GOOGLE_SEARCH_API_KEY: SecretStr | None = None
    GOOGLE_SEARCH_CSE_ID: str | None = None  # Custom Search Engine ID
    # SearXNG (self-hosted meta-search — recommended for institutions)
    SEARXNG_URL: str | None = None  # e.g., "http://localhost:8080"
    # Ordered semicolon-separated engine groups. Each group may contain a
    # comma-separated SearXNG engine list. The next group runs only when the
    # earlier group yields no results. These defaults were live-checked from
    # the project Compose deployment; operators may replace them for their
    # regional network without changing application code.
    # Live-checked 2026-09-21 against the project Compose deployment by reading
    # the engine credited on each result, not the result count: SearXNG falls
    # back to its default engines when a name is unrecognised, which makes an
    # unavailable engine look like a working one. Verified answering: google
    # cse (20), google scholar (10), bing (10), openairepublications (10).
    # "brave" is dropped: it times out on every call here, and the application
    # already queries the Brave API directly. DuckDuckGo and Startpage return
    # CAPTCHAs from this address and are not configured.
    # Verified engine by engine against the per-result `engines` field, because
    # Verify an engine by the relevance of its results, never by their count.
    # SearXNG silently answers an unrecognised name with its default engine
    # set, and `bing` answers a phrase query with results for its first word:
    # a quoted book title returned dictionary entries for "The". Measured over
    # five real reference titles, `google cse` is the only general engine that
    # works from here (5/5 relevant, 20 results, 0.44s) and it needs no
    # credentials; `bing` scored 0/5 while returning 10 results every time.
    # This engine is a scraper: Google suspends it with "too many requests"
    # under sustained load, which is why SearXNG stays an unrequired fallback.
    SEARXNG_ENGINE_GROUPS: str = "google cse"
    # Optional retries apply only to upstream timeouts. CAPTCHA/access/rate
    # failures open the circuit immediately. Retries remain off until a
    # controlled source-admission experiment demonstrates benefit.
    SEARXNG_TIMEOUT_RETRIES: int = 0
    SEARXNG_TIMEOUT_CIRCUIT_THRESHOLD: int = 3
    # First cooldown after an engine answers "too many requests". Each further
    # consecutive failure multiplies it by four (health store); success resets.
    SEARXNG_RATE_LIMIT_COOLDOWN_SECONDS: int = 30
    SEARXNG_REQUEST_TIMEOUT_SECONDS: float = 15.0
    # Per-request ceiling forwarded to SearXNG for its upstream engines. This
    # remains at the current three-second baseline until a controlled corpus
    # experiment establishes that a larger value improves admitted sources.
    # Above the measured 2.9-8.0s engine latency from this network, and below
    # the container's 20s max_request_timeout ceiling.
    SEARXNG_ENGINE_TIMEOUT_SECONDS: float = 12.0
    # SearXNG requests also honor RETRIEVAL_PROVIDER_CONFIG's existing
    # min_interval_seconds (default 1s), shared across local worker processes.
    # Increasing the engine limit requires the instance max_request_timeout
    # to permit it; the Compose template permits at most 6s, not an automatic
    # six-second production default.
    # Brave Search API; account credits and retention rights are independent.
    BRAVE_SEARCH_API_KEY: SecretStr | None = None
    # Tavily (AI-focused search, free tier 1000/month)
    TAVILY_API_KEY: SecretStr | None = None
    # Exa (neural/semantic search, free tier — good for academic content)
    EXA_API_KEY: SecretStr | None = None
    # Mojeek (independent index). Evaluation harness only; not a search route.
    # The key travels in the query string, so callers must keep request URLs
    # out of logs and exception text.
    MOJEEK_API_KEY: SecretStr | None = None
    # Bright Data SERP API: a contracted service returning parsed engine
    # results. Both the token and a SERP zone name are required; the zone is
    # created in the Bright Data console and names the product being billed.
    # Unset by default — an absent token is a provider that is not configured,
    # never an empty search.
    BRIGHTDATA_API_TOKEN: SecretStr | None = None
    BRIGHTDATA_SERP_ZONE: str | None = None
    # Which engine the SERP zone queries. Verified by relevance on real
    # reference titles before any reliance, as with every other provider.
    BRIGHTDATA_SERP_ENGINE: Literal["google", "bing"] = "google"
    # Measured 2026-09-22 from this network: median 6-31s across runs with
    # maxima near 80s, so 30s produced spurious timeouts. The retrieval
    # deadline still caps this, so a slow provider cannot extend a run.
    # Budget for one search in total, retries and their pauses included.
    BRIGHTDATA_TIMEOUT_SECONDS: float = 60.0
    # Interface language and result region sent with every query, and the
    # collection region pinned in the request body. Without them the
    # engine and the zone each infer one, so identical queries can return
    # different result sets between calls.
    BRIGHTDATA_SEARCH_LANGUAGE: str = "en"
    BRIGHTDATA_SEARCH_REGION: str = "us"

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
    GOOGLE_BOOKS_API_KEY: SecretStr | None = None

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

    @field_validator("TAVILY_USD_PER_CREDIT", "BRAVE_USD_PER_REQUEST", "BRIGHTDATA_USD_PER_REQUEST",
                     mode="before")
    @classmethod
    def _plain_price(cls, value):
        """Accept a price written as "$0.008" or "0.008 USD"; a stray currency
        sign must not stop the whole application from starting."""
        if isinstance(value, str):
            value = value.strip().removeprefix("$").removesuffix("USD").strip()
            return value or None
        return value


def secret_value(secret: SecretStr | None) -> str | None:
    """Return a credential's raw value for the one call that needs it."""
    return secret.get_secret_value() if secret is not None else None


settings = Settings()
