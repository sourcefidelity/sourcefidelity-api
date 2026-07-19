# SourceFidelity API

**Self-hosted source-verification tool for academic papers — for instructors, institutions, and reviewers.**

SourceFidelity verifies the citations in a student paper against the actual cited sources — retrieving full texts from open-access databases, publisher sites, direct PDF links, HTML pages, and instructor-uploaded copies — then checking whether each reference is accurate and if each quotation and paraphrase is supported. It is evidence for an instructor's or reviewer's judgment, not an automated verdict.

It is primarily an **institutional** tool (deployed on a department or university server, integrated via the planned Moodle plugin), but it is also usable directly by an **individual instructor** who wants to use it independently of their institution, and by **journal editors and conference organizers** for manuscript screening.

> **Status:** Pre-release, active development. The reference-extraction, source-retrieval, and in-text-citation-extraction pipelines are working and validated on real student papers. The verification engine (4-verdict scoring) and reporting system are under construction. See [Roadmap](#roadmap).

---

## Why SourceFidelity

Existing tools (Turnitin and similar) **only** focus on similarity matching — finding text that looks copied. SourceFidelity *also* focuses on similarity matching (patchwriting and plagiarism detection are part of the roadmap), but its distinctive focus is on **source verification**: did this student represent their sources accurately? SourceFidelity targets that gap:

- **Verifies cited claims against retrieved source texts**, not just surface similarity.
- **Four verdicts** per citation: *Consistent*, *Misrepresentation*, *Topical mismatch*, *Inconclusive*. Each carries a distinct judgement so instructors can distinguish honest misreading from likely fabrication.
- **Text-availability confidence** so verdicts from full, retrieved texts and paywalled/abstract-only sources are honestly marked with different confidence.
- **Self-hosted by the institution or instructor.** Student papers never leave your infrastructure. Cloud LLMs (if used) receive text with PII stripped beforehand.
- **Benefit-of-doubt by design.** When the system can't tell whether a passage is the student's own analysis or an uncited paraphrase, it defaults to not flagging. A false accusation of plagiarism destroys trust permanently.
- **Formative when integrated, analytical when standalone.** When deployed through the planned Moodle plugin, students can submit their own papers for screening and receive a simplified, pedagogically-framed report — patterns and suggestions rather than punitive scores — and the plugin's LLM explains problems without generating corrected text. Used this way SourceFidelity is a learning tool. When used standalone by an instructor (or journal editor), it functions as analytical evidence for the reviewer's own judgment.
- **Source verification, not AI-text detection.** Research consensus is that AI-text detectors have 16–61% false-positive rates and are biased against L2 and neurodivergent writers. SourceFidelity does not build one. Instead, it detects fabricated references and quotations and paraphrases that do not match their sources. These are reliable and verifiable signals that demonstrate a lack of engagement with sources regardless of whether the student used AI.

---

## Features

### Working now

- **Text extraction** — PDF (dual-backend: pdfplumber + PyMuPDF fallback) and DOCX, including bytes-based handling for direct uploads.
- **Reference extraction** — LLM-first parsing of the reference list, format-aware (APA 7th, MLA 9th). Validated on 10 APA papers (99.3% reference recall) and 11 MLA papers (100%).
- **In-text citation extraction** — hybrid regex + LLM extraction of quotations and paraphrases from the body, with author-surname and reference-title injection. The extractor distinguishes three citation types because they have very different difficulty levels:
  - **Explicit quotations** (text in quote marks with citation markers) — highest recall, regex catches these directly.
  - **Paraphrases with in-text citations in the sentence** (parenthetical or narrative markers like `(Smith, 2020)` or `Smith argues`).
  - **Continuation paraphrases** (sentences with no marker that continue discussing a previously-cited source — "This reflects...", "He argues...") — hardest case, addressed via the LLM two-pass stage and title/topic injection.
  - Per-type precision/recall numbers will be published once ground-truth annotation on a larger paper corpus is complete. The verifier also revises extraction using the retrieved sources: every sentence is embedded against retrieved sources, so over-extracted low-confidence continuations that don't match their cited source are silently removed from the attribution (rather than flagged), while genuine non-engagement at explicit citations is still caught. This keeps the "topical mismatch" verdict reserved for cases that actually signal non-engagement.
- **Source retrieval** — resolution chain across Elsevier, OpenAlex, CORE, Semantic Scholar, and Crossref for academic texts, plus Project Gutenberg and Wikisource for public-domain primary texts. **Beyond academic databases:** direct PDF link/URL resolution, HTML page fetching, publisher-PDF URL construction for six major publishers (Springer, Taylor & Francis, Wiley, SAGE, OUP, Cambridge), and configurable retrieval-source priority.
- **Source repository** — instructor uploads of articles, books, and book chapters to a local source cache, with edited-collection splitting, completeness checking, and N-up scan detection. **Optionally stores and vectorizes every retrieved source** so future papers citing the same source skip re-retrieval and re-verification — the sources are pre-indexed and ready for immediate use in judgment, making the app progressively faster and cheaper to run as the cache grows.
- **Abstract verification** — when a paywalled source has only an abstract available, student citations are checked against the abstract at medium confidence.
- **Website source verification** — in-memory fetch, verify, discard. Web content is never persisted to the repository.
- **Link validation** — every URL in a reference list is checked and categorized (content_match / paywall / dead / redirect / mismatch / etc.) for the instructor report.
- **Model-agnostic LLM** — provider config dict supports DeepSeek (default, cheapest), OpenAI GPT-4, Anthropic Claude, and local Ollama. No provider lock-in.
- **Asynchronous jobs** — Celery + Redis with retry, time/memory limits, and a dead-letter queue.
- **REST API** — FastAPI with Swagger UI at `/docs`.

### In progress

- **Verification engine** — 6-stage hybrid pipeline (similarity gate → atomic-claim extraction → NLI + AlignScore → RAGAS groundedness → LLM judge → aggregate). Uses BGE-M3 multilingual embeddings (chosen for Chinese + English support) and FAISS for passage retrieval.
- **Reporting system** — individual and batch reports, HTML dashboard (Jinja2 + Chart.js), color-highlighted paper rendering, and a student-facing simplified report.

### Planned

- **Patchwriting and plagiarism detection** — Keck overlap thresholds for patchwriting against cited sources, Wikipedia plagiarism checks, within-batch and historical cross-paper comparison (FAISS index of anonymized papers).
- **Moodle plugin** (`sourcefidelity-moodle`) — student submission, instructor report, and the formative student-facing experience described above.
- **Chinese-language support** — GB/T 7714 reference format, numeric `[1]` citation markers, AMiner and ChinaXiv retrieval adapters, Chinese-web-optimized search fallback, Chinese UI localization.
- **Authentication** — Student / Instructor / Admin roles, API key or JWT, optional LDAP.

---

## How it works

```
1. Text extraction          PDF/DOCX → plain text
2. PII stripping            name, ID, email, university removed before LLM calls
3. Reference parsing        LLM splits + parses the reference list
4. Subject identification   LLM tags primary vs secondary sources, paper keywords
5. Citation extraction      regex + LLM find attributed sentences in the body
6. Reference-body check     phantom references, orphan citations (no LLM)
7. Source retrieval         chain: Elsevier → OpenAlex → CORE → S2 → Crossref
                            → Gutenberg → Wikisource → publisher PDF →
                            direct PDF link → HTML fetch → web search
8. Extraction revision      retrieved sources used to correct extraction:
                            low-similarity low-confidence continuations
                            removed from attribution; high-confidence
                            extractions with low similarity kept (genuine
                            non-engagement signal). Keeps "topical mismatch"
                            verdict reserved for real cases.
9. Verification             quotation exact-match → embedding retrieval →
                            LLM judgement → patchwriting scan
10. Integrity analysis      fabricated references, cross-paper, Wikipedia
11. Report generation       individual + batch, HTML, color-coded
```

The architecture is modular by design: each stage (extraction, parsing, retrieval, verification, reporting) is an independent service, so individual components can be replaced or extended without rewriting the pipeline.

---

## Tech stack

- **Python 3.12**, FastAPI, Celery
- **PostgreSQL** (job + report tracking), **Redis** (Celery broker), **MinIO / S3** (source text cache)
- **pdfplumber + PyMuPDF** (PDF extraction), **python-docx** (DOCX)
- **httpx** (HTTP client), **trafilatura** (HTML readable-text extraction)
- **Pydantic + pydantic-settings** (schemas, config)
- **Alembic** (database migrations)
- **Docker Compose** (deployment)

---

## Quick start

### Option A — Docker Compose (recommended)

```bash
git clone https://github.com/sourcefidelity/sourcefidelity-api.git
cd sourcefidelity-api
cp .env.example .env          # fill in your LLM + retrieval API keys
docker-compose up -d
```

- API root: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>
- MinIO console: <http://localhost:9001> (default credentials in `.env.example`)

### Option B — Local development

```bash
git clone https://github.com/sourcefidelity/sourcefidelity-api.git
cd sourcefidelity-api
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in your keys

# You'll need PostgreSQL + Redis running (or use docker-compose for just those)
docker-compose up -d postgres redis minio

alembic upgrade head          # apply database migrations
uvicorn app.main:app --reload # API at http://localhost:8000
```

### Required API keys

- **LLM_API_KEY** — DeepSeek (default), OpenAI, Anthropic, or point at a local Ollama instance.
- **OPENALEX_API_KEY** — required since Feb 2025. Free at <https://openalex.org/settings/api>.
- **CORE_API_KEY** — free at <https://core.ac.uk/services/api>.
- Optional: Semantic Scholar, Google Books, Elsevier (for full-text retrieval).

See [`.env.example`](./.env.example) for the full configuration reference, including retrieval-source priority, strictness mode, and paywalled-PDF caching policy.

---

## Deployment models

| | Option A — Personal computer | Option B — Institutional server |
|---|---|---|
| Hardware | Any personal computer, no GPU | 16–32 GB RAM server, GPU optional |
| LLM | Cloud API (DeepSeek/OpenAI/Claude) | Cloud or local (Ollama) |
| Local models | None (LLM handles verification) | BGE-M3, FAISS, DeBERTa, AlignScore |
| Patchwriting detection | Weaker (no lexical-overlap counting) | Strong (Keck + embeddings) |
| Cross-paper comparison | Not practical | Full (FAISS index) |

Per-paper cost figures will be published once measured against real batch runs, not estimates. Both deployments keep student data on the user's own infrastructure.

---

## Roadmap

A high-level view:

- ✅ Phases 0–2 — project setup, dev environment, core structure
- ✅ Phase 3.1–3.5 — text/reference extraction, source repository, retrieval adapters
- ✅ Phase 3.6–3.7 — download paths, publisher PDFs, website source verification
- ✅ Phase 4 — in-text citation extraction (APA/MLA)
- ⏳ Phase 3.8 — verification engine (6-stage hybrid pipeline)
- ⏳ Phase 5 — reporting system + color-highlighted paper output
- ⏳ Phase 6 — Celery parallelism (paper + reference level)
- 📝 Phase 9 — Moodle plugin (`sourcefidelity-moodle`, GPLv3)
- 📝 Phase 15 — Chinese-language support (GB/T 7714, AMiner, Chinese-web-optimized search)
- 📝 Phase 12 — authentication and roles

---

## Contributing

Contributions are welcome, especially around:

- **Chinese-language support** (Phase 15) — GB/T 7714 parsing, AMiner/ChinaXiv adapters, UI localization. See the `Phase 15` issues for scoped tasks.
- **Retrieval coverage** — additional adapters (CNKI is a known gap with no clean public API).
- **Citation format support** — Chicago notes-bibliography, Harvard, Vancouver.
- **Testing** — pytest coverage, real-paper test cases (with PII removed).

A `CONTRIBUTING.md` with setup instructions and coding conventions is in progress. In the meantime, please open an issue before starting work on a non-trivial change.

---

## License

MIT. See [`LICENSE`](./LICENSE).

The companion Moodle plugin (`sourcefidelity-moodle`, planned) will be GPLv3 to match Moodle's licensing requirements.

---

## Acknowledgments

SourceFidelity's verification model draws on research from writing studies, composition pedagogy, and applied linguistics — including work by Hyland, Keck, Howard, Swales, Graff & Birkenstein, and Teufel on paraphrase taxonomy, argumentative zoning, and academic attribution; on NLI / RAG-faithfulness / fact-checking research for the verification pipeline; and on academic-integrity principles from ICAI, QAA, and TEQSA.
