# SourceFidelity API

**Open-source, AI-augmented analysis of how academic papers use their sources.**

SourceFidelity identifies, verifies and retrieves the sources a paper cites, then produces an inspectable report on how the paper uses them: whether each cited work can be located, how each citation relates to the retrieved source text, where academic practice or referencing style needs attention, and what evidence each finding rests on.

It produces evidence for an instructor's own judgment and feedback to help students improve their writing. It does not detect AI-written text, determine intent or misconduct, assign grades or sanctions, or replace an institution's academic-integrity process.

> **Status:** Working prototype. The full pipeline — paper upload to evidence package to live and exported report — runs end to end on real student papers. Prototype acceptance is still in progress; reference-verification accuracy is the current focus. Moodle and library integrations are planned, not built. See [Roadmap](#roadmap).

---

## Why SourceFidelity

Generative AI has made it easy to submit work that cites sources the writer has not read, or that do not say what the paper claims. Existing tools do not address this well:

- **Similarity reports** (e.g., Turnitin originality reports) find copied text. They do not check whether a cited source exists or whether it supports the sentence that cites it.
- **AI-text detectors** flag writing, not source use, and cannot distinguish inappropriate AI use from acceptable or encouraged use.

SourceFidelity targets source use directly. It does not try to detect AI use, but checking whether sources exist and are used accurately can discourage inappropriate reliance on AI, save instructors time spent locating and reading sources, and give institutions more control over where student data goes.

---

## What it does

### Working now

- **Source identification, verification and retrieval** — the core feature. Each reference is parsed and searched across academic metadata services, book and public-domain catalogues, and web search. Located works are retrieved as full text where permitted. A reference that every search suited to its kind has failed to locate is reported as **Cannot be verified**; a failed or incomplete search is reported as incomplete, never as a negative result. A retrieved web page counts as full text only when it shows no sign of being cut off and its length fits the kind of source cited.
- **Citation–source relationship judgment** — runs automatically on every checked paper. Each citation is compared with evidence sentences selected from the retrieved source and labelled **Supports**, **Qualified or Mixed**, **Contradicts** or **Insufficient Evidence**; when the model cannot decide, the citation is labelled **LLM Undecided**. Every judgment links to the source sentences it is based on, so students can see how they used a source and instructors can see where a source may have been used incorrectly.
- **Academic-practice flags** — misquotation, patchwriting, secondary citation and related practices, with the student's wording shown above the source wording and copied words in bold.
- **APA citation and referencing style checks** — for example, incorrect title formatting, missing quotation locators, missing DOIs or URLs for references that should have them, parenthetical citations placed after a sentence's final punctuation, and references that are not cited in the paper.
- **Source upload** — in Personal deployments, users can upload sources the application could not retrieve.
- **Source storage that respects copyright** — verified open-access academic sources are stored permanently; all other sources are stored temporarily. Stored sources do not need to be searched for again, so later papers citing them process faster and cost less.
- **Reports** — a live report with source viewing and upload, and portable HTML/PDF exports.

### Planned

- **Cross-assessment comparison** and an **authorized student-paper repository**.
- **Wikipedia similarity scoring**.
- **Moodle plugin** — institutional sign-in, formative student self-checks with near-instant feedback, reports in the assessment marking area, batch reports that flag papers needing review, and upload of course texts and other course sources.
- **Library platform integration** — links to library holdings in reports, metadata from services the institution already licenses, and full-text access where database providers' terms permit. Each library capability is enabled separately.
- **Additional referencing systems** used in other disciplines.

---

## Data handling

Only two kinds of text from a student's paper leave the application:

- **Reference-list entries**, sent to academic metadata services, web search providers and LLM providers.
- **Citation passages**, sent to LLM providers together with excerpts of the cited source.

The rest of the paper, including the student's name and other identifying details, is not sent. Complete source files are not sent to LLM providers. Credentials stay inside the configured process and are never written to logs or reports.

## Security

- Uploaded and retrieved PDFs are scanned for viruses (ClamAV) before they are stored.
- Links are checked before they are fetched, and malicious links are blocked.
- Student papers, retrieved sources and model outputs are treated as untrusted input, with prompt-injection defences in every LLM prompt that contains them.

---

## Deployment

SourceFidelity has one shared core with two deployment profiles.

| | Personal | Institutional |
|---|---|---|
| Who | An individual instructor or student | A university or department |
| Where | Your own computer (or a rented server) | University servers |
| Integration | None required | Moodle, institutional sign-in, library platform (planned) |
| Main use | Checking individual papers | Formative student checking, deterrence, instructor review at scale |

Personal deployment works on its own but takes some setup, because the application relies on several external services:

- **Academic metadata and full text:** Crossref, OpenAlex, CORE, Semantic Scholar, DataCite, Unpaywall, Elsevier (optional)
- **Books and public-domain texts:** Google Books, Open Library, Internet Archive, Project Gutenberg, Wikisource
- **Web search:** Brave Search, Exa, Tavily, SearXNG (self-hosted, included)
- **LLMs:** DeepSeek, GLM (Z.ai); other providers can be configured behind the same interface

---

## Tech stack

- **Python 3.12**, FastAPI, Celery workers
- **PostgreSQL** (jobs, reports, evidence packages), **Redis** (task broker), **MinIO / S3-compatible storage** (papers and sources)
- **ClamAV** (virus scanning), **SearXNG** (self-hosted metasearch)
- **Alembic** (database migrations)
- **Docker Compose** (deployment)

---

## Quick start

Requirements: Docker with Docker Compose, and API keys for the services you want to use (see [Deployment](#deployment)).

**1. Clone and configure**

```bash
git clone https://github.com/sourcefidelity/sourcefidelity-api.git
cd sourcefidelity-api
cp .env.example .env
```

In `.env`:

- Set `POSTGRES_PASSWORD`, `MINIO_ROOT_PASSWORD` and `SEARXNG_SECRET`, and use the same database password in `DATABASE_URL`.
- Add `LLM_API_KEY` (DeepSeek), `OPENALEX_API_KEY`, `CORE_API_KEY` and any search keys (`BRAVE_SEARCH_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY`, `GOOGLE_BOOKS_API_KEY`).
- For relationship judgment, add `ZAI_API_KEY` (GLM) and set `ZAI_TERMS_VERIFIED_ON` to the date you reviewed the provider's data-use terms (`YYYY-MM-DD`). Judgment does not run until both are set.
- For access over plain `http://localhost`, set `REPORT_SESSION_COOKIE_SECURE=false`.

**2. Start the stack**

```bash
docker compose up -d --build
```

This starts the API, the paper and judgment workers, the scheduler, PostgreSQL, Redis, MinIO, ClamAV and SearXNG. ClamAV downloads its virus definitions on first start, which can take several minutes; the API waits for it.

**3. Create the database tables** (first run, and after each update)

```bash
docker compose exec api alembic upgrade head
```

**4. Check a paper**

- Health check: <http://localhost:8000/health/ready>
- Swagger UI: <http://localhost:8000/docs> — submit a PDF or DOCX to `POST /check/`, follow progress at `GET /status/{job_id}`, and open the report at `http://localhost:8000/report/{report_id}`.

The API binds to `127.0.0.1` only, and the default `REPORT_AUTH_MODE=personal_local` has no password. Use `personal_bearer` or an institutional sign-in for any shared deployment. See [`.env.example`](./.env.example) for all settings.

---

## Roadmap

- ✅ Reference extraction, source identification and retrieval
- ✅ In-text citation extraction (APA, MLA)
- ✅ Evidence packages and live/exported reports
- ✅ Citation–source relationship judgment
- ✅ Academic-practice flags and APA style checks
- 📝 Moodle plugin (`sourcefidelity-moodle`, GPLv3)
- 📝 Library platform integration
- 📝 Cross-assessment comparison, student-paper repository, Wikipedia similarity scoring
- 📝 Additional referencing systems

---

## License

MIT. See [`LICENSE`](./LICENSE).

The planned Moodle plugin (`sourcefidelity-moodle`) will be GPLv3 to match Moodle's licensing requirements.
