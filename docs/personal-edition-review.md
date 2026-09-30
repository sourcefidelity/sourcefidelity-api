# Personal edition review

This optional Personal-only workflow records an owner's Yes/No assessment of a
book copy against an explicitly supplied work and edition. It does not admit
the source, change a student report, establish unchanged text or authorize
quotation, paraphrase or locator checks. Institutional deployments do not
require a human edition-review role.

## Start and decide

Open `/edition-reviews/new?representation_id=<source-representation-UUID>` in an
authorized Personal session and supply the reference to compare. This interface
does not automatically bind the reference to a student report.

The current v3 viewer asks one question and shows up to six front-matter page
previews, with full-size links:

| Option | Select when |
|---|---|
| Yes | The displayed pages establish the cited work and edition; a later printing of that edition counts. |
| No | The displayed pages do not establish that match, including when the evidence is insufficient or unclear. The result remains unverified, not a finding that the source is wrong. |

No notes, page-number selections or acknowledgment checkbox are required.
Translations and abridgments need separate retrieval, not equivalent-original
approval. Detected signals block confirmation; absence of a detected signal
is not a comprehensive guarantee.

The server records the exact reference, source/page bindings, authenticated
reviewer and time in a separate binary-answer receipt. Each snapshot accepts
one decision; reconsideration creates a new snapshot, not an edited answer.
Export distinguishes pending from completed decisions. No report/admission
consumer automatically treats the answer as source equivalence.

## Existing older reviews

Saved v1/v2 reviews keep their original questions, required evidence fields,
options, hashes and detailed records. Those requirements apply only when
opening an older snapshot, not to new v3 reviews. A v3 binary answer cannot
be converted into a detailed legacy attestation. All versions preserve their
independent admission and task-use restrictions.

## Access and lifecycle

- The Personal review capability and source-content capability are both required.
- Only same-scope, non-expired, non-deleted original book PDFs in `accepted` or
  `needs_review` state are eligible. OCR derivatives are not supported here.
- Stored cleanliness must be clean, and current source bytes must pass the
  existing structural and malware inspection before any page is rendered.
- Missing, changed, withdrawn or expired sources invalidate access, submission
  and export. Inspection failure never grants permission.
- Pages are rendered at 120 DPI, at most six pages and eight million pixels per
  page, with no OCR. A single-page request renders only its requested page.
- The review is bounded front-matter inspection, not proof of complete-source
  identity or equivalence. Insufficient or unreadable pages require uncertainty.
- Evidence hashes in this workflow identify exact PNG page renders, not extracted
  quotations; the snapshot retains renderer/version and page-index bindings.
- Snapshots and decisions share the source's lifecycle: foreign-key cascading
  deletion removes them with the source representation. No extra PDF or PNG
  copies are persisted by the workflow. Immutable decisions mean no ordinary
  editing, not indefinite retention after authorized deletion.

The application uses same-origin mutation checks, escaped text, restricted
content-security policy and no-store responses. The new migration is
`a1b7c3d8e425`; applying it and restarting the running application are deployment
operations, separate from workspace implementation and tests.
