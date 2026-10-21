# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`ARCHITECTURE.md`**: accepted design decisions and rationale. Read the section for the component you are about to touch, and any contract it links.
- **`STATE.md`**: current implementation status, measured results and open limitations.
- **`ROADMAP.md`**: planned work, priorities and open questions.
- **`PLAN.md`**: product scope, responsibility boundary and risk register.
- **`CONTEXT.md`** at the repo root, if it exists: the domain glossary.

These are private, git-ignored guiding documents. AGENTS.md defines their roles and maintenance rules; follow those rules when updating them.

## File structure

Single-context repo. This repo does not use `docs/adr/`.

```
/
├── AGENTS.md
├── ARCHITECTURE.md   ← decisions (in place of ADRs)
├── STATE.md
├── ROADMAP.md
├── PLAN.md
├── CONTEXT.md        ← glossary only (created lazily)
└── app/
```

## Recording decisions

When `/domain-modeling` or another skill would write an ADR, record the decision in the relevant `ARCHITECTURE.md` section instead, maintaining current guidance in place. Never create `docs/adr/`.

## Use the glossary's vocabulary

`CONTEXT.md` holds domain terms only, never decisions or status. `/domain-modeling` creates it lazily; if it does not exist, proceed silently.

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md` and the product vocabulary in AGENTS.md (for example `unlocated_after_search` versus `search_incomplete`). Don't drift to synonyms.

If the concept you need isn't defined yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag conflicts

If your output contradicts `ARCHITECTURE.md` or a recorded owner decision, surface it explicitly rather than silently overriding:

> _Contradicts ARCHITECTURE §5 (Evidence Package retrieval), but worth reopening because…_

If the guiding documents and the code disagree, report the discrepancy rather than treating either as correct.
