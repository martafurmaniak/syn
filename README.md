# Synthetic Source-of-Wealth (SoW) Data Generator

Generates realistic, internally-consistent synthetic SoW case files — a client
history document, claims about the client's source of wealth, and corroboration
documents — together with trustworthy ground truth, for developing and evaluating
SoW analysis agents.

## Start here

1. Read [`CLAUDE.md`](./CLAUDE.md) — project orientation and the one core principle.
2. Then the `docs/`:
   - [`docs/architecture.md`](./docs/architecture.md) — why the redesign, and the full pipeline.
   - [`docs/data-model.md`](./docs/data-model.md) — nodes, edges, ledger, schemas.
   - [`docs/ground-truth-and-eval.md`](./docs/ground-truth-and-eval.md) — agent task, sufficiency, perturbations, bundle format.
   - [`docs/roadmap.md`](./docs/roadmap.md) — phased build plan and acceptance criteria.
   - [`docs/open-questions.md`](./docs/open-questions.md) — decisions to confirm before scaling.

## Status

Design complete; implementation not yet started. The documents in this repo capture
the full design rationale so work can begin directly.

## The core idea in one sentence

Code owns all truth (an event-sourced ledger and a frozen fact graph); documents are
renderings of that truth; the LLM only does surface realization and never decides a
number, date, or linkage.
