# Synthetic Source-of-Wealth (SoW) Data Generator

Generates realistic, internally-consistent synthetic SoW case files — a client
history document, claims about the client's source of wealth, and corroboration
documents — together with trustworthy ground truth, for developing and evaluating
SoW analysis agents.

## Start here

1. Read [`CLAUDE.md`](./CLAUDE.md) — project orientation and the one core principle.
2. Then the design docs:
   - [`architecture.md`](./architecture.md) — why the design, and the full pipeline.
   - [`data-model.md`](./data-model.md) — nodes, edges, ledger, schemas.
   - [`ground-truth-and-eval.md`](./ground-truth-and-eval.md) — agent task, sufficiency, perturbations, bundle format.
   - [`roadmap.md`](./roadmap.md) — phased build plan and acceptance criteria.
   - [`open-questions.md`](./open-questions.md) — decisions to confirm before scaling.

## Status

Phases 0–3 complete. The full pipeline runs end-to-end — fact core, document
planning with format diversity, rendering, verify-and-repair, OCR noise, and HTML
export. Phase 4 (packaging, eval harness, batch generation) is next.

### What works today

```
python generate_sample.py --seed 42 --employment 2 --inheritance 1 --gift 1 --business 1
python generate_sample.py --seed 7 --export-dir ./output_sample
python -m pytest tests/ -v
```

- **22 property-based and unit tests** pass across hundreds of seeds.
- **5 document types per run** (varies by seed): payslip, bank statement, employer
  letter, probate grant, will extract, gift letter, gift deed, solicitor letter,
  email thread, bloomberg article, company accounts, distribution statement,
  companies house filing, bank transfer confirmation, and more.
- **Format diversity**: employment claims may be evidenced by a payslip, bank
  statement, employer letter, email thread, or bloomberg article — chosen
  deterministically per seed from a weighted registry.
- **Verify-and-repair** checks every rendered primary amount against the fact layer
  with precision-aware tolerance (exact / rounded / approximate / narrative).
- **HTML export** (`--export-dir`) writes one scrollable HTML per document for
  visual inspection, plus a JSON bundle per document and an index page.

## The core idea in one sentence

Code owns all truth (an event-sourced ledger and a frozen fact graph); documents are
renderings of that truth; templates inject numbers from the fact layer and the LLM
(when added) only fills natural-language flavor fields.
