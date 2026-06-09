# Roadmap

Validate the highest-risk part (trustworthy ground truth) before investing in rendering.
Each phase has concrete deliverables and acceptance criteria.

## Phase 0 — Scaffolding
**Deliverables:** repo structure (see `CLAUDE.md`), `models.py` with Pydantic v2 schemas
for `Profile`, `Event`, `Claim`, `Document`, edges, `ScenarioSpec`; CI running tests.
**Acceptance:** schemas import cleanly; a `ScenarioSpec` can be constructed and round-trips
through JSON.

## Phase 1 — Fact core (highest priority)
**Deliverables:** `events.py` (typed dated event generation), `ledger.py` (deterministic
fold → net worth), `claims.py` (projections), `graph.py` (assemble + freeze), plus
Hypothesis property tests.
**Acceptance (invariants hold across thousands of seeds):**
- `net_worth = Σ inflows − Σ outflows` for every generated timeline.
- `Claim.amount == Σ amounts of covered events` for every claim.
- Causal ordering never violated (no distribution before founding, etc.).
- `(spec, seed)` reproduces an identical fact graph bit-for-bit.
- A target net-worth band is reliably hit (via nudging or rejection sampling).

**Outcome:** trustworthy ground truth with *no documents yet*. If this is solid,
everything downstream is "just" rendering.

## Phase 2 — One document type end-to-end
**Deliverables:** `docplan.py` for a single doc type (e.g. payslip), one Jinja2 template,
`render/realize.py` (constrained LLM surface realization), `verify.py`
(extract-and-assert + repair), `ocr.py` (emit target schema + programmatic noise).
**Acceptance:**
- Every figure in the rendered payslip traces to an `Event` (verify passes, or repairs).
- The clean copy is retained; noise is applied only to a copy.
- The OCR JSON matches the confirmed target engine schema.
- Round-trip: generate → render → verify → noise produces a valid bundle for one claim.

## Phase 3 — Full document matrix + perturbations
**Deliverables:** `docplan.py` requirements matrix for all SoW types and doc types;
templates for each; `perturb.py` implementing the taxonomy in
`docs/ground-truth-and-eval.md`.
**Acceptance:**
- Each SoW type produces its expected corroboration document set.
- Each perturbation produces the correct ground-truth label and the intended document
  artifact (e.g. a contradictory doc actually contradicts; a dropped doc is absent).
- Clean (no-perturbation) samples are fully corroborated by construction.

## Phase 4 — Packaging + eval harness + scale
**Deliverables:** `package.py` (bundle format), `eval/` harness, a spec sampler to
generate a *distribution* of specs, batch generation with cost tracking.
**Acceptance:**
- Bundles match the format in `docs/ground-truth-and-eval.md`.
- The harness scores a baseline agent and reports per-sub-task and per-perturbation
  metrics.
- A full eval set (clean + graded-hard + adversarial) can be generated reproducibly
  within the LLM-call budget.

## Cross-cutting (every phase)
- Hypothesis property tests guard the invariants; never let them regress.
- Schema versioning: bump a version when a generator change alters output, so old eval
  sets aren't silently invalidated.
- Prefer templating over generation to control cost and drift; only use the LLM for
  natural-language flavor.
