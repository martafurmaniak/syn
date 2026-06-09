# Architecture

## Why this design (what we're replacing)

The original draft was a 7-step pipeline where amounts and claims flowed *into* an LLM
and realistic documents flowed *out*: an LLM "embedded" specific figures into prose and
was relied on to keep arithmetic consistent across a multi-page document, and a final
step asked an LLM to "introduce OCR noise." The original steps were:

1. Load a CSV (name, DOB, occupation), pick a random client row.
2. LLM generates a client storyline + how many claims per SoW type.
3. Generate financial amounts; sample net worth as `Uniform(1, Σ claim amounts)`;
   derive outflows from the gap.
4. LLM generates claims with amounts embedded.
5. LLM generates corroboration documents per claim.
6. LLM generates the multi-page client history embedding claims, net worth, outflows.
7. LLM adds OCR noise to the client history JSON.

Problems this caused:

- **The LLM was the source of truth.** LLMs are unreliable at faithfully copying exact
  figures and preserving an accounting identity across pages, so the structured ground
  truth and the rendered documents would silently drift apart, with no way to know
  which samples were corrupted.
- **The financial model sampled the answer and back-solved.** `Uniform(1, Σinflows)`
  frequently implies the client consumed almost all lifetime inflows; there was no time
  dimension, no asset appreciation (so net worth could only ever be *less* than
  inflows — the common "inherited 2M, invested, now worth 6M" story was
  unrepresentable), no taxation, and outflows were untyped random amounts that can't be
  described or corroborated coherently.
- **Only the easy case was generated.** Every sample was internally consistent and
  fully corroborated, but the agent's whole job is to catch gaps, contradictions, and
  insufficient corroboration. An eval set with no negative cases overstates agent
  performance and can't discriminate good agents from bad ones.
- **OCR schema mismatch and LLM noise.** "OCR-like JSON" must match the real downstream
  engine's schema or you evaluate on a format you'll never see. Asking an LLM to add
  noise is an uncontrolled transform that can turn a 7 into a 1 and corrupt ground truth.

## Core principles

1. **Invert the relationship: code owns truth, the LLM renders.** A canonical fact
   layer holds every number, date, and linkage. Documents are downstream renderings.
   The LLM only does surface realization.
2. **Sample the causes, not the answer.** Build a dated, typed event ledger and
   *compute* net worth. The accounting identity holds by construction and cannot be
   violated.
3. **Ground truth by construction.** Claim↔document links are recorded when documents
   are planned from known events, not reconstructed afterward.
4. **Difficulty is a first-class, labeled output.** Negative and hard cases (missing,
   partial, contradictory, temporally impossible, red-herring) are deliberately injected
   graph mutations with the gap recorded in ground truth.
5. **Validation as a safety net.** After rendering, extract every figure and assert it
   exists in the fact layer; repair on mismatch. This catches any LLM drift so the
   dataset can be trusted at scale.
6. **Reproducibility.** A sample is a pure function of `(ScenarioSpec, seed)`. Property
   tests assert invariants across thousands of seeds.

## The pipeline (stages 0–11)

### Stage 0 — Scenario spec
A declarative `ScenarioSpec` (Pydantic) is the one human/sampler-written input: target
net-worth band, profile constraints, SoW composition (claims per type), narrative hooks
(e.g. "at least one private-equity link"), currency mode, and a *difficulty profile*
(which perturbations, how many). Everything downstream is a pure function of
`(spec, seed)`. Build a *distribution* of specs to cover the eval space.

### Stage 1 — Profile resolution
Load from CSV or synthesize, then enrich: DOB → age → plausible career length;
nationality/domicile → jurisdiction (tax rules, inheritance law, document types,
currency); industry → plausible income scale. Profile and spec co-constrain (don't pair
a 25-year-old with a 40-year career story); reject-and-resample on conflict.

### Stage 2 — Event timeline (code, no LLM)
The heart of the system. Sample typed, dated events from a generative model of a life:
career arc (roles with start/end and salary progression), windfalls (inheritance dated
near a parent's plausible death, gifts received), business activity (founding → profit
distributions → optional exit), investment contributions/returns, and outflow events
(property purchase, tax, gifts given, spending, losses). Magnitudes from realistic
distributions (log-normal for incomes/wealth). Enforce causal ordering (no distribution
before founding; no property purchase without prior liquidity).

### Stage 3 — Ledger evaluation
Deterministically fold chronologically-sorted events into a balance trajectory; closing
net worth falls out by construction, so `NW = Σinflows − Σoutflows` cannot be violated.
Tax is a derived event, not a free parameter. If closing net worth misses the target
band, nudge a free knob (investment return, spending rate) and re-fold, or reject and
resample — but never abandon the ledger as truth.

### Stage 4 — Claim projection
A claim is a typed, labeled aggregation over a subset of events
(`employment claim = Σ salary events for role X`). Defined as code projections, so
amounts are exact and each claim carries provenance to the event IDs it `covers`.

### Stage 5 — Document planning
One `DocumentPlan` is created per `Claim`. The document format is drawn from a
**weighted format registry** keyed on `SowType`, so different seeds produce different
document types for the same claim type — for example, an employment claim may be
corroborated by a payslip, a bank statement, an employer letter, an email thread, or
a bloomberg article extract, in proportion to realistic frequencies.

Format selection is two-dimensional: **EvidenceType** (what fact — employment,
inheritance, gift, business) × **FormatType** (how it looks — structured statement,
legal document, correspondence, press extract). Sixteen format types are registered
across four categories.

**The `corroborates` edges are created here, by construction** — before the graph is
frozen, before any rendering — so the ground-truth mapping is established at planning
time and is never reconstructed from rendered content.

Each `DocumentPlan` also carries `verify_hints` — a list of `{key, expected, precision}`
dicts that `verify.py` uses to check rendered content against the fact layer with
precision-aware tolerance (`exact` / `rounded` / `approximate` / `narrative`).

### Stage 6 — Graph assembly
Assemble profile, events, claims, and documents into a typed graph (NetworkX): nodes per
entity, edges `states` / `covers` / `corroborates` / `derived_from`. This graph is the
canonical ground truth. **Freeze it.** Everything after is rendering.

### Stage 7 — Perturbation injection
Apply *labeled* graph mutations before rendering to manufacture negative/hard cases:
drop a document (missing corroboration), mutate a doc's amount (contradiction), shift a
date (temporal impossibility), inject a decoy document (red herring), leave a claim
partially covered. Each is logged as a ground-truth annotation. See
`docs/ground-truth-and-eval.md` for the taxonomy.

### Stage 8 — Surface realization (template-driven; LLM stubbed)
Currently fully template-driven: Jinja2 templates per format type, with all numbers
injected from `DocumentPlan.template_context` (which comes from the fact layer). The
LLM is not called in the current implementation — all natural-language flavor text is
generated deterministically from `formats/flavor.py` using hash-seeded generators.

When the LLM is enabled, it will fill only natural-language fields (headlines, bylines,
prose paragraphs) and must return a validated schema; it never decides a number, date,
or entity. Use structured outputs / constrained decoding so the model output is
schema-valid.

The dispatcher in `formats/realize.py` routes each plan to the correct context builder
and template family based on `format_type`. Page count varies by document complexity
(payslips: 1–4 pages depending on gross pay; other types: 1–2 pages).

### Stage 9 — Verify-and-repair
Each `DocumentPlan` carries `verify_hints` — a list of `{key, expected, precision}`
records — that `verify.py` evaluates against the rendered document's `key_values`.
Precision modes: `exact` (must match to the penny), `rounded` (within ±1 unit),
`approximate` (within 10%), `narrative` (amount is in prose — skip numeric check).
On mismatch, the offending `key_value` is overwritten in-place. The clean rendered
copy is always retained.

### Stage 10 — OCR layout + noise
Convert clean documents into the target OCR engine's actual schema (pages, lines, words,
polygons, confidence, key-value/table blocks), then apply *programmatic*, parameterized
noise (character-confusion matrices, bbox jitter, confidence degradation, segmentation
errors). The noisy copy is model input only; ground truth and evaluation reference the
clean copy.

### Stage 11 — Packaging
Emit the sample bundle: noisy OCR docs (agent input), clean docs, the ground-truth graph
(events, claims, claim↔doc map, net worth, perturbation labels), and the `spec`+`seed`
for reproduction. Version schemas so a generator change can't silently invalidate an old
eval set.

## Build order

Validate the core bet before building all eleven stages. See `docs/roadmap.md` for the
phased plan and acceptance criteria. In short: fact core (stages 2–3, 6) with property
tests first; then one document type end-to-end (5, 8, 9, 10); then scale to the full
document matrix and perturbation taxonomy; then packaging and the eval harness.
