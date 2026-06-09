# Open questions / decisions to confirm

These were surfaced during design and should be resolved before generating at scale.
Each can silently re-label or invalidate data if changed late, so pin them down early.
Recommended defaults let implementation start without blocking.

## 1. Sufficiency rule (highest priority)
**Why it matters:** it defines the flag-for-review ground-truth label — the agent's
fourth sub-task — and is part of the ground-truth contract.
**Options:** A (strict binary coverage), B (graded score + threshold), C (analyst-style
multi-factor with document-type adequacy). Detail in `docs/ground-truth-and-eval.md`.
**Sub-decisions:** does one contradictory doc flag the whole claim? Is partial coverage
"flag" or "sufficient with note"? How does temporal impossibility interact with full
coverage?
**Recommended default:** start with **Option A** for an unambiguous v1 label; graduate to
B/C when finer flagging discrimination is needed.

## 2. Target OCR schema
**Why it matters:** agents must see the same JSON format in eval as in production, or
results don't transfer. Drives the stage-10 emitter.
**Recommended default:** **Azure Document Intelligence** (consistent with the Azure
OpenAI stack), unless the production pipeline uses Textract or Google Document AI —
confirm which engine the downstream agents actually consume.

## 3. Currency ✓ RESOLVED (single currency)
Single currency is implemented. `ScenarioSpec.currency` defaults to `"GBP"`. All
events, claims, and documents use the same currency. Multi-currency remains a future
option but is not needed for Phase 4.

## 4. Extraction scoring
**Why it matters:** determines how the extraction sub-task is graded and therefore what
the client-history ground truth must record (exact text vs. span vs. amount).
**Options:** exact-match on claim text; span-overlap (IoU) against `asserted_text`
location; amount-tolerance matching.
**Recommended default:** record both `asserted_text` and its character span so any of
these metrics can be computed later; decide the headline metric with the eval harness.

## 5. Scale & LLM-call budget per sample ✓ RESOLVED FOR CURRENT PHASE (zero calls)
Phase 3 is fully template-driven — no LLM calls. All flavor text (names, addresses,
article prose, letter bodies) is generated deterministically via hash-seeded generators
in `formats/flavor.py`. When the LLM is enabled in a future phase, it should fill only
natural-language flavor fields and be constrained to schema-valid output.

## 6. Time dimension (confirmed)
Events are dated and the ledger is temporal; net worth is a point-in-time snapshot at an
"as of" date. This is assumed throughout and is needed for temporal-impossibility
perturbations. Flagged here only so it stays an explicit, revisitable assumption.

## 7. Profile source ✓ RESOLVED (synthesis only for now)
`profile.py` fully synthesizes profiles from `ScenarioSpec` constraints and the rng
(name, DOB, occupation, industry, domicile, age, career start). CSV-driven profiles
can be added behind the same `resolve_profile(spec, rng)` interface when needed.
