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

## 3. Currency
**Why it matters:** multi-currency adds realism but requires per-event FX dates in the
ledger fold and FX handling in documents.
**Recommended default:** **single currency** for v1 (set in `ScenarioSpec.currency_mode`);
add multi-currency in a later phase once the single-currency ledger is proven.

## 4. Extraction scoring
**Why it matters:** determines how the extraction sub-task is graded and therefore what
the client-history ground truth must record (exact text vs. span vs. amount).
**Options:** exact-match on claim text; span-overlap (IoU) against `asserted_text`
location; amount-tolerance matching.
**Recommended default:** record both `asserted_text` and its character span so any of
these metrics can be computed later; decide the headline metric with the eval harness.

## 5. Scale & LLM-call budget per sample
**Why it matters:** drives how much is templated (cheap, deterministic) vs. generated
(expensive, needs verify-and-repair), and whether to recommend batching.
**Recommended default:** maximize templating; reserve LLM calls for natural-language
flavor and the narrative client-history document. Set a per-sample call budget in config
and track it in batch generation.

## 6. Time dimension (confirmed)
Events are dated and the ledger is temporal; net worth is a point-in-time snapshot at an
"as of" date. This is assumed throughout and is needed for temporal-impossibility
perturbations. Flagged here only so it stays an explicit, revisitable assumption.

## 7. Profile source
**Why it matters:** the original draft loads a CSV of (name, DOB, occupation).
**Open:** keep CSV-driven profiles, fully synthesize them, or both? Enrichment
(nationality, domicile, industry, age) is needed either way.
**Recommended default:** support both — CSV when provided, synthesis otherwise — behind
a single `profile.py` interface.
