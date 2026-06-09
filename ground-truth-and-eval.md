# Ground truth & evaluation

## The agent's task

The SoW agent under evaluation performs four sub-tasks:

1. **Extract** each claim from the client history document.
2. **Classify** each claim into a SoW type (`employment`, `gift`, `inheritance`,
   `business_profits`).
3. **Find** the corroboration document(s) that support each claim.
4. **Flag for human review** any claim that is not sufficiently corroborated.

## Sub-task → ground-truth mapping

| Sub-task | Scored against | Source in the graph |
|----------|----------------|---------------------|
| Extraction | the set of `states` edges (and ideally each claim's text span in the client history) | `ClientHistory --states--> Claim`, `Claim.asserted_text` |
| Classification | `Claim.sow_type` label | node attribute |
| Finding corroboration | the `corroborates` edges (document↔claim mapping) | `Document --corroborates--> Claim` |
| Flag for review | a per-claim **sufficiency verdict** | computed (see below) |

The agent sees and is scored on `states`, `corroborates`, `sow_type`, and the
sufficiency verdict. `covers` and `derived_from` are internal scaffolding (see
`docs/data-model.md`).

## Sufficiency rule (OPEN DECISION — pick before scaling)

A claim's "sufficiently corroborated" label is computed from: the claimed amount (via
`covers`), the evidenced amount (sum over present, non-contradictory `corroborates`
documents), document-type adequacy, and temporal validity. Three candidate definitions,
increasing in realism and complexity:

**Option A — strict binary coverage.**
Sufficient iff evidenced amount ≥ claimed amount AND no contradictory document AND all
supporting documents are temporally valid. Any shortfall, contradiction, or temporal
break → flag. Simplest to implement and label; good for v1.

**Option B — graded coverage with threshold.**
Compute a corroboration score in `[0, 1]` = evidenced ÷ claimed, then flag if score
< threshold (e.g. 0.9) OR a contradiction/temporal break exists. Allows partial
corroboration and a tunable bar; the score is also a useful per-claim eval signal.

**Option C — analyst-style multi-factor.**
Sufficiency is a function of coverage ratio, **document-type adequacy** (a single
payslip is not enough to evidence a 2M employment claim — you'd expect a contract plus
bank credits), recency/temporal validity, and contradiction. Most realistic, closest to
how a human analyst decides, but requires an explicit adequacy matrix per SoW type and
doc type.

**Recommendation:** start with Option A for v1 to get a working, unambiguous label, and
graduate to B or C once the agent's flagging behaviour needs finer discrimination.
Whatever is chosen, document it precisely and version it — it is part of the ground-truth
contract, and changing it silently re-labels every sample. Open sub-questions to answer:
does one contradictory document among several good ones flag the whole claim? Is partial
coverage "flag" or "sufficient with note"? How does temporal impossibility interact with
otherwise-full coverage?

## Perturbation taxonomy (stage 7)

Each perturbation is a labeled graph mutation applied before rendering, producing a
ground-truth annotation. Designed so the eval set exercises every failure mode the agent
must catch.

| Perturbation | Graph mutation | Sub-task stressed | Expected agent behaviour |
|--------------|----------------|-------------------|--------------------------|
| Missing corroboration | drop document + its `corroborates` edge | finding / flagging | flag for review |
| Partial corroboration | document evidences only part of the claimed amount | flagging | flag (under Option A) or score < 1 (B/C) |
| Contradictory document | mutate document amount; relabel edge `contradictory` | finding / flagging | flag; not treat as support |
| Temporal impossibility | shift a document/event date (e.g. will before death) | flagging | flag |
| Red herring | add a decoy document with no real `corroborates` edge | finding | do **not** link it to the claim |
| Unreconciled net worth | net worth inconsistent with stated flows | (system-level) | flag / surface inconsistency |
| Misclassification bait | claim phrased to look like a different SoW type | classification | classify by substance, not surface |

The difficulty profile in the `ScenarioSpec` controls which perturbations appear and how
many, so you can generate clean sets, mildly-hard sets, and adversarial sets on demand.

## Ground-truth bundle (stage 11 output)

Each generated sample is emitted as a bundle:

```
sample_<id>/
├── input/                     # what the agent receives
│   └── *.ocr.json             # noisy OCR docs (client history + corroboration)
├── clean/                     # clean rendered docs (never shown to agent)
│   └── *.ocr.json
├── ground_truth.json          # the frozen graph projected to GT:
│   ├── claims[]               #   id, sow_type, amount, asserted_text, text span
│   ├── corroborates[]         #   claim_id ↔ doc_id mapping
│   ├── net_worth              #   computed, with as-of date
│   ├── sufficiency[]          #   per-claim verdict (+ score if Option B/C)
│   └── perturbations[]        #   labels applied to this sample
└── meta.json                  # ScenarioSpec + seed (for exact reproduction)
```

## Eval harness (`eval/`)

Consumes bundles and scores agent output against `ground_truth.json`. Suggested metrics:
- **Extraction:** precision/recall over claims (with span overlap or amount tolerance
  per the open decision on extraction scoring).
- **Classification:** accuracy / per-class F1 over `sow_type`.
- **Corroboration finding:** precision/recall/F1 over the claim↔doc mapping; report
  red-herring false-positive rate separately.
- **Flagging:** precision/recall on the flag decision against the sufficiency verdict;
  report per-perturbation-type breakdown so you can see *which* failure modes the agent
  misses.
