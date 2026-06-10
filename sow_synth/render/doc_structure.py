"""Document structure planning for client history.

Defines the closed vocabulary of section types and the Pydantic schema the
LLM fills in during Phase 1 of client history generation.

The LLM decides:
  - which sections to include (net_worth_table, compliance_notes are optional)
  - the order of sections
  - the style depth (detailed / concise) for each prose section
  - the overall tone of the document

The code enforces hard constraints afterwards:
  - client_profile must be present (inserted at position 0 if absent)
  - exactly one sow_claim per claim_id (missing ones are appended; extras dropped)
  - at least one prose section must exist
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator


class SectionType(str, Enum):
    client_profile     = "client_profile"      # always tabular — code-rendered
    introduction       = "introduction"        # LLM prose
    sow_claim          = "sow_claim"           # one per claim — LLM prose
    net_worth_table    = "net_worth_table"     # tabular — code-rendered (optional)
    financial_overview = "financial_overview"  # LLM prose (optional)
    compliance_notes   = "compliance_notes"    # LLM prose (optional, rare)
    rm_assessment      = "rm_assessment"       # LLM prose


# Sections the code renders — no LLM call needed
CODE_RENDERED: frozenset[SectionType] = frozenset({
    SectionType.client_profile,
    SectionType.net_worth_table,
})

# Sections that must appear exactly once (before constraint enforcement)
REQUIRED_ONCE: frozenset[SectionType] = frozenset({
    SectionType.client_profile,
    SectionType.rm_assessment,
})


class SectionSpec(BaseModel):
    section_type: SectionType
    claim_id: str | None = None   # set only for sow_claim sections
    style: Literal["detailed", "concise"] = "detailed"


class DocumentStructurePlan(BaseModel):
    """Returned by the LLM — the structural plan for one client history document."""
    sections: list[SectionSpec]
    overall_tone: Literal["formal", "narrative", "technical"] = "formal"


# ---------------------------------------------------------------------------
# Constraint enforcement
# ---------------------------------------------------------------------------

def enforce_constraints(
    plan: DocumentStructurePlan,
    claim_ids: list[str],
) -> DocumentStructurePlan:
    """Guarantee required sections are present and claim coverage is complete.

    Returns a new DocumentStructurePlan (does not mutate the input).
    """
    sections = list(plan.sections)

    # --- client_profile: must be present, ideally first ---
    if not any(s.section_type == SectionType.client_profile for s in sections):
        sections.insert(0, SectionSpec(section_type=SectionType.client_profile))

    # --- sow_claim: exactly one per claim_id ---
    covered = {s.claim_id for s in sections if s.section_type == SectionType.sow_claim}
    # remove duplicate claim sections (keep first occurrence)
    seen_claims: set[str] = set()
    deduped = []
    for s in sections:
        if s.section_type == SectionType.sow_claim:
            if s.claim_id in seen_claims:
                continue
            seen_claims.add(s.claim_id)
        deduped.append(s)
    sections = deduped

    # append any missing claims
    for cid in claim_ids:
        if cid not in seen_claims:
            sections.append(SectionSpec(
                section_type=SectionType.sow_claim,
                claim_id=cid,
                style="detailed",
            ))

    # drop sow_claim entries whose claim_id is not in our set
    valid_cids = set(claim_ids)
    sections = [
        s for s in sections
        if s.section_type != SectionType.sow_claim or s.claim_id in valid_cids
    ]

    # --- rm_assessment: must be present (append if absent) ---
    if not any(s.section_type == SectionType.rm_assessment for s in sections):
        sections.append(SectionSpec(section_type=SectionType.rm_assessment))

    return DocumentStructurePlan(sections=sections, overall_tone=plan.overall_tone)


# ---------------------------------------------------------------------------
# Planning prompt
# ---------------------------------------------------------------------------

def build_plan_prompt(
    profile_summary: str,
    claim_summaries: list[str],
    n_pages_hint: int,
) -> str:
    claims_block = "\n".join(f"  - {c}" for c in claim_summaries)
    section_menu = "\n".join(
        f"  {t.value}" for t in SectionType
    )
    return (
        f"You are structuring a source-of-wealth client history document for a UK "
        f"private bank. Choose which sections to include, in what order, and at what "
        f"depth. Vary the structure — do not default to a fixed template.\n\n"
        f"Client summary:\n{profile_summary}\n\n"
        f"Claims to cover (one sow_claim section per entry — all required):\n"
        f"{claims_block}\n\n"
        f"Available section types:\n{section_menu}\n\n"
        f"Rules:\n"
        f"  - Include client_profile exactly once (position is your choice)\n"
        f"  - Include one sow_claim per claim listed above\n"
        f"  - Include rm_assessment exactly once\n"
        f"  - net_worth_table and compliance_notes are optional — include them "
        f"only when they add value\n"
        f"  - introduction and financial_overview are optional\n"
        f"  - Target approximately {n_pages_hint} page(s) of content\n"
        f"  - Choose overall_tone: formal / narrative / technical\n\n"
        f"Return the plan as structured output."
    )
