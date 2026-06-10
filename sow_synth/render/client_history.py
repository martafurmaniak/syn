"""Stage 8 (LLM path) — Client history document generation.

Two-phase generation:

  Phase 1 — Structure planning (1 LLM call)
    The LLM returns a DocumentStructurePlan: an ordered list of SectionSpecs
    that determines which sections appear, in what order, and at what depth.
    Code enforces hard constraints (client_profile present, all claims covered,
    rm_assessment present) and then accepts the plan.

  Phase 2 — Section rendering (1 LLM call per prose section)
    Code-rendered sections (client_profile, net_worth_table) are built entirely
    from the fact layer — no LLM involved.
    Prose sections (introduction, sow_claim, financial_overview,
    compliance_notes, rm_assessment) each get one LLM call with all facts
    injected; the model writes natural language only.

  Phase 3 — Pagination (code only)
    Rendered section strings are accumulated into pages by character budget.
    Sections are never split mid-section. Page count varies naturally with
    section selection and style depth.

Entry point:
    render_client_history(profile, events, claims, spec, llm, telemetry)
    → (Document, updated_claims)
"""
from __future__ import annotations

import textwrap
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from sow_synth.formats.realize import _fmt
from sow_synth.ledger import compute_net_worth
from sow_synth.models import (
    Claim, Document, DocType, Event, OcrPage, Profile, SowType,
)
from sow_synth.render.doc_structure import (
    CODE_RENDERED, DocumentStructurePlan, SectionSpec, SectionType,
    build_plan_prompt, enforce_constraints,
)
from sow_synth.spec import ScenarioSpec

if TYPE_CHECKING:
    from sow_synth.llm import LlmClient
    from sow_synth.telemetry import Telemetry

_WRAP = 90
_PAGE_CHAR_BUDGET = 3_000   # soft limit: start a new page when exceeded

_SYSTEM = (
    "You are a relationship manager at a UK private bank writing a source-of-wealth "
    "assessment report. Write concisely and professionally. Use only the facts "
    "provided — do not invent figures, dates, names, or events. Do not add "
    "disclaimers about verification. Do not use bullet points."
)

# ---------------------------------------------------------------------------
# LLM response schemas — prose only
# ---------------------------------------------------------------------------

class _Prose(BaseModel):
    paragraph: str


# ---------------------------------------------------------------------------
# Code-rendered section builders (no LLM)
# ---------------------------------------------------------------------------

_RULE = "=" * 72

def _section_header(title: str) -> str:
    return f"\n{_RULE}\n{title}\n{_RULE}\n"


def _render_client_profile(
    doc_id: str,
    as_of_str: str,
    currency: str,
    profile: Profile,
) -> str:
    lines = [
        f"CLIENT SOURCE-OF-WEALTH ASSESSMENT",
        _RULE,
        f"Reference:    {doc_id}",
        f"Prepared:     {as_of_str}",
        f"Currency:     {currency}",
        _RULE,
        "",
        "CLIENT PROFILE",
        "",
        f"  Full Name:      {profile.name}",
        f"  Date of Birth:  {profile.date_of_birth.strftime('%d %B %Y')}",
        f"  Age:            {profile.age}",
        f"  Occupation:     {profile.occupation}",
        f"  Nationality:    {profile.nationality}",
        f"  Domicile:       {profile.domicile}",
        f"  Industry:       {(profile.industry or 'Finance').title()}",
        "",
    ]
    return "\n".join(lines)


def _render_net_worth_table(
    as_of_str: str,
    currency: str,
    net_worth: Decimal,
    events: list[Event],
) -> str:
    from sow_synth.models import INFLOW_TYPES

    in_totals: dict[str, Decimal] = {}
    out_totals: dict[str, Decimal] = {}
    for e in events:
        label = e.type.value.replace("_", " ").title()
        if e.type in INFLOW_TYPES:
            in_totals[label] = in_totals.get(label, Decimal("0")) + e.amount
        else:
            out_totals[label] = out_totals.get(label, Decimal("0")) + e.amount

    def _rows(totals: dict[str, Decimal]) -> list[str]:
        return [
            f"  {label:<42} {currency} {_fmt(v):>14}"
            for label, v in sorted(totals.items())
        ]

    sep = "  " + "-" * 62
    total_in  = sum(in_totals.values(),  Decimal("0"))
    total_out = sum(out_totals.values(), Decimal("0"))

    lines = [
        _section_header(f"NET WORTH SUMMARY  (as at {as_of_str})"),
        f"  Computed net worth:    {currency} {_fmt(net_worth)}",
        "",
        "  INFLOWS",
        *_rows(in_totals),
        sep,
        f"  Total inflows:         {currency} {_fmt(total_in)}",
        "",
        "  OUTFLOWS",
        *_rows(out_totals),
        sep,
        f"  Total outflows:        {currency} {_fmt(total_out)}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prose section prompt builders (facts injected, LLM writes language)
# ---------------------------------------------------------------------------

def _prompt_introduction(
    profile: Profile,
    net_worth: Decimal,
    spec: ScenarioSpec,
    tone: str,
) -> str:
    return (
        f"Write an introduction section ({tone} tone, 2-4 sentences) for a "
        f"source-of-wealth assessment.\n\n"
        f"Facts:\n"
        f"  Name:         {profile.name}\n"
        f"  Age:          {profile.age}\n"
        f"  Occupation:   {profile.occupation}\n"
        f"  Industry:     {profile.industry or 'Finance'}\n"
        f"  Domicile:     {profile.domicile}\n"
        f"  Career start: approximately {profile.plausible_career_start.year}\n"
        f"  Net worth:    {spec.currency} {net_worth:,.0f} (as at {spec.as_of})\n"
    )


def _prompt_sow_claim(
    claim: Claim,
    events: list[Event],
    profile: Profile,
    spec: ScenarioSpec,
    style: str,
    tone: str,
) -> str:
    event_by_id = {e.event_id: e for e in events}
    covered = [event_by_id[eid] for eid in claim.covered_event_ids if eid in event_by_id]
    sentences = "2-3" if style == "concise" else "3-5"

    lines = [
        f"Write {sentences} sentences asserting this source-of-wealth claim "
        f"({tone} tone). State the amount explicitly.",
        "",
        f"Facts:",
        f"  Client:    {profile.name}",
        f"  SoW type:  {claim.sow_type.value.replace('_', ' ').title()}",
        f"  Amount:    {spec.currency} {claim.amount:,.2f}",
    ]

    if claim.sow_type == SowType.employment and covered:
        dates = sorted(e.date for e in covered)
        meta = covered[0].meta
        employer = meta.get("employer_name", "")
        if employer:
            lines.append(f"  Employer:  {employer}")
        lines += [
            f"  Period:    {dates[0].strftime('%B %Y')} to {dates[-1].strftime('%B %Y')}",
            f"  Years:     {len(set(e.date.year for e in covered))}",
        ]
    elif claim.sow_type == SowType.inheritance and covered:
        evt = covered[0]
        lines += [
            f"  Source:    estate of {evt.meta.get('deceased_name', 'the deceased')}",
            f"  Date:      {evt.date.strftime('%d %B %Y')}",
        ]
    elif claim.sow_type == SowType.gift and covered:
        evt = covered[0]
        donor = evt.meta.get("donor_name", "")
        rel   = evt.meta.get("donor_relationship", "")
        if donor:
            lines.append(f"  Donor:     {donor}")
        if rel:
            lines.append(f"  Relationship: {rel}")
        lines.append(f"  Date:      {evt.date.strftime('%d %B %Y')}")
    elif claim.sow_type == SowType.business_profits and covered:
        dates = sorted(e.date for e in covered)
        co_name = covered[0].meta.get("company_name", "")
        if co_name:
            lines.append(f"  Company:   {co_name}")
        lines.append(
            f"  Period:    {dates[0].strftime('%B %Y')} to {dates[-1].strftime('%B %Y')}"
        )

    return "\n".join(l for l in lines if l is not None)


def _prompt_financial_overview(
    events: list[Event],
    net_worth: Decimal,
    spec: ScenarioSpec,
    style: str,
    tone: str,
) -> str:
    from sow_synth.models import INFLOW_TYPES

    in_totals: dict[str, Decimal] = {}
    out_totals: dict[str, Decimal] = {}
    for e in events:
        label = e.type.value.replace("_", " ")
        if e.type in INFLOW_TYPES:
            in_totals[label] = in_totals.get(label, Decimal("0")) + e.amount
        else:
            out_totals[label] = out_totals.get(label, Decimal("0")) + e.amount

    in_block  = "\n".join(f"  {k}: {spec.currency} {v:,.2f}" for k, v in in_totals.items())
    out_block = "\n".join(f"  {k}: {spec.currency} {v:,.2f}" for k, v in out_totals.items())
    sentences = "2-3" if style == "concise" else "3-5"

    return (
        f"Write {sentences} sentences summarising the client's financial position "
        f"({tone} tone).\n\n"
        f"Net worth (as at {spec.as_of}): {spec.currency} {net_worth:,.2f}\n\n"
        f"Inflows:\n{in_block}\n\nOutflows:\n{out_block}\n"
    )


def _prompt_compliance_notes(
    profile: Profile,
    claims: list[Claim],
    spec: ScenarioSpec,
    tone: str,
) -> str:
    claim_lines = "\n".join(
        f"  {c.sow_type.value}: {spec.currency} {c.amount:,.2f}" for c in claims
    )
    return (
        f"Write 2-3 sentences of compliance observations for this client's "
        f"source-of-wealth file ({tone} tone). Comment on documentation coverage "
        f"and any areas requiring further verification. Do not make definitive "
        f"compliance judgements.\n\n"
        f"Client: {profile.name}, {profile.occupation}\n"
        f"Claims:\n{claim_lines}\n"
    )


def _prompt_rm_assessment(
    profile: Profile,
    claims: list[Claim],
    net_worth: Decimal,
    spec: ScenarioSpec,
    style: str,
    tone: str,
) -> str:
    claim_lines = "\n".join(
        f"  {c.sow_type.value}: {spec.currency} {c.amount:,.2f}" for c in claims
    )
    sentences = "2-3" if style == "concise" else "3-4"
    return (
        f"Write a relationship manager assessment ({sentences} sentences, "
        f"{tone} tone). Comment on the overall source-of-wealth picture and "
        f"any notable features. Do not make compliance judgements.\n\n"
        f"Client:    {profile.name}, age {profile.age}, {profile.occupation}\n"
        f"Domicile:  {profile.domicile}\n"
        f"Net worth: {spec.currency} {net_worth:,.2f}\n\n"
        f"Claims:\n{claim_lines}\n"
    )


# ---------------------------------------------------------------------------
# Section header labels
# ---------------------------------------------------------------------------

_SECTION_LABELS: dict[SectionType, str] = {
    SectionType.introduction:       "INTRODUCTION",
    SectionType.sow_claim:          "SOURCE OF WEALTH",
    SectionType.financial_overview: "FINANCIAL OVERVIEW",
    SectionType.compliance_notes:   "COMPLIANCE NOTES",
    SectionType.rm_assessment:      "RELATIONSHIP MANAGER ASSESSMENT",
}


def _rm_name(client_id: str) -> str:
    import hashlib
    names = [
        "J. Hargreaves", "S. Pemberton", "A. Whitfield",
        "C. Thornton",   "R. Ashworth",  "M. Castleton",
    ]
    idx = int(hashlib.md5(client_id.encode()).hexdigest(), 16) % len(names)
    return names[idx]


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

def _paginate(sections: list[str], budget: int = _PAGE_CHAR_BUDGET) -> list[str]:
    """Split rendered sections into pages by character budget.

    Sections are never split mid-section; a section that exceeds the budget
    on its own starts a fresh page.
    """
    pages: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sec in sections:
        if current_parts and current_len + len(sec) > budget:
            pages.append("\n".join(current_parts))
            current_parts = [sec]
            current_len   = len(sec)
        else:
            current_parts.append(sec)
            current_len += len(sec)

    if current_parts:
        pages.append("\n".join(current_parts))

    return pages


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_client_history(
    profile: Profile,
    events: list[Event],
    claims: list[Claim],
    spec: ScenarioSpec,
    llm: "LlmClient",
    telemetry: "Telemetry",
) -> tuple[Document, list[Claim]]:
    """Generate the client history document via two-phase LLM generation.

    Phase 1: LLM produces a DocumentStructurePlan (section order + style).
    Phase 2: Each section rendered — code for tabular, LLM for prose.
    Phase 3: Sections paginated by character budget.

    Returns:
        doc            — rendered Document (role='client_history', N pages)
        updated_claims — same claims with Claim.asserted_text populated
    """
    currency  = spec.currency
    net_worth = compute_net_worth(events, spec.as_of)
    doc_id    = f"doc_client_history_{spec.seed:08d}"
    as_of_str = spec.as_of.strftime("%d %B %Y")
    claim_ids = [c.claim_id for c in claims]
    claim_by_id = {c.claim_id: c for c in claims}

    # -- Phase 1: structural plan --
    profile_summary = (
        f"{profile.name}, age {profile.age}, {profile.occupation} "
        f"({profile.industry or 'finance'}), domicile {profile.domicile}. "
        f"Net worth: {currency} {net_worth:,.0f}."
    )
    claim_summaries = [
        f"{c.sow_type.value.replace('_', ' ').title()}: "
        f"{currency} {c.amount:,.2f}  [id={c.claim_id}]"
        for c in claims
    ]
    n_pages_hint = max(2, 1 + len(claims) // 2)

    raw_plan: DocumentStructurePlan = llm.complete(
        step="ch_structure_plan",
        system=(
            "You are structuring a source-of-wealth client history document. "
            "Return a DocumentStructurePlan with sections in the order they "
            "should appear. Vary the structure across documents — avoid fixed templates."
        ),
        user=build_plan_prompt(profile_summary, claim_summaries, n_pages_hint),
        response_model=DocumentStructurePlan,
    )
    plan = enforce_constraints(raw_plan, claim_ids)
    tone = plan.overall_tone

    # -- Phase 2: render each section --
    rendered_sections: list[str] = []
    claim_paragraphs: dict[str, str] = {}

    for spec_sec in plan.sections:
        stype  = spec_sec.section_type
        style  = spec_sec.style
        header = _SECTION_LABELS.get(stype)

        if stype == SectionType.client_profile:
            rendered_sections.append(
                _render_client_profile(doc_id, as_of_str, currency, profile)
            )

        elif stype == SectionType.net_worth_table:
            rendered_sections.append(
                _render_net_worth_table(as_of_str, currency, net_worth, events)
            )

        elif stype == SectionType.introduction:
            prose = llm.complete(
                step="ch_introduction",
                system=_SYSTEM,
                user=_prompt_introduction(profile, net_worth, spec, tone),
                response_model=_Prose,
            ).paragraph
            block = _section_header("INTRODUCTION") + "\n"
            block += "\n".join(textwrap.wrap(prose, _WRAP))
            rendered_sections.append(block)

        elif stype == SectionType.sow_claim:
            claim = claim_by_id[spec_sec.claim_id]
            label = (
                f"{claim.sow_type.value.replace('_', ' ').upper()}"
                f"  [{currency} {_fmt(claim.amount)}]"
            )
            prose = llm.complete(
                step=f"ch_claim_{claim.claim_id}",
                system=_SYSTEM,
                user=_prompt_sow_claim(claim, events, profile, spec, style, tone),
                response_model=_Prose,
            ).paragraph
            claim_paragraphs[claim.claim_id] = prose
            sep = "-" * len(label)
            block = _section_header("SOURCE OF WEALTH") if rendered_sections and (
                not rendered_sections[-1].endswith("SOURCE OF WEALTH\n" + "=" * 72 + "\n")
            ) else ""
            # Only emit the section header once before the first claim; subsequent
            # claims in sequence just get their label+separator.
            if block:
                rendered_sections.append(block)
            rendered_sections.append(
                f"\n{label}\n{sep}\n\n"
                + "\n".join(textwrap.wrap(prose, _WRAP))
            )

        elif stype == SectionType.financial_overview:
            prose = llm.complete(
                step="ch_financial_overview",
                system=_SYSTEM,
                user=_prompt_financial_overview(events, net_worth, spec, style, tone),
                response_model=_Prose,
            ).paragraph
            block = _section_header("FINANCIAL OVERVIEW") + "\n"
            block += "\n".join(textwrap.wrap(prose, _WRAP))
            rendered_sections.append(block)

        elif stype == SectionType.compliance_notes:
            prose = llm.complete(
                step="ch_compliance_notes",
                system=_SYSTEM,
                user=_prompt_compliance_notes(profile, claims, spec, tone),
                response_model=_Prose,
            ).paragraph
            block = _section_header("COMPLIANCE NOTES") + "\n"
            block += "\n".join(textwrap.wrap(prose, _WRAP))
            rendered_sections.append(block)

        elif stype == SectionType.rm_assessment:
            prose = llm.complete(
                step="ch_rm_assessment",
                system=_SYSTEM,
                user=_prompt_rm_assessment(profile, claims, net_worth, spec, style, tone),
                response_model=_Prose,
            ).paragraph
            block = _section_header("RELATIONSHIP MANAGER ASSESSMENT") + "\n"
            block += "\n".join(textwrap.wrap(prose, _WRAP))
            block += f"\n\nPrepared by: {_rm_name(profile.client_id)}"
            rendered_sections.append(block)

    # -- footer on last section --
    rendered_sections.append(f"\n{_RULE}\nDocument ID: {doc_id}\n{_RULE}")

    # -- Phase 3: paginate --
    pages_text = _paginate(rendered_sections)
    pages = [
        OcrPage(page_number=i + 1, page_text=f"<pre>{text}</pre>")
        for i, text in enumerate(pages_text)
    ]

    doc = Document(
        doc_id=doc_id,
        doc_type=DocType.client_history,
        role="client_history",
        pages=pages,
    )

    # -- Populate Claim.asserted_text --
    updated_claims = [
        c.model_copy(update={"asserted_text": claim_paragraphs.get(c.claim_id)})
        for c in claims
    ]

    return doc, updated_claims
