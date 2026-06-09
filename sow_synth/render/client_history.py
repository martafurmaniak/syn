"""Stage 8 (LLM path) — Client history document generation.

This is the only module in the pipeline that calls the LLM.  All figures,
dates, and names are injected from the fact layer via structured prompts;
the model writes natural-language prose only.

Four call types per sample:
  ch_intro              — 1 call  (client introduction paragraph)
  ch_claim_{claim_id}   — 1 per claim  (becomes Claim.asserted_text)
  ch_financial_summary  — 1 call  (financial overview paragraph)
  ch_advisor_notes      — 1 call  (RM assessment paragraph)

Entry point:
    render_client_history(profile, events, claims, spec, llm, telemetry)
    → (Document, updated_claims)

The returned claims have Claim.asserted_text populated.  Pass them to
plan_documents() and assemble_graph() instead of the originals.
"""
from __future__ import annotations

import textwrap
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from sow_synth.formats.realize import _fmt, _line_to_ocr
from sow_synth.ledger import compute_net_worth
from sow_synth.models import (
    Claim, Document, DocType, Event, EventType,
    KeyValue, OcrPage, Profile, SowType,
)
from sow_synth.spec import ScenarioSpec

# Deferred imports to keep optional at the top level
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sow_synth.llm import LlmClient
    from sow_synth.telemetry import Telemetry

_TEMPLATES_ROOT = Path(__file__).parent.parent / "formats" / "templates"

from jinja2 import Environment, FileSystemLoader, StrictUndefined
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
    undefined=StrictUndefined,
    autoescape=False,
)

_WRAP_WIDTH = 90

# ---------------------------------------------------------------------------
# LLM response schemas — model writes prose, nothing else
# ---------------------------------------------------------------------------

class _IntroParagraph(BaseModel):
    paragraph: str

class _ClaimParagraph(BaseModel):
    paragraph: str   # natural-language assertion of the claim (becomes asserted_text)

class _SummaryParagraph(BaseModel):
    paragraph: str

class _AdvisorNotes(BaseModel):
    paragraph: str


# ---------------------------------------------------------------------------
# Prompt builders — all figures injected, LLM fills prose only
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a relationship manager at a UK private bank writing a source-of-wealth "
    "assessment report. Write concisely and professionally. Use only the facts provided "
    "— do not invent figures, dates, names, or events. Do not add disclaimers about "
    "verification. Do not use bullet points."
)


def _intro_prompt(profile: Profile, net_worth: Decimal, spec: ScenarioSpec) -> str:
    lo, hi = spec.target_net_worth
    return (
        f"Write an introduction paragraph (2-3 sentences) for a source-of-wealth assessment.\n\n"
        f"Client facts:\n"
        f"  Name:         {profile.name}\n"
        f"  Age:          {profile.age}\n"
        f"  Occupation:   {profile.occupation}\n"
        f"  Industry:     {profile.industry or 'finance'}\n"
        f"  Domicile:     {profile.domicile}\n"
        f"  Career start: approximately {profile.plausible_career_start.year}\n"
        f"  Net worth:    {spec.currency} {net_worth:,.0f} (as at {spec.as_of})\n\n"
        f"The paragraph should introduce who the client is and give a brief overview "
        f"of how their wealth was accumulated."
    )


def _claim_prompt(claim: Claim, events: list[Event], profile: Profile, spec: ScenarioSpec) -> str:
    event_by_id = {e.event_id: e for e in events}
    covered = [event_by_id[eid] for eid in claim.covered_event_ids if eid in event_by_id]

    lines = [
        f"Write one paragraph (2-4 sentences) asserting this client's source of wealth.",
        f"",
        f"Facts — use ALL of these, do not omit the amount:",
        f"  Client:    {profile.name}",
        f"  SoW type:  {claim.sow_type.value.replace('_', ' ').title()}",
        f"  Amount:    {spec.currency} {claim.amount:,.2f}",
    ]

    if claim.sow_type == SowType.employment and covered:
        dates = sorted(e.date for e in covered)
        meta  = covered[0].meta
        employer = meta.get("employer_name", "")
        lines += [
            f"  Employer:  {employer}" if employer else "",
            f"  Period:    {dates[0].strftime('%B %Y')} to {dates[-1].strftime('%B %Y')}",
            f"  Years:     {len(set(e.date.year for e in covered))}",
        ]
    elif claim.sow_type == SowType.inheritance and covered:
        evt = covered[0]
        dec_name = evt.meta.get("deceased_name", "the deceased")
        lines += [
            f"  Source:    estate of {dec_name}",
            f"  Date:      {evt.date.strftime('%d %B %Y')}",
        ]
    elif claim.sow_type == SowType.gift and covered:
        evt = covered[0]
        donor = evt.meta.get("donor_name", "")
        rel   = evt.meta.get("donor_relationship", "")
        lines += [
            f"  Donor:     {donor}" if donor else "",
            f"  Relationship: {rel}" if rel else "",
            f"  Date:      {evt.date.strftime('%d %B %Y')}",
        ]
    elif claim.sow_type == SowType.business_profits and covered:
        dates = sorted(e.date for e in covered)
        biz_id = covered[0].meta.get("business_id", "")
        co_name = covered[0].meta.get("company_name", "")
        lines += [
            f"  Company:   {co_name}" if co_name else "",
            f"  Period:    {dates[0].strftime('%B %Y')} to {dates[-1].strftime('%B %Y')}",
            f"  Director / principal shareholder: {profile.name}",
        ]

    lines = [l for l in lines if l]
    lines.append("")
    lines.append(
        "The paragraph must state the amount explicitly using the currency code and numeric value."
    )
    return "\n".join(lines)


def _financial_summary_prompt(
    events: list[Event],
    net_worth: Decimal,
    spec: ScenarioSpec,
) -> str:
    from sow_synth.models import INFLOW_TYPES, OUTFLOW_TYPES

    inflows_by_type: dict[str, Decimal] = {}
    outflows_by_type: dict[str, Decimal] = {}
    for e in events:
        if e.type in INFLOW_TYPES:
            inflows_by_type[e.type.value] = inflows_by_type.get(e.type.value, Decimal("0")) + e.amount
        else:
            outflows_by_type[e.type.value] = outflows_by_type.get(e.type.value, Decimal("0")) + e.amount

    inflow_lines  = "\n".join(f"  {k}: {spec.currency} {v:,.2f}" for k, v in inflows_by_type.items())
    outflow_lines = "\n".join(f"  {k}: {spec.currency} {v:,.2f}" for k, v in outflows_by_type.items())

    return (
        f"Write one paragraph (3-4 sentences) summarising this client's financial position.\n\n"
        f"Net worth (as at {spec.as_of}): {spec.currency} {net_worth:,.2f}\n\n"
        f"Inflows:\n{inflow_lines}\n\n"
        f"Outflows:\n{outflow_lines}\n\n"
        f"The paragraph should summarise the overall financial picture, referencing the net worth "
        f"and the main categories of inflow and outflow."
    )


def _advisor_notes_prompt(
    profile: Profile,
    claims: list[Claim],
    net_worth: Decimal,
    spec: ScenarioSpec,
) -> str:
    claim_lines = "\n".join(
        f"  {c.sow_type.value}: {spec.currency} {c.amount:,.2f}"
        for c in claims
    )
    return (
        f"Write a brief relationship manager assessment (2-3 sentences) for this client's "
        f"source-of-wealth file. Comment on the overall picture and any notable features.\n\n"
        f"Client:    {profile.name}, age {profile.age}, {profile.occupation}\n"
        f"Domicile:  {profile.domicile}\n"
        f"Net worth: {spec.currency} {net_worth:,.2f}\n\n"
        f"Claims:\n{claim_lines}\n\n"
        f"Write in professional, neutral tone. Do not make compliance judgements."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap(text: str) -> list[str]:
    """Word-wrap a paragraph into lines for the OCR template."""
    return textwrap.wrap(text, width=_WRAP_WIDTH) or [""]


def _fmt_amount(currency: str, amount: Decimal) -> str:
    return f"{currency} {_fmt(amount)}"


def _inflow_outflow_rows(events: list[Event], currency: str):
    from sow_synth.models import INFLOW_TYPES, OUTFLOW_TYPES

    in_totals: dict[str, Decimal]  = {}
    out_totals: dict[str, Decimal] = {}
    for e in events:
        if e.type in INFLOW_TYPES:
            in_totals[e.type.value] = in_totals.get(e.type.value, Decimal("0")) + e.amount
        else:
            out_totals[e.type.value] = out_totals.get(e.type.value, Decimal("0")) + e.amount

    def _rows(totals):
        rows = []
        for k, v in sorted(totals.items()):
            label = k.replace("_", " ").title()
            rows.append({"line": f"{label:<40} {currency} {_fmt(v):>14}"})
        return rows

    return (
        _rows(in_totals),
        _rows(out_totals),
        sum(in_totals.values(), Decimal("0")),
        sum(out_totals.values(), Decimal("0")),
    )


def _rm_name(client_id: str) -> str:
    import hashlib
    names = [
        "J. Hargreaves", "S. Pemberton", "A. Whitfield",
        "C. Thornton",   "R. Ashworth",  "M. Castleton",
    ]
    idx = int(hashlib.md5(client_id.encode()).hexdigest(), 16) % len(names)
    return names[idx]


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def _render_page(template_path: str, ctx: dict, page_num: int,
                 key_values: list[KeyValue]) -> OcrPage:
    tpl = _JINJA_ENV.get_template(template_path)
    rendered = tpl.render(**ctx)
    pw, ph = 595.0, 842.0
    lines = [
        _line_to_ocr(line, pw, i)
        for i, line in enumerate(rendered.splitlines())
        if line.strip()
    ]
    return OcrPage(page_number=page_num, width=pw, height=ph,
                   lines=lines, key_values=key_values)


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
    """Generate the client history document.

    Makes (3 + len(claims)) LLM calls for prose sections.  All figures are
    injected — the model only writes natural-language paragraphs.

    Returns:
        doc            — rendered Document (role='client_history', 2 pages)
        updated_claims — same claims with Claim.asserted_text populated
    """
    currency  = spec.currency
    net_worth = compute_net_worth(events, spec.as_of)
    doc_id    = f"doc_client_history_{spec.seed:08d}"
    as_of_str = spec.as_of.strftime("%d %B %Y")

    # -- LLM call 1: intro paragraph --
    intro_para = llm.complete(
        step="ch_intro",
        system=_SYSTEM,
        user=_intro_prompt(profile, net_worth, spec),
        response_model=_IntroParagraph,
    ).paragraph

    # -- LLM calls 2..N: one paragraph per claim --
    claim_paragraphs: dict[str, str] = {}
    for claim in claims:
        para = llm.complete(
            step=f"ch_claim_{claim.claim_id}",
            system=_SYSTEM,
            user=_claim_prompt(claim, events, profile, spec),
            response_model=_ClaimParagraph,
        ).paragraph
        claim_paragraphs[claim.claim_id] = para

    # -- LLM call N+1: financial summary --
    financial_para = llm.complete(
        step="ch_financial_summary",
        system=_SYSTEM,
        user=_financial_summary_prompt(events, net_worth, spec),
        response_model=_SummaryParagraph,
    ).paragraph

    # -- LLM call N+2: advisor notes --
    advisor_para = llm.complete(
        step="ch_advisor_notes",
        system=_SYSTEM,
        user=_advisor_notes_prompt(profile, claims, net_worth, spec),
        response_model=_AdvisorNotes,
    ).paragraph

    # -- Build page 1 context --
    claim_sections = []
    for claim in claims:
        label = f"{claim.sow_type.value.replace('_',' ').upper()}  [{currency} {_fmt(claim.amount)}]"
        claim_sections.append({
            "label": label,
            "lines": _wrap(claim_paragraphs[claim.claim_id]),
        })

    p1_ctx = {
        "doc_id":        doc_id,
        "as_of_date":    as_of_str,
        "currency":      currency,
        "client_name":   profile.name,
        "dob":           profile.date_of_birth.strftime("%d %B %Y"),
        "age":           profile.age,
        "occupation":    profile.occupation,
        "domicile":      profile.domicile,
        "industry":      profile.industry or "Finance",
        "intro_lines":   _wrap(intro_para),
        "claim_sections": claim_sections,
    }

    inflow_rows, outflow_rows, total_in, total_out = _inflow_outflow_rows(events, currency)

    p2_ctx = {
        "doc_id":        doc_id,
        "as_of_date":    as_of_str,
        "currency":      currency,
        "net_worth":     _fmt(net_worth),
        "inflow_rows":   inflow_rows,
        "outflow_rows":  outflow_rows,
        "total_inflows":  _fmt(total_in),
        "total_outflows": _fmt(total_out),
        "financial_lines": _wrap(financial_para),
        "advisor_lines":   _wrap(advisor_para),
        "rm_name":         _rm_name(profile.client_id),
    }

    # -- Key values --
    p1_kvs = [
        KeyValue(key="Client Name",     value=profile.name),
        KeyValue(key="Assessment Date", value=as_of_str),
        KeyValue(key="Document Type",   value="Client History"),
        KeyValue(key="Page",            value="1 of 2"),
    ]
    p2_kvs = [
        KeyValue(key="Net Worth",       value=f"{currency} {_fmt(net_worth)}"),
        KeyValue(key="Total Inflows",   value=f"{currency} {_fmt(total_in)}"),
        KeyValue(key="Total Outflows",  value=f"{currency} {_fmt(total_out)}"),
        KeyValue(key="Page",            value="2 of 2"),
    ]

    # -- Render pages --
    page1 = _render_page("structured/client_history_p1.j2", p1_ctx, 1, p1_kvs)
    page2 = _render_page("structured/client_history_p2.j2", p2_ctx, 2, p2_kvs)

    doc = Document(
        doc_id=doc_id,
        doc_type=DocType.client_history,
        role="client_history",
        pages=[page1, page2],
    )

    # -- Update claims with asserted_text --
    updated_claims = [
        claim.model_copy(update={"asserted_text": claim_paragraphs[claim.claim_id]})
        for claim in claims
    ]

    return doc, updated_claims
