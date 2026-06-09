"""Stage 5 — Document planning.

One DocumentPlan is created per Claim.  The format is drawn from the weighted
registry so different seeds produce different document types for the same
claim type (e.g. an employment claim might be corroborated by a payslip, a
bank statement, an employer letter, an email, or a Bloomberg article).

The claim↔document linkage (corroborates edges) is established HERE — before
the graph is frozen and before any rendering happens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal

import numpy as np

from sow_synth.formats import FORMATS, FormatSpec
from sow_synth.formats.flavor import (
    account_last4, bank_name, company_address, company_name as gen_company_name,
    company_number, deceased_name, donor_name, donor_relationship, sort_code,
    solicitor_firm,
)
from sow_synth.formats.registry import select_format
from sow_synth.models import Claim, DocType, Event, EventType, Profile, SowType
from sow_synth.spec import ScenarioSpec

_TWO = Decimal("0.01")

# ---------------------------------------------------------------------------
# Employer pool (used for employment docs)
# ---------------------------------------------------------------------------

_EMPLOYERS: dict[str, list[dict]] = {
    "finance": [
        {"name": "Meridian Capital Partners LLP",  "address": "30 St Mary Axe, London EC3A 8BF"},
        {"name": "Blackwater Asset Management Ltd", "address": "1 Canada Square, London E14 5AB"},
        {"name": "Thornfield Investment Group",     "address": "25 Gresham Street, London EC2V 7HN"},
    ],
    "technology": [
        {"name": "Nexus Systems Ltd",          "address": "10 Bishops Square, London E1 6EG"},
        {"name": "Vertex Technologies plc",    "address": "5 New Street Square, London EC4A 3TW"},
        {"name": "Cloudbridge Solutions Ltd",  "address": "1 Euston Square, London NW1 2FD"},
    ],
    "real_estate": [
        {"name": "Harrington Property Group Ltd", "address": "48 Grosvenor Square, London W1K 2HP"},
        {"name": "Ashworth & Partners LLP",       "address": "15 Berkeley Street, London W1J 8DY"},
        {"name": "Castleton Real Estate Ltd",     "address": "33 Cavendish Square, London W1G 0PW"},
    ],
    "professional_services": [
        {"name": "Whitmore & Associates LLP",   "address": "4 Temple Place, London WC2R 2PG"},
        {"name": "Holborn Advisory Group Ltd",  "address": "7 Lincoln's Inn Fields, London WC2A 3BP"},
        {"name": "Stanton Consulting plc",      "address": "12 Grays Inn Road, London WC1X 8AL"},
    ],
}
_DEFAULT_EMPLOYERS = _EMPLOYERS["finance"]

_TAX_RATE = Decimal("0.25")
_NI_RATE  = Decimal("0.05")


# ---------------------------------------------------------------------------
# DocumentPlan
# ---------------------------------------------------------------------------

@dataclass
class DocumentPlan:
    """Blueprint for one corroboration document (created in Stage 5).

    `verify_hints` is a list of {"key": ..., "expected": ..., "precision": ...}
    dicts that verify.py uses to check rendered content against the fact layer.
    """
    doc_id: str
    doc_type: DocType
    role: Literal["corroboration"] = "corroboration"
    source_event_ids: list[str]      = field(default_factory=list)
    corroborates_claim_ids: list[str]= field(default_factory=list)
    template_context: dict           = field(default_factory=dict)
    verify_hints: list[dict]         = field(default_factory=list)

    def to_document(self):
        from sow_synth.models import Document
        return Document(doc_id=self.doc_id, doc_type=self.doc_type,
                        role=self.role, pages=[])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _employer_for(profile: Profile, role_index: int, seed: int) -> dict:
    pool = _EMPLOYERS.get(profile.industry or "", _DEFAULT_EMPLOYERS)
    return pool[(role_index + seed) % len(pool)]


def _fmt_date(d: date) -> str:
    return d.strftime("%d %b %Y")


def _period_label(start: date, end: date) -> str:
    if start.year == end.year:
        return str(start.year)
    return f"{start.year}–{end.year}"


# ---------------------------------------------------------------------------
# Per-SoW-type evidence context builders
# ---------------------------------------------------------------------------

def _employment_ctx(
    profile: Profile,
    claim: Claim,
    events: list[Event],
    spec: ScenarioSpec,
    fmt: FormatSpec,
    rng: np.random.Generator,
) -> dict:
    """Aggregate all employment_income events for this claim into one context."""
    event_by_id = {e.event_id: e for e in events}
    emp_events = sorted(
        [event_by_id[eid] for eid in claim.covered_event_ids
         if event_by_id[eid].type == EventType.employment_income],
        key=lambda e: e.date,
    )
    if not emp_events:
        return {}

    role_index = emp_events[0].meta.get("role_index", 0)
    employer   = _employer_for(profile, role_index, spec.seed)
    gross      = claim.amount
    income_tax = (gross * _TAX_RATE).quantize(_TWO)
    ni         = (gross * _NI_RATE).quantize(_TWO)
    net        = gross - income_tax - ni
    start_date = emp_events[0].date
    end_date   = emp_events[-1].date
    ty_start   = date(start_date.year, 4, 6)
    ty_end     = date(end_date.year + 1, 4, 5)

    # Monthly transactions for bank_statement format
    from sow_synth.formats.realize import _monthly_split, _fmt, _MONTHS
    months_split = _monthly_split(gross)
    transactions_raw = [
        {"date": f"28 {m}", "description": f"SALARY - {employer['name'][:20].upper()}",
         "credit": _fmt(a), "debit": ""}
        for m, a in zip(_MONTHS, months_split)
    ]

    return {
        "subject_name": profile.name,
        "subject_address": "",
        "employer_name": employer["name"],
        "employer_address": employer["address"],
        "employer_paye_ref": f"{100 + role_index:03d}/A{spec.seed % 10000:04d}",
        "employee_id": f"EMP{abs(hash(profile.client_id)) % 100000:05d}",
        "ni_number": f"AB{abs(hash(profile.client_id + 'ni')) % 1000000:06d}C",
        "tax_code": "1257L",
        "gross_pay": str(gross),
        "income_tax": str(income_tax),
        "ni_contributions": str(ni),
        "net_pay": str(net),
        "period_start": _fmt_date(start_date),
        "period_end": _fmt_date(end_date),
        "period_label": _period_label(start_date, end_date),
        "tax_year_start": ty_start.isoformat(),
        "tax_year_end": ty_end.isoformat(),
        "tax_year_label": f"{start_date.year}/{str(end_date.year + 1)[2:]}",
        "year": str(start_date.year),
        "still_employed": end_date >= spec.as_of,
        "industry": profile.industry or "finance",
        "role_index": role_index,
        "primary_amount": str(gross),
        "transactions_raw": transactions_raw,
        "opening_balance_amount": "5000.00",
        "bank_name": bank_name(profile.client_id + str(role_index)),
        "sort_code": sort_code(profile.client_id + str(role_index)),
        "account_last4": account_last4(profile.client_id + str(role_index)),
        # bloomberg context
        "bloomberg_type": "employment",
        # email context
        "email_type": "employment",
        # solicitor context
        "letter_type": "employment",
    }


def _inheritance_ctx(
    profile: Profile,
    claim: Claim,
    events: list[Event],
    spec: ScenarioSpec,
    fmt: FormatSpec,
    rng: np.random.Generator,
) -> dict:
    event_by_id = {e.event_id: e for e in events}
    evt = event_by_id[claim.covered_event_ids[0]]
    amount = claim.amount
    seed_str = profile.client_id + evt.event_id
    dec_name = deceased_name(seed_str)
    probate_d = date(evt.date.year, evt.date.month, min(evt.date.day + 60, 28))
    will_d    = date(evt.date.year - 2, 1, 15)

    return {
        "subject_name": profile.name,
        "deceased_name": dec_name,
        "event_date": _fmt_date(evt.date),
        "death_date": _fmt_date(evt.date),
        "grant_date": _fmt_date(probate_d),
        "will_date": _fmt_date(will_d),
        "domicile": "England and Wales",
        "primary_amount": str(amount),
        "currency": spec.currency,
        # bank transfer
        "sender_name": dec_name,
        "recipient_name": profile.name,
        "transfer_date": _fmt_date(probate_d),
        # probate
        "gross_estate": str(amount),
        "executor_name": profile.name,
        "executor_address": "",
        # solicitor
        "letter_type": "inheritance",
        "matter_description": f"Estate of {dec_name} — Beneficiary Distribution",
        # bloomberg
        "bloomberg_type": "inheritance",
        # email
        "email_type": "inheritance",
        "solicitor_firm": solicitor_firm(seed_str)["name"],
        "beneficiary_name": profile.name,
        "estate_amount": str(amount),
        "transfer_date": _fmt_date(probate_d),
    }


def _gift_ctx(
    profile: Profile,
    claim: Claim,
    events: list[Event],
    spec: ScenarioSpec,
    fmt: FormatSpec,
    rng: np.random.Generator,
) -> dict:
    event_by_id = {e.event_id: e for e in events}
    evt = event_by_id[claim.covered_event_ids[0]]
    amount = claim.amount
    seed_str = profile.client_id + evt.event_id
    dn = donor_name(seed_str)
    dr = donor_relationship(seed_str)

    return {
        "subject_name": profile.name,
        "donor_name": dn,
        "donor_relationship": dr,
        "donor_address": "Address provided separately",
        "event_date": _fmt_date(evt.date),
        "gift_date": _fmt_date(evt.date),
        "primary_amount": str(amount),
        "currency": spec.currency,
        "intended_use": "personal wealth management",
        # bank transfer
        "sender_name": dn,
        "recipient_name": profile.name,
        "transfer_date": _fmt_date(evt.date),
        "payment_reference": f"GIFT{abs(hash(seed_str)) % 100000:05d}",
        # solicitor
        "letter_type": "gift",
        "matter_description": f"Gift from {dn} to {profile.name}",
        # bloomberg (unusual but possible)
        "bloomberg_type": "inheritance",
        # email
        "email_type": "gift",
    }


def _business_ctx(
    profile: Profile,
    claim: Claim,
    events: list[Event],
    spec: ScenarioSpec,
    fmt: FormatSpec,
    rng: np.random.Generator,
) -> dict:
    event_by_id = {e.event_id: e for e in events}
    biz_events = sorted(
        [event_by_id[eid] for eid in claim.covered_event_ids
         if event_by_id[eid].type == EventType.business_profit_distribution],
        key=lambda e: e.date,
    )
    if not biz_events:
        return {}

    biz_id   = biz_events[0].meta.get("business_id", profile.client_id + "biz")
    co_name  = gen_company_name(biz_id)
    co_num   = company_number(biz_id)
    co_addr  = company_address(biz_id)
    amount   = claim.amount
    start    = biz_events[0].date
    end      = biz_events[-1].date

    # Transactions for bank_statement
    from sow_synth.formats.realize import _fmt
    transactions_raw = [
        {"date": _fmt_date(e.date),
         "description": f"DIV - {co_name[:20].upper()}",
         "credit": _fmt(e.amount), "debit": ""}
        for e in biz_events
    ]

    return {
        "subject_name": profile.name,
        "company_name": co_name,
        "company_number": co_num,
        "company_address": co_addr,
        "incorporation_date": _fmt_date(start - __import__("datetime").timedelta(days=730)),
        "director_name": profile.name,
        "event_date": _fmt_date(end),
        "period_start": _fmt_date(start),
        "period_end": _fmt_date(end),
        "period_label": _period_label(start, end),
        "tax_year_label": f"{end.year}/{str(end.year + 1)[2:]}",
        "primary_amount": str(amount),
        "currency": spec.currency,
        "transactions_raw": transactions_raw,
        "opening_balance_amount": "10000.00",
        "bank_name": bank_name(biz_id),
        "sort_code": sort_code(biz_id),
        "account_last4": account_last4(biz_id),
        "bloomberg_type": "business",
        "email_type": "business",
        "letter_type": "business",
        "matter_description": f"{co_name} — Profit Distribution",
    }


_CTX_BUILDERS = {
    SowType.employment:      _employment_ctx,
    SowType.inheritance:     _inheritance_ctx,
    SowType.gift:            _gift_ctx,
    SowType.business_profits:_business_ctx,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan_documents(
    profile: Profile,
    events: list[Event],
    claims: list[Claim],
    spec: ScenarioSpec,
    rng: np.random.Generator,
) -> list[DocumentPlan]:
    """One DocumentPlan per Claim, format selected from the weighted registry."""
    plans: list[DocumentPlan] = []
    counter = 1

    for claim in claims:
        fmt = select_format(claim.sow_type, rng)
        ctx_builder = _CTX_BUILDERS.get(claim.sow_type)
        if ctx_builder is None:
            continue

        evidence_ctx = ctx_builder(profile, claim, events, spec, fmt, rng)
        if not evidence_ctx:
            continue

        evidence_ctx.update({
            "doc_id": f"doc_{fmt.doc_type_value}_{spec.seed:08d}_{counter:04d}",
            "currency": spec.currency,
            "format_type": fmt.doc_type_value,
            "precision_mode": fmt.precision_mode.value,
        })

        doc_type = DocType(fmt.doc_type_value)
        doc_id   = evidence_ctx["doc_id"]
        counter += 1

        # verify_hints: what verify.py will check
        verify_hints = [
            {
                "key": fmt.primary_amount_field,
                "expected": str(claim.amount),
                "precision": fmt.precision_mode.value,
            }
        ]

        plans.append(DocumentPlan(
            doc_id=doc_id,
            doc_type=doc_type,
            source_event_ids=list(claim.covered_event_ids),
            corroborates_claim_ids=[claim.claim_id],
            template_context=evidence_ctx,
            verify_hints=verify_hints,
        ))

    return plans
