"""Stage 5 — Corroboration bundle planning.

For each Claim, a CorroborationBundle is produced: a group of 1..N DocumentSpecs
that together constitute evidence for that claim.  Bundle composition is chosen
deterministically from the seed — different seeds produce different bundle templates
(some claims get a single payslip, others get a contract + bank statement, etc.).

Each DocumentSpec carries:
  - doc_type + evidential_role: what kind of document and what it proves
  - mandatory_facts: dict[str, str] — every figure the rendered text MUST contain
    (amounts, dates, names, references); verify.py checks these post-rendering

The claim↔document linkage (corroborates edges) is established HERE, before the
graph is frozen.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

import numpy as np

from sow_synth.formats.flavor import (
    account_last4, bank_name, company_address, company_name as gen_company_name,
    company_number, deceased_name, donor_name, donor_relationship, sort_code,
    solicitor_firm,
)
from sow_synth.formats.helpers import fmt, fmt_currency, h, monthly_split, MONTHS
from sow_synth.models import Claim, DocType, EvidentialRole, Event, EventType, Profile, SowType
from sow_synth.spec import ScenarioSpec

_TWO = Decimal("0.01")
_TAX_RATE = Decimal("0.25")
_NI_RATE  = Decimal("0.05")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DocumentSpec:
    """Blueprint for one document within a corroboration bundle."""
    doc_id: str
    doc_type: DocType
    evidential_role: EvidentialRole
    mandatory_facts: dict[str, str]        # key → value; rendered text must contain all values
    style_hint: str = "formal"
    source_event_ids: list[str] = field(default_factory=list)

    def to_document(self):
        from sow_synth.models import Document
        return Document(doc_id=self.doc_id, doc_type=self.doc_type,
                        role="corroboration", pages=[])


@dataclass
class CorroborationBundle:
    """All documents required to corroborate one Claim."""
    bundle_id: str
    claim_id: str
    doc_specs: list[DocumentSpec]

    @property
    def all_doc_ids(self) -> list[str]:
        return [s.doc_id for s in self.doc_specs]


# ---------------------------------------------------------------------------
# Employer pool
# ---------------------------------------------------------------------------

_EMPLOYERS: dict[str, list[dict]] = {
    "finance": [
        {"name": "Meridian Capital Partners LLP",   "address": "30 St Mary Axe, London EC3A 8BF"},
        {"name": "Blackwater Asset Management Ltd",  "address": "1 Canada Square, London E14 5AB"},
        {"name": "Thornfield Investment Group",      "address": "25 Gresham Street, London EC2V 7HN"},
    ],
    "technology": [
        {"name": "Nexus Systems Ltd",         "address": "10 Bishops Square, London E1 6EG"},
        {"name": "Vertex Technologies plc",   "address": "5 New Street Square, London EC4A 3TW"},
        {"name": "Cloudbridge Solutions Ltd", "address": "1 Euston Square, London NW1 2FD"},
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


def _employer_for(profile: Profile, role_index: int, seed: int) -> dict:
    pool = _EMPLOYERS.get(profile.industry or "", _DEFAULT_EMPLOYERS)
    return pool[(role_index + seed) % len(pool)]


def _fmt_date(d: date) -> str:
    return d.strftime("%d %b %Y")


def _period_label(start: date, end: date) -> str:
    if start.year == end.year:
        return str(start.year)
    return f"{start.year}–{end.year}"


def _ref(seed_str: str, prefix: str = "REF") -> str:
    return f"{prefix}{h(seed_str) % 1000000:06d}"


# ---------------------------------------------------------------------------
# Bundle option tables  (DocType, EvidentialRole, style_hint)
# ---------------------------------------------------------------------------

_BundleOption = list[tuple[DocType, EvidentialRole, str]]

_EMPLOYMENT_BUNDLES: list[_BundleOption] = [
    # A — payslip + employer letter
    [(DocType.payslip,             EvidentialRole.proves_income_amount,    "formal"),
     (DocType.employer_letter,     EvidentialRole.proves_employment_tenure,"formal")],
    # B — contract + bank statement (transaction trail)
    [(DocType.employment_contract, EvidentialRole.proves_employment_tenure,"legal"),
     (DocType.bank_statement,      EvidentialRole.proves_transaction,      "formal")],
    # C — payslip + bank statement
    [(DocType.payslip,             EvidentialRole.proves_income_amount,    "formal"),
     (DocType.bank_statement,      EvidentialRole.proves_transaction,      "formal")],
    # D — single employer letter (simpler)
    [(DocType.employer_letter,     EvidentialRole.proves_employment_tenure,"formal")],
    # E — press profile + employer letter (executive)
    [(DocType.bloomberg_article,   EvidentialRole.corroborates_narrative,  "press"),
     (DocType.employer_letter,     EvidentialRole.proves_employment_tenure,"formal")],
]

_INHERITANCE_BUNDLES: list[_BundleOption] = [
    # A — probate + bank transfer
    [(DocType.probate_grant,               EvidentialRole.proves_inheritance_right,"formal"),
     (DocType.bank_transfer_confirmation,  EvidentialRole.proves_transaction,      "formal")],
    # B — will + solicitor letter
    [(DocType.will_extract,                EvidentialRole.proves_inheritance_right,"legal"),
     (DocType.solicitor_letter,            EvidentialRole.corroborates_narrative,  "formal")],
    # C — full chain: probate + will + bank
    [(DocType.probate_grant,               EvidentialRole.proves_inheritance_right,"formal"),
     (DocType.will_extract,                EvidentialRole.proves_inheritance_right,"legal"),
     (DocType.bank_transfer_confirmation,  EvidentialRole.proves_transaction,      "formal")],
    # D — solicitor + bank transfer (simpler)
    [(DocType.solicitor_letter,            EvidentialRole.corroborates_narrative,  "formal"),
     (DocType.bank_transfer_confirmation,  EvidentialRole.proves_transaction,      "formal")],
]

_GIFT_BUNDLES: list[_BundleOption] = [
    # A — gift letter + bank transfer
    [(DocType.gift_letter,                EvidentialRole.proves_gift_transfer,"formal"),
     (DocType.bank_transfer_confirmation, EvidentialRole.proves_transaction,  "formal")],
    # B — gift deed only (formal single-doc)
    [(DocType.gift_deed,                  EvidentialRole.proves_gift_transfer,"legal")],
    # C — full chain: deed + solicitor + bank
    [(DocType.gift_deed,                  EvidentialRole.proves_gift_transfer,"legal"),
     (DocType.solicitor_letter,           EvidentialRole.corroborates_narrative,"formal"),
     (DocType.bank_transfer_confirmation, EvidentialRole.proves_transaction,  "formal")],
    # D — solicitor + bank (no deed)
    [(DocType.solicitor_letter,           EvidentialRole.corroborates_narrative,"formal"),
     (DocType.bank_transfer_confirmation, EvidentialRole.proves_transaction,  "formal")],
]

_BUSINESS_BUNDLES: list[_BundleOption] = [
    # A — CoI + accounts + distribution statement
    [(DocType.companies_house_filing,    EvidentialRole.proves_entity_ownership, "regulatory"),
     (DocType.company_accounts,          EvidentialRole.proves_income_amount,    "formal"),
     (DocType.distribution_statement,    EvidentialRole.proves_transaction,      "formal")],
    # B — CoI + bank statement
    [(DocType.companies_house_filing,    EvidentialRole.proves_entity_ownership, "regulatory"),
     (DocType.bank_statement,            EvidentialRole.proves_transaction,      "formal")],
    # C — accounts + bank transfer
    [(DocType.company_accounts,          EvidentialRole.proves_income_amount,    "formal"),
     (DocType.bank_transfer_confirmation,EvidentialRole.proves_transaction,      "formal")],
    # D — full chain: CoI + accounts + bank + distribution
    [(DocType.companies_house_filing,    EvidentialRole.proves_entity_ownership, "regulatory"),
     (DocType.company_accounts,          EvidentialRole.proves_income_amount,    "formal"),
     (DocType.bank_statement,            EvidentialRole.proves_transaction,      "formal"),
     (DocType.distribution_statement,    EvidentialRole.proves_transaction,      "formal")],
    # E — press + CoI (high-profile exit)
    [(DocType.bloomberg_article,         EvidentialRole.corroborates_narrative,  "press"),
     (DocType.companies_house_filing,    EvidentialRole.proves_entity_ownership, "regulatory")],
]

_BUNDLE_OPTIONS: dict[SowType, list[_BundleOption]] = {
    SowType.employment:       _EMPLOYMENT_BUNDLES,
    SowType.inheritance:      _INHERITANCE_BUNDLES,
    SowType.gift:             _GIFT_BUNDLES,
    SowType.business_profits: _BUSINESS_BUNDLES,
}


def _pick_bundle(sow_type: SowType, seed_str: str) -> _BundleOption:
    options = _BUNDLE_OPTIONS[sow_type]
    return options[h(seed_str) % len(options)]


# ---------------------------------------------------------------------------
# Mandatory-facts builders per DocType
# ---------------------------------------------------------------------------

def _facts_payslip(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Employee Name":         ctx["employee_name"],
        "Employer":              ctx["employer_name"],
        "PAYE Reference":        ctx["paye_ref"],
        "National Insurance No": ctx["ni_number"],
        "Tax Year":              ctx["tax_year_label"],
        "Total Gross Pay":       fmt_currency(c, ctx["gross"]),
        "Income Tax":            fmt_currency(c, ctx["income_tax"]),
        "NI Contributions":      fmt_currency(c, ctx["ni"]),
        "Net Pay":               fmt_currency(c, ctx["net"]),
        "Employment Period":     f"{ctx['period_start']} to {ctx['period_end']}",
    }


def _facts_employer_letter(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Employee":            ctx["employee_name"],
        "Employer":            ctx["employer_name"],
        "Employment Period":   f"{ctx['period_start']} to {ctx['period_end']}",
        "Annual Gross Salary": fmt_currency(c, ctx["gross"]),
        "Job Title":           ctx["job_title"],
        "Reference":           ctx["letter_ref"],
    }


def _facts_employment_contract(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Employee":      ctx["employee_name"],
        "Employer":      ctx["employer_name"],
        "Start Date":    ctx["period_start"],
        "Job Title":     ctx["job_title"],
        "Annual Salary": fmt_currency(c, ctx["gross"]),
        "Contract Ref":  ctx["contract_ref"],
    }


def _facts_bank_statement(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Account Holder": ctx["account_holder"],
        "Bank":           ctx["bank"],
        "Sort Code":      ctx["sort_code"],
        "Account No":     f"****{ctx['account_last4']}",
        "Period":         f"{ctx['period_start']} to {ctx['period_end']}",
        "Total Credits":  fmt_currency(c, ctx["primary_amount"]),
    }


def _facts_bank_transfer(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Sender":        ctx["sender_name"],
        "Recipient":     ctx["recipient_name"],
        "Amount":        fmt_currency(c, ctx["primary_amount"]),
        "Transfer Date": ctx["transfer_date"],
        "Reference":     ctx["transfer_ref"],
        "Bank":          ctx["bank"],
    }


def _facts_probate_grant(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Deceased":          ctx["deceased_name"],
        "Executor":          ctx["executor_name"],
        "Date of Death":     ctx["death_date"],
        "Grant Date":        ctx["grant_date"],
        "Gross Estate":      fmt_currency(c, ctx["primary_amount"]),
        "Probate Reference": ctx["probate_ref"],
    }


def _facts_will_extract(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Testator":          ctx["deceased_name"],
        "Beneficiary":       ctx["executor_name"],
        "Bequest Amount":    fmt_currency(c, ctx["primary_amount"]),
        "Will Date":         ctx["will_date"],
        "Probate Reference": ctx["probate_ref"],
    }


def _facts_solicitor_letter(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Client":   ctx["subject_name"],
        "Firm":     ctx["firm_name"],
        "Our Ref":  ctx["matter_ref"],
        "Matter":   ctx["matter_description"],
        "Amount":   fmt_currency(c, ctx["primary_amount"]),
        "Date":     ctx["letter_date"],
    }


def _facts_gift_letter(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Donor":        ctx["donor_name"],
        "Recipient":    ctx["recipient_name"],
        "Relationship": ctx["donor_relationship"],
        "Gift Amount":  fmt_currency(c, ctx["primary_amount"]),
        "Gift Date":    ctx["gift_date"],
        "Purpose":      "personal wealth / no expectation of repayment",
    }


def _facts_gift_deed(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Donor":          ctx["donor_name"],
        "Donee":          ctx["recipient_name"],
        "Gift Amount":    fmt_currency(c, ctx["primary_amount"]),
        "Execution Date": ctx["gift_date"],
        "Deed Reference": ctx["deed_ref"],
    }


def _facts_companies_house(ctx: dict) -> dict[str, str]:
    return {
        "Company Name":       ctx["company_name"],
        "Company Number":     ctx["company_number"],
        "Director":           ctx["director_name"],
        "Incorporation Date": ctx["incorporation_date"],
        "Registered Address": ctx["company_address"],
    }


def _facts_company_accounts(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Company":               ctx["company_name"],
        "Company Number":        ctx["company_number"],
        "Accounting Period":     f"{ctx['period_start']} to {ctx['period_end']}",
        "Director":              ctx["director_name"],
        "Distribution / Profit": fmt_currency(c, ctx["primary_amount"]),
    }


def _facts_distribution_statement(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Company":          ctx["company_name"],
        "Shareholder":      ctx["director_name"],
        "Net Distribution": fmt_currency(c, ctx["primary_amount"]),
        "Payment Date":     ctx["period_end"],
        "Tax Year":         ctx["tax_year_label"],
        "Reference":        ctx["dist_ref"],
    }


def _facts_press(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "Subject":           ctx["subject_name"],
        "Amount (approx)":   fmt_currency(c, ctx["primary_amount"]),
        "Publication Date":  ctx["event_date"],
    }


def _facts_email_thread(ctx: dict) -> dict[str, str]:
    c = ctx["currency"]
    return {
        "From":    ctx["sender_name"],
        "To":      ctx["recipient_name"],
        "Subject": ctx["email_subject"],
        "Amount":  fmt_currency(c, ctx["primary_amount"]),
        "Date":    ctx["event_date"],
    }


_FACTS_BUILDERS: dict[DocType, callable] = {
    DocType.payslip:                    _facts_payslip,
    DocType.employer_letter:            _facts_employer_letter,
    DocType.employment_contract:        _facts_employment_contract,
    DocType.bank_statement:             _facts_bank_statement,
    DocType.bank_transfer_confirmation: _facts_bank_transfer,
    DocType.probate_grant:              _facts_probate_grant,
    DocType.will_extract:               _facts_will_extract,
    DocType.solicitor_letter:           _facts_solicitor_letter,
    DocType.gift_letter:                _facts_gift_letter,
    DocType.gift_deed:                  _facts_gift_deed,
    DocType.companies_house_filing:     _facts_companies_house,
    DocType.company_accounts:           _facts_company_accounts,
    DocType.distribution_statement:     _facts_distribution_statement,
    DocType.bloomberg_article:          _facts_press,
    DocType.ft_article:                 _facts_press,
    DocType.email_thread:               _facts_email_thread,
}


# ---------------------------------------------------------------------------
# Per-SoW-type claim context builders
# ---------------------------------------------------------------------------

def _employment_claim_ctx(
    profile: Profile, claim: Claim, events: list[Event], spec: ScenarioSpec,
) -> dict:
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
    start      = emp_events[0].date
    end        = emp_events[-1].date
    ty_start   = date(start.year, 4, 6)
    ty_end     = date(end.year + 1, 4, 5)
    seed_str   = profile.client_id + claim.claim_id

    _JOB_TITLES = {
        "finance":               ["Analyst", "Associate", "Vice President", "Director", "Managing Director"],
        "technology":            ["Software Engineer", "Senior Engineer", "Principal Engineer", "Engineering Manager"],
        "real_estate":           ["Negotiator", "Senior Negotiator", "Associate Director", "Director"],
        "professional_services": ["Associate", "Senior Associate", "Manager", "Senior Manager", "Partner"],
    }
    titles = _JOB_TITLES.get(profile.industry or "finance", _JOB_TITLES["finance"])
    job_title = titles[min(role_index, len(titles) - 1)]

    return {
        "subject_name":    profile.name,
        "employee_name":   profile.name,
        "employer_name":   employer["name"],
        "employer_address":employer["address"],
        "paye_ref":        f"{100 + role_index:03d}/A{spec.seed % 10000:04d}",
        "ni_number":       f"AB{abs(hash(profile.client_id + 'ni')) % 1000000:06d}C",
        "gross":           gross,
        "income_tax":      income_tax,
        "ni":              ni,
        "net":             net,
        "period_start":    _fmt_date(start),
        "period_end":      _fmt_date(end),
        "tax_year_start":  ty_start.isoformat(),
        "tax_year_end":    ty_end.isoformat(),
        "tax_year_label":  f"{start.year}/{str(end.year + 1)[2:]}",
        "job_title":       job_title,
        "letter_ref":      _ref(seed_str, "HR"),
        "contract_ref":    _ref(seed_str, "EC"),
        "primary_amount":  gross,
        "currency":        spec.currency,
        "account_holder":  profile.name,
        "bank":            bank_name(profile.client_id + str(role_index)),
        "sort_code":       sort_code(profile.client_id + str(role_index)),
        "account_last4":   account_last4(profile.client_id + str(role_index)),
        "sender_name":     employer["name"],
        "recipient_name":  profile.name,
        "transfer_date":   _fmt_date(end),
        "transfer_ref":    _ref(seed_str, "TRF"),
        "sender_name_em":  employer["name"],
        "recipient_name_em": profile.name,
        "email_subject":   f"Salary confirmation — {profile.name}",
        "event_date":      _fmt_date(end),
    }


def _inheritance_claim_ctx(
    profile: Profile, claim: Claim, events: list[Event], spec: ScenarioSpec,
) -> dict:
    event_by_id = {e.event_id: e for e in events}
    evt = event_by_id[claim.covered_event_ids[0]]
    seed_str = profile.client_id + evt.event_id
    dec  = deceased_name(seed_str)
    firm = solicitor_firm(seed_str)
    probate_d = date(evt.date.year, evt.date.month, min(evt.date.day + 60, 28))
    will_d    = date(evt.date.year - 2, 1, 15)

    return {
        "subject_name":      profile.name,
        "deceased_name":     dec,
        "executor_name":     profile.name,
        "death_date":        _fmt_date(evt.date),
        "grant_date":        _fmt_date(probate_d),
        "will_date":         _fmt_date(will_d),
        "probate_ref":       _ref(seed_str, "PROB"),
        "primary_amount":    claim.amount,
        "currency":          spec.currency,
        "sender_name":       dec,
        "recipient_name":    profile.name,
        "transfer_date":     _fmt_date(probate_d),
        "transfer_ref":      _ref(seed_str, "TRF"),
        "bank":              bank_name(seed_str),
        "sort_code":         sort_code(seed_str),
        "account_last4":     account_last4(seed_str),
        "account_holder":    profile.name,
        "period_start":      _fmt_date(evt.date),
        "period_end":        _fmt_date(probate_d),
        "firm_name":         firm["name"],
        "matter_ref":        _ref(seed_str, "MAT"),
        "matter_description":f"Estate of {dec} — Beneficiary Distribution",
        "letter_date":       _fmt_date(probate_d),
        "event_date":        _fmt_date(evt.date),
        "sender_name_em":    dec,
        "recipient_name_em": profile.name,
        "email_subject":     f"Estate distribution — {profile.name}",
    }


def _gift_claim_ctx(
    profile: Profile, claim: Claim, events: list[Event], spec: ScenarioSpec,
) -> dict:
    event_by_id = {e.event_id: e for e in events}
    evt = event_by_id[claim.covered_event_ids[0]]
    seed_str = profile.client_id + evt.event_id
    dn   = donor_name(seed_str)
    dr   = donor_relationship(seed_str)
    firm = solicitor_firm(seed_str)

    return {
        "subject_name":       profile.name,
        "donor_name":         dn,
        "donor_relationship": dr,
        "recipient_name":     profile.name,
        "gift_date":          _fmt_date(evt.date),
        "deed_ref":           _ref(seed_str, "DEED"),
        "primary_amount":     claim.amount,
        "currency":           spec.currency,
        "sender_name":        dn,
        "transfer_date":      _fmt_date(evt.date),
        "transfer_ref":       _ref(seed_str, "TRF"),
        "bank":               bank_name(seed_str),
        "sort_code":          sort_code(seed_str),
        "account_last4":      account_last4(seed_str),
        "account_holder":     profile.name,
        "period_start":       _fmt_date(evt.date),
        "period_end":         _fmt_date(evt.date),
        "firm_name":          firm["name"],
        "matter_ref":         _ref(seed_str, "MAT"),
        "matter_description": f"Gift from {dn} to {profile.name}",
        "letter_date":        _fmt_date(evt.date),
        "event_date":         _fmt_date(evt.date),
        "sender_name_em":     dn,
        "recipient_name_em":  profile.name,
        "email_subject":      f"Gift transfer — {profile.name}",
    }


def _business_claim_ctx(
    profile: Profile, claim: Claim, events: list[Event], spec: ScenarioSpec,
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
    start    = biz_events[0].date
    end      = biz_events[-1].date
    inc_date = _fmt_date(start - timedelta(days=730))
    seed_str = profile.client_id + claim.claim_id

    return {
        "subject_name":       profile.name,
        "company_name":       co_name,
        "company_number":     co_num,
        "company_address":    co_addr,
        "director_name":      profile.name,
        "incorporation_date": inc_date,
        "period_start":       _fmt_date(start),
        "period_end":         _fmt_date(end),
        "tax_year_label":     f"{end.year}/{str(end.year + 1)[2:]}",
        "primary_amount":     claim.amount,
        "currency":           spec.currency,
        "dist_ref":           _ref(seed_str, "DIST"),
        "account_holder":     profile.name,
        "bank":               bank_name(biz_id),
        "sort_code":          sort_code(biz_id),
        "account_last4":      account_last4(biz_id),
        "sender_name":        co_name,
        "recipient_name":     profile.name,
        "transfer_date":      _fmt_date(end),
        "transfer_ref":       _ref(seed_str, "TRF"),
        "event_date":         _fmt_date(end),
        "sender_name_em":     co_name,
        "recipient_name_em":  profile.name,
        "email_subject":      f"Distribution payment — {co_name}",
    }


_CTX_BUILDERS = {
    SowType.employment:       _employment_claim_ctx,
    SowType.inheritance:      _inheritance_claim_ctx,
    SowType.gift:             _gift_claim_ctx,
    SowType.business_profits: _business_claim_ctx,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan_bundles(
    profile: Profile,
    events: list[Event],
    claims: list[Claim],
    spec: ScenarioSpec,
    rng: np.random.Generator,
) -> list[CorroborationBundle]:
    """Return one CorroborationBundle per Claim.

    Bundle composition is determined deterministically from the seed.
    """
    bundles: list[CorroborationBundle] = []
    doc_counter = 1

    for claim in claims:
        ctx_builder = _CTX_BUILDERS.get(claim.sow_type)
        if ctx_builder is None:
            continue
        ctx = ctx_builder(profile, claim, events, spec)
        if not ctx:
            continue

        bundle_seed = f"{spec.seed}_{claim.claim_id}"
        bundle_option = _pick_bundle(claim.sow_type, bundle_seed)
        bundle_id = f"bundle_{spec.seed:08d}_{claim.claim_id}"

        doc_specs: list[DocumentSpec] = []
        for doc_type, role, style in bundle_option:
            doc_id = f"doc_{doc_type.value}_{spec.seed:08d}_{doc_counter:04d}"
            doc_counter += 1

            # email threads use email-specific sender/recipient keys
            if doc_type == DocType.email_thread:
                email_ctx = {**ctx,
                             "sender_name":    ctx.get("sender_name_em",    ctx.get("sender_name", "")),
                             "recipient_name": ctx.get("recipient_name_em", ctx.get("recipient_name", ""))}
                mandatory_facts = _facts_email_thread(email_ctx)
            else:
                facts_builder = _FACTS_BUILDERS.get(doc_type)
                mandatory_facts = facts_builder(ctx) if facts_builder else {}

            mandatory_facts["Document ID"] = doc_id

            doc_specs.append(DocumentSpec(
                doc_id=doc_id,
                doc_type=doc_type,
                evidential_role=role,
                mandatory_facts=mandatory_facts,
                style_hint=style,
                source_event_ids=list(claim.covered_event_ids),
            ))

        bundles.append(CorroborationBundle(
            bundle_id=bundle_id,
            claim_id=claim.claim_id,
            doc_specs=doc_specs,
        ))

    return bundles
