"""Stage 8 — Format-aware surface realization.

Dispatches each DocumentPlan to the right context builder + template family
based on its FormatType.  Numbers are always injected from template_context
(the fact layer); narrative prose is template-generated in Phase 3 (LLM
calls are stubbed behind USE_LLM=1).

Entry point: `realize_all(plans, documents)` — mutates Document.pages in-place.
"""
from __future__ import annotations

import hashlib
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from sow_synth.formats import FORMATS, FormatSpec
from sow_synth.formats.flavor import (
    account_last4, bank_name, bloomberg_byline, company_address,
    company_number, company_name as gen_company_name,
    dateline_city, deceased_name, donor_name, donor_relationship,
    ft_byline, solicitor_firm, solicitor_ref, sort_code,
)
from sow_synth.models import Document, KeyValue, OcrLine, OcrPage, OcrWord

_TEMPLATES_ROOT = Path(__file__).parent / "templates"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_ROOT)),
    undefined=StrictUndefined,
    autoescape=False,
)

_TWO = Decimal("0.01")
_MONTHS = ["April","May","June","July","August","September",
           "October","November","December","January","February","March"]

# ---------------------------------------------------------------------------
# Helpers shared across context builders
# ---------------------------------------------------------------------------

def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _pick(items: list, seed: str) -> str:
    return items[_h(seed) % len(items)]


def _fmt(amount: Decimal) -> str:
    return f"{amount:,.2f}"


def _words(amount: Decimal) -> str:
    """Very simplified number-to-words for amounts (whole pounds only)."""
    n = int(amount)
    if n >= 1_000_000:
        m = n // 1_000_000
        r = (n % 1_000_000) // 1000
        return f"{m} million{' ' + str(r) + ' thousand' if r else ''} pounds"
    if n >= 1_000:
        k = n // 1000
        r = n % 1000
        return f"{k} thousand{' ' + str(r) if r else ''} pounds"
    return f"{n} pounds"


def _monthly_split(annual: Decimal, n: int = 12) -> list[Decimal]:
    base = (annual / n).quantize(_TWO, rounding=ROUND_HALF_UP)
    months = [base] * n
    months[-1] += annual - base * n
    return months


def _n_pages(spec: FormatSpec, ctx: dict) -> int:
    lo, hi = spec.min_pages, spec.max_pages
    if lo == hi:
        return lo
    # Use gross amount to vary page count — higher complexity → more pages
    try:
        amount = Decimal(str(ctx.get("primary_amount", ctx.get("gross_pay", "0"))))
        thresholds = [Decimal("50000"), Decimal("150000"), Decimal("500000")]
        for i, t in enumerate(thresholds[:hi - lo]):
            if amount < t:
                return lo + i
        return hi
    except Exception:
        return lo


# ---------------------------------------------------------------------------
# Context builders per format type
# ---------------------------------------------------------------------------

_JOB_TITLES = {
    "finance": ["Analyst", "Associate", "Vice President", "Director", "Managing Director"],
    "technology": ["Software Engineer", "Senior Engineer", "Principal Engineer", "Engineering Manager"],
    "real_estate": ["Negotiator", "Senior Negotiator", "Associate Director", "Director"],
    "professional_services": ["Associate", "Senior Associate", "Manager", "Senior Manager", "Partner"],
}
_DEPARTMENTS = {
    "finance": ["Equities", "Fixed Income", "M&A", "Risk", "Compliance"],
    "technology": ["Platform Engineering", "Product", "Data", "Infrastructure"],
    "real_estate": ["Residential Sales", "Commercial", "Asset Management"],
    "professional_services": ["Advisory", "Transactions", "Restructuring", "Tax"],
}


def _n_pages_for_gross(gross: Decimal) -> int:
    if gross < Decimal("40000"):
        return 1
    if gross < Decimal("80000"):
        return 2
    if gross < Decimal("150000"):
        return 3
    return 4


def _build_page1_ctx(ctx: dict) -> dict:
    gross = Decimal(ctx["gross_pay"])
    ni_er = (gross * Decimal("0.138")).quantize(_TWO)
    signed_date = ctx["tax_year_end"][:7] + "-30"
    return {**ctx, "ni_er": str(ni_er), "tax_code": ctx.get("tax_code", "1257L"),
            "signed_date": signed_date, "total_pages": ctx["total_pages"]}


def _build_page2_ctx(ctx: dict) -> dict:
    gross = Decimal(ctx["gross_pay"])
    tax   = Decimal(ctx["income_tax"])
    ni    = Decimal(ctx["ni_contributions"])
    net   = Decimal(ctx["net_pay"])
    currency = ctx["currency"]

    gross_months = _monthly_split(gross)
    tax_months   = _monthly_split(tax)
    ni_months    = _monthly_split(ni)
    net_months   = _monthly_split(net)

    monthly_rows = [
        {"month": m,
         "gross": f"{currency} {gm:,.2f}", "tax": f"{currency} {tm:,.2f}",
         "ni": f"{currency} {nm:,.2f}", "net": f"{currency} {ntm:,.2f}",
         "line": f"{m:<17}{currency} {gm:>10,.2f}  {currency} {tm:>9,.2f}  {currency} {nm:>9,.2f}  {currency} {ntm:>9,.2f}"}
        for m, gm, tm, nm, ntm in zip(_MONTHS, gross_months, tax_months, ni_months, net_months)
    ]

    seed_int = _h(ctx["doc_id"])
    benefit_options = [
        ("Private Medical Insurance", Decimal("1500.00")),
        ("Company Car Allowance", Decimal("5000.00")),
        ("Life Assurance", Decimal("200.00")),
        ("Gym Membership", Decimal("600.00")),
        ("Season Ticket Loan", Decimal("2400.00")),
    ]
    n_benefits = seed_int % 3
    benefits = [{"description": benefit_options[(seed_int + i) % len(benefit_options)][0],
                 "value": str(benefit_options[(seed_int + i) % len(benefit_options)][1])}
                for i in range(n_benefits)]
    benefits_total = str(sum(benefit_options[(seed_int + i) % len(benefit_options)][1]
                             for i in range(n_benefits)))

    return {**ctx, "monthly_rows": monthly_rows,
            "ytd_gross": f"{currency} {gross:,.2f}", "ytd_tax": f"{currency} {tax:,.2f}",
            "ytd_ni": f"{currency} {ni:,.2f}", "ytd_net": f"{currency} {net:,.2f}",
            "benefits": benefits, "benefits_total": benefits_total,
            "total_pages": ctx["total_pages"]}


def _build_page3_ctx(ctx: dict) -> dict:
    gross = Decimal(ctx["gross_pay"])
    income_tax = Decimal(ctx["income_tax"])
    personal_all = Decimal("12570.00")
    taxable = max(gross - personal_all, Decimal("0"))
    uel = Decimal("50270.00")
    lel = Decimal("6396.00")
    niable = max(min(gross, uel) - lel, Decimal("0"))
    ni_er = (gross * Decimal("0.138")).quantize(_TWO)
    effective_rate = f"{(income_tax / gross * 100).quantize(Decimal('0.1'))}%"

    bands = []
    remaining = taxable
    if remaining > 0:
        b = min(remaining, Decimal("37700.00"))
        bands.append({"label": "Basic rate band (20%)", "amount": f"{b:,.2f}", "rate": "20%"})
        remaining -= b
    if remaining > 0:
        b = min(remaining, Decimal("87440.00"))
        bands.append({"label": "Higher rate band (40%)", "amount": f"{b:,.2f}", "rate": "40%"})
        remaining -= b
    if remaining > 0:
        bands.append({"label": "Additional rate band (45%)", "amount": f"{remaining:,.2f}", "rate": "45%"})

    currency = ctx["currency"]
    return {**ctx, "personal_allowance": f"{personal_all:,.2f}",
            "taxable_income": f"{taxable:,.2f}", "tax_bands": bands,
            "effective_rate": effective_rate, "ni_category": "A",
            "lel": f"{lel:,.2f}", "uel": f"{uel:,.2f}", "niable": f"{niable:,.2f}",
            "ni_ee_rate": "12%", "ni_er_rate": "13.8%", "ni_er": str(ni_er),
            "total_pages": ctx["total_pages"]}


def _build_page4_ctx(ctx: dict) -> dict:
    seed_str = ctx["doc_id"]
    industry = ctx.get("industry", "finance")
    role_idx = ctx.get("role_index", 0)
    titles = _JOB_TITLES.get(industry, _JOB_TITLES["finance"])
    depts  = _DEPARTMENTS.get(industry, _DEPARTMENTS["finance"])
    title  = titles[min(role_idx, len(titles) - 1)]
    dept   = _pick(depts, seed_str + "dept")

    sc = (f"{_pick(['20','30','40','50','60','77'], seed_str+'sc1')}-"
          f"{_pick(['11','22','33','44','55','66'], seed_str+'sc2')}-"
          f"{_pick(['00','01','10','11','12','13'], seed_str+'sc3')}")
    acct_last4  = str(_h(seed_str + "acct") % 10000).zfill(4)
    payroll_num = f"PR{_h(seed_str) % 1000000:06d}"

    gross      = Decimal(ctx["gross_pay"])
    pension_ee = (gross * Decimal("0.05")).quantize(_TWO)
    pension_er = (gross * Decimal("0.08")).quantize(_TWO)
    scheme     = _pick(["Nest Workplace Pension", "Scottish Widows Group Pension",
                         "Aviva Workplace Pension", "Legal & General WorkSave"],
                        seed_str + "pension")

    return {**ctx, "job_title": title, "department": dept,
            "role_start": ctx.get("tax_year_start", ""), "payroll_number": payroll_num,
            "payment_method": "BACS", "sort_code": sc, "account_last4": acct_last4,
            "pension_rows": [{"scheme": scheme,
                               "employee_contrib": f"{pension_ee:,.2f}",
                               "employer_contrib": f"{pension_er:,.2f}"}],
            "pension_ee_total": f"{pension_ee:,.2f}",
            "pension_er_total": f"{pension_er:,.2f}",
            "total_pages": ctx["total_pages"]}


def _ctx_payslip(ctx: dict) -> list[dict]:
    gross = Decimal(ctx["gross_pay"])
    n = _n_pages_for_gross(gross)
    base = {**ctx, "total_pages": n}
    builders = [_build_page1_ctx, _build_page2_ctx, _build_page3_ctx, _build_page4_ctx]
    return [builders[i](base) for i in range(n)]


def _ctx_bank_statement(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    currency = ctx["currency"]
    primary = Decimal(ctx["primary_amount"])
    period_start_str = ctx.get("period_start", "")
    period_end_str   = ctx.get("period_end", "")

    # Build transaction rows
    raw_txs: list[dict] = ctx.get("transactions_raw", [])
    if not raw_txs:
        # fallback: generate monthly salary credits
        months = _monthly_split(primary)
        raw_txs = [
            {"date": f"28 {m} {ctx.get('year', '2022')[:4]}",
             "description": f"SALARY — {ctx.get('employer_name','EMPLOYER')[:20].upper()}",
             "credit": _fmt(amt), "debit": ""}
            for m, amt in zip(_MONTHS, months)
        ]

    running = Decimal(ctx.get("opening_balance_amount", "5000.00"))
    opening = running
    total_credits = Decimal("0")
    total_debits  = Decimal("0")
    tx_lines = []
    for tx in raw_txs:
        credit = Decimal(tx["credit"].replace(",","")) if tx.get("credit") else Decimal("0")
        debit  = Decimal(tx["debit"].replace(",",""))  if tx.get("debit")  else Decimal("0")
        running += credit - debit
        total_credits += credit
        total_debits  += debit
        cr_str = _fmt(credit) if credit else ""
        db_str = _fmt(debit)  if debit  else ""
        tx_lines.append({
            "line": f"{tx['date']:<13}{tx['description']:<41}{db_str:>12}  {cr_str:>12}  {_fmt(running):>12}"
        })

    base = {
        **ctx,
        "bank_name": ctx.get("bank_name", bank_name(seed)),
        "account_holder": ctx["subject_name"],
        "account_last4": ctx.get("account_last4", account_last4(seed)),
        "sort_code": ctx.get("sort_code", sort_code(seed)),
        "branch_address": ctx.get("branch_address", "PO Box 1000, London EC2V 8RT"),
        "period_start": period_start_str,
        "period_end": period_end_str,
        "transactions": tx_lines,
        "opening_balance": _fmt(opening),
        "total_credits": _fmt(total_credits),
        "total_debits": _fmt(total_debits),
        "closing_balance": _fmt(opening + total_credits - total_debits),
        "statement_ref": f"STMT/{_h(seed) % 10000000:07d}",
        "total_pages": 1,
    }
    return [base]


def _ctx_bank_transfer(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    amount = Decimal(ctx["primary_amount"])
    base = {
        **ctx,
        "bank_name": bank_name(seed),
        "transfer_ref": f"TRF{_h(seed+'tr') % 10000000:07d}",
        "transfer_date": ctx.get("transfer_date", ctx.get("event_date", "")),
        "sender_name": ctx.get("sender_name", ctx["subject_name"]),
        "sender_sort": sort_code(seed + "ss"),
        "sender_last4": account_last4(seed + "sl"),
        "recipient_name": ctx.get("recipient_name", ctx["subject_name"]),
        "recipient_sort": sort_code(seed + "rs"),
        "recipient_last4": account_last4(seed + "rl"),
        "transfer_amount": _fmt(amount),
        "payment_type": "CHAPS" if amount > Decimal("10000") else "BACS",
        "payment_reference": ctx.get("payment_reference", f"REF{_h(seed+'pr') % 100000:05d}"),
        "total_pages": 1,
    }
    return [base]


def _ctx_probate_grant(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    estate = Decimal(ctx["primary_amount"])
    debts  = (estate * Decimal("0.05")).quantize(_TWO)
    net    = estate - debts
    iht    = max(Decimal("0"), (net - Decimal("325000")) * Decimal("0.40")).quantize(_TWO)
    base = {
        **ctx,
        "deceased_name": ctx.get("deceased_name", deceased_name(seed)),
        "executor_name": ctx["subject_name"],
        "executor_address": ctx.get("subject_address", ""),
        "probate_ref": f"PROB{_h(seed+'pr') % 1000000:06d}",
        "grant_date": ctx.get("grant_date", ctx.get("event_date", "")),
        "death_date": ctx.get("death_date", ctx.get("event_date", "")),
        "domicile": ctx.get("domicile", "England and Wales"),
        "gross_estate": _fmt(estate),
        "debts": _fmt(debts),
        "net_estate": _fmt(net),
        "iht_paid": _fmt(iht),
        "will_date": ctx.get("will_date", ""),
        "total_pages": 1,
    }
    return [base]


def _ctx_will_extract(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    amount = Decimal(ctx["primary_amount"])
    sf = solicitor_firm(seed)
    base = {
        **ctx,
        "deceased_name": ctx.get("deceased_name", deceased_name(seed)),
        "beneficiary_name": ctx["subject_name"],
        "beneficiary_relationship": ctx.get("donor_relationship", ""),
        "bequest_amount": _fmt(amount),
        "bequest_amount_words": _words(amount),
        "will_date": ctx.get("will_date", ""),
        "extract_pages": "3-5",
        "bequest_clause": str(_h(seed+"bc") % 8 + 3),
        "residue_clause": str(_h(seed+"rc") % 8 + 10),
        "solicitor_firm": sf["name"],
        "solicitor_ref": solicitor_ref(seed),
        "total_pages": 1,
    }
    return [base]


def _ctx_gift_deed(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    amount = Decimal(ctx["primary_amount"])
    dn = ctx.get("donor_name", donor_name(seed))
    base = {
        **ctx,
        "donor_name": dn,
        "donor_address": ctx.get("donor_address", "Address provided separately"),
        "recipient_name": ctx["subject_name"],
        "recipient_address": ctx.get("subject_address", ""),
        "deed_date": ctx.get("event_date", ""),
        "gift_amount": _fmt(amount),
        "gift_amount_words": _words(amount),
        "witness_name": _pick(["Sarah Mitchell","James Okafor","Claire Dunmore"], seed+"wn"),
        "witness_address": "Address provided separately",
        "total_pages": 1,
    }
    return [base]


def _ctx_share_purchase(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    price = Decimal(ctx["primary_amount"])
    shares = _h(seed + "sh") % 900000 + 100000
    pps = (price / shares).quantize(_TWO)
    ev  = (price * Decimal("1.15")).quantize(_TWO)
    buyer = gen_company_name(seed + "buyer")
    base = {
        **ctx,
        "seller_name": ctx["subject_name"],
        "seller_address": ctx.get("subject_address", ""),
        "buyer_name": buyer,
        "buyer_address": company_address(seed + "baddr"),
        "company_name": ctx.get("company_name", gen_company_name(seed)),
        "company_number": ctx.get("company_number", company_number(seed)),
        "shares_sold": f"{shares:,}",
        "share_percentage": str(_h(seed+"sp") % 80 + 20),
        "purchase_price": _fmt(price),
        "purchase_price_words": _words(price),
        "price_per_share": _fmt(pps),
        "enterprise_value": _fmt(ev),
        "agreement_date": ctx.get("event_date", ""),
        "completion_date": ctx.get("event_date", ""),
        "total_pages": 2,
    }
    return [base, {**base, "total_pages": 2}]


def _ctx_company_accounts(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    dist = Decimal(ctx["primary_amount"])
    turnover = (dist * Decimal(str(_h(seed+"tr") % 3 + 4))).quantize(_TWO)
    cos      = (turnover * Decimal("0.35")).quantize(_TWO)
    gp       = turnover - cos
    admin    = (gp * Decimal("0.30")).quantize(_TWO)
    op       = gp - admin
    interest = (op * Decimal("0.02")).quantize(_TWO)
    pbt      = op + interest
    tax      = (pbt * Decimal("0.19")).quantize(_TWO)
    pat      = pbt - tax
    retained = (pat - dist).quantize(_TWO)
    base = {
        **ctx,
        "company_name": ctx.get("company_name", gen_company_name(seed)),
        "company_number": ctx.get("company_number", company_number(seed)),
        "director_name": ctx["subject_name"],
        "year_end_date": ctx.get("period_end", ""),
        "approval_date": ctx.get("period_end", ""),
        "turnover": _fmt(turnover),
        "cost_of_sales": _fmt(cos),
        "gross_profit": _fmt(gp),
        "admin_expenses": _fmt(admin),
        "operating_profit": _fmt(op),
        "interest_receivable": _fmt(interest),
        "profit_before_tax": _fmt(pbt),
        "corporation_tax": _fmt(tax),
        "profit_after_tax": _fmt(pat),
        "retained_bf": _fmt(dist * Decimal("2")),
        "retained_cf": _fmt(retained + dist * Decimal("2")),
        "distribution_amount": _fmt(dist),
        "total_pages": 2,
    }
    return [base, {**base}]


def _ctx_distribution_statement(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    gross = Decimal(ctx["primary_amount"])
    tc_rate = Decimal("10")
    tc = (gross * tc_rate / 100).quantize(_TWO)
    net = gross - tc
    base = {
        **ctx,
        "company_name": ctx.get("company_name", gen_company_name(seed)),
        "company_number": ctx.get("company_number", company_number(seed)),
        "shareholder_name": ctx["subject_name"],
        "share_percentage": str(_h(seed+"sp") % 80 + 20),
        "shares_held": f"{_h(seed+'sh') % 900000 + 100000:,}",
        "distribution_date": ctx.get("event_date", ""),
        "tax_year_label": ctx.get("tax_year_label", ""),
        "gross_distribution": _fmt(gross),
        "tax_credit_rate": str(tc_rate),
        "tax_credit": _fmt(tc),
        "net_distribution": _fmt(net),
        "account_last4": account_last4(seed),
        "company_secretary": _pick(["J. Pemberton","A. Whitfield","H. Forsythe"], seed+"cs"),
        "total_pages": 1,
    }
    return [base]


def _ctx_employer_letter(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    gross = Decimal(ctx["primary_amount"])
    titles = {"finance":["Managing Director","Director","Vice President"],
              "technology":["Engineering Manager","Director of Engineering","CTO"],
              "real_estate":["Director","Associate Director","Head of Sales"],
              "professional_services":["Partner","Senior Manager","Director"]}
    industry = ctx.get("industry", "finance")
    base = {
        **ctx,
        "employer_name": ctx.get("employer_name", ""),
        "employer_address": ctx.get("employer_address", ""),
        "letter_date": ctx.get("period_end", ctx.get("event_date", "")),
        "recipient_org": "Source of Wealth Verification Team",
        "employee_name": ctx["subject_name"],
        "job_title": _pick(titles.get(industry, titles["finance"]), seed+"jt"),
        "department": ctx.get("department", ""),
        "start_date": ctx.get("period_start", ""),
        "end_date": ctx.get("period_end", ""),
        "still_employed": ctx.get("still_employed", False),
        "period_label": ctx.get("period_label", ""),
        "gross_pay": _fmt(gross),
        "pay_basis": "Annual salary and discretionary bonus",
        "additional_notes": "",
        "signatory_name": _pick(["H. Pemberton","A. Whitfield","S. Thornton"], seed+"sn"),
        "signatory_title": "Head of Human Resources",
        "employer_tel": f"+44 20 {_h(seed+'tel') % 90000000 + 10000000:08d}",
        "letter_ref": f"HR/{_h(seed+'lr') % 100000:05d}",
        "total_pages": 1,
    }
    return [base]


def _ctx_solicitor_letter(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    sf = solicitor_firm(seed)
    amount = Decimal(ctx["primary_amount"])
    letter_type = ctx.get("letter_type", "inheritance")
    base = {
        **ctx,
        "solicitor_firm": sf["name"],
        "solicitor_address": sf["address"],
        "dx_ref": f"DX {_h(seed+'dx') % 100000}",
        "letter_date": ctx.get("event_date", ""),
        "our_ref": solicitor_ref(seed),
        "your_ref": "—",
        "recipient_name": "The Compliance Team",
        "salutation": "Sir/Madam",
        "matter_description": ctx.get("matter_description", "Source of Wealth Verification"),
        "letter_type": letter_type,
        "subject_name": ctx["subject_name"],
        "confirmed_amount": _fmt(amount),
        "confirmed_amount_words": _words(amount),
        # inheritance
        "deceased_name": ctx.get("deceased_name", deceased_name(seed)),
        "beneficiary_name": ctx["subject_name"],
        "will_date": ctx.get("will_date", ""),
        "distribution_date": ctx.get("event_date", ""),
        "probate_ref": f"PROB{_h(seed+'pref') % 1000000:06d}",
        "net_estate": _fmt(amount),
        "share_percentage": str(_h(seed+"share") % 80 + 20),
        "acting_for_estate": letter_type == "inheritance",
        # gift
        "donor_name": ctx.get("donor_name", donor_name(seed)),
        "recipient_name_g": ctx["subject_name"],
        "gift_date": ctx.get("event_date", ""),
        # business
        "company_name": ctx.get("company_name", gen_company_name(seed)),
        "director_name": ctx["subject_name"],
        "period_end": ctx.get("period_end", ""),
        "enclosures": ctx.get("enclosures", "Copy of relevant documentation"),
        "total_pages": 1,
    }
    return [base]


def _ctx_email_thread(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    amount = Decimal(ctx["primary_amount"])
    email_type = ctx.get("email_type", "employment")
    domains = {"finance":"meridian-capital.co.uk","technology":"nexus-systems.co.uk",
               "real_estate":"harrington-property.co.uk","professional_services":"whitmore-llp.co.uk"}
    domain = domains.get(ctx.get("industry","finance"), "company.co.uk")
    subject_first = ctx["subject_name"].split()[0]
    base = {
        **ctx,
        "sender_email": f"hr@{domain}" if email_type == "employment" else f"admin@{domain}",
        "recipient_email": f"{subject_first.lower()}@personal.email",
        "email_date": ctx.get("event_date", ""),
        "email_subject": {
            "employment": f"RE: Employment Confirmation — {ctx['subject_name']}",
            "gift": f"Re: Transfer Confirmation",
            "inheritance": f"RE: Estate of {ctx.get('deceased_name', 'the Deceased')} — Distribution",
            "business": f"Q{_h(seed+'q') % 4 + 1} Distribution — Confirmed",
        }.get(email_type, "RE: Confirmation"),
        "email_greeting": f"Dear {subject_first}",
        "email_type": email_type,
        "employer_name": ctx.get("employer_name", ""),
        "job_title": ctx.get("job_title", ""),
        "period_label": ctx.get("period_label", ""),
        "gross_pay": _fmt(amount),
        "base_salary": _fmt(amount),
        "still_employed": ctx.get("still_employed", False),
        "gift_amount": _fmt(amount),
        "gift_date": ctx.get("event_date", ""),
        "transfer_ref": f"TRF{_h(seed+'tr') % 10000000:07d}",
        "donor_name": ctx.get("donor_name", donor_name(seed)),
        "deceased_name": ctx.get("deceased_name", deceased_name(seed)),
        "beneficiary_name": ctx["subject_name"],
        "estate_amount": _fmt(amount),
        "transfer_date": ctx.get("event_date", ""),
        "solicitor_firm": solicitor_firm(seed)["name"],
        "distribution_amount": _fmt(amount),
        "company_name": ctx.get("company_name", gen_company_name(seed)),
        "recipient_first_name": subject_first,
        "sender_name": _pick(["Sophie Hargreaves","Tom Reynolds","Claire Dunmore","James Pemberton"], seed+"sn"),
        "sender_title": "HR Manager" if email_type == "employment" else "Company Secretary",
        "print_date": ctx.get("event_date", ""),
        "total_pages": 1,
    }
    return [base]


def _ctx_gift_letter(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    amount = Decimal(ctx["primary_amount"])
    dn = ctx.get("donor_name", donor_name(seed))
    dr = ctx.get("donor_relationship", donor_relationship(seed))
    base = {
        **ctx,
        "donor_name": dn,
        "donor_address": "Address provided separately",
        "donor_relationship": dr,
        "recipient_name": ctx["subject_name"],
        "recipient_address": ctx.get("subject_address", ""),
        "gift_amount": _fmt(amount),
        "gift_date": ctx.get("event_date", ""),
        "letter_date": ctx.get("event_date", ""),
        "transfer_ref": f"TRF{_h(seed+'tr') % 10000000:07d}",
        "intended_use": "personal wealth management / property purchase",
        "witness_name": _pick(["Sarah Mitchell","James Okafor","Claire Dunmore"], seed+"wn"),
        "witness_address": "Address provided separately",
        "total_pages": 1,
    }
    return [base]


def _ctx_bloomberg(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    amount = Decimal(ctx["primary_amount"])
    approx = (amount / Decimal("1000000")).quantize(Decimal("0.1"))
    subject = ctx["subject_name"]
    bloomberg_type = ctx.get("bloomberg_type", "business")

    if bloomberg_type == "inheritance":
        headline = f"Estate of {ctx.get('deceased_name', deceased_name(seed))} Valued at GBP {approx}M"
        lead = (f"{subject}, a beneficiary of the estate of the late "
                f"{ctx.get('deceased_name', deceased_name(seed))}, is set to receive assets "
                f"valued at approximately GBP {approx} million, according to probate records.")
        body1 = (f"The estate, which includes property and investment holdings, was "
                 f"estimated at GBP {approx}M in filings submitted to the Principal Registry "
                 f"of the Family Division.")
        body2 = (f"Beneficiaries are expected to receive distributions over the coming "
                 f"months as the executors complete administration of the estate.")
        body3 = f"The grant of probate was issued in {ctx.get('event_date','')[:7]}."
        financial_detail = f"  Estate gross value:   GBP {_fmt(amount)}"
        quote = f"The estate has been administered in accordance with the terms of the Will"
        quote_attr = "a representative of the estate"
    elif bloomberg_type == "employment":
        co = ctx.get("employer_name", gen_company_name(seed))
        headline = f"{co} Raises Headcount; Senior Hire Earns GBP {approx}M Package"
        lead = (f"{co} has expanded its senior team, with total compensation for "
                f"new hires averaging GBP {approx} million per annum.")
        body1 = (f"The firm confirmed that remuneration for senior roles in its "
                 f"{ctx.get('industry','finance')} division is in line with market benchmarks.")
        body2 = (f"Industry surveys indicate that senior professionals at comparable "
                 f"firms earn between GBP {_fmt(amount * Decimal('0.8'))} and "
                 f"GBP {_fmt(amount * Decimal('1.2'))} annually.")
        body3 = "The firm declined to comment on individual compensation."
        financial_detail = f"  Reported annual remuneration range: GBP {_fmt(amount)}"
        quote = "We remain committed to attracting and retaining top talent"
        quote_attr = f"spokesperson for {co}"
    else:  # business
        co = ctx.get("company_name", gen_company_name(seed))
        headline = f"{co} Sold for GBP {approx}M in Management Buyout"
        lead = (f"{co} has been acquired in a deal valuing the business at approximately "
                f"GBP {approx} million, with proceeds distributed to its shareholder-directors.")
        body1 = (f"The transaction, structured as a management buyout, saw the founders "
                 f"realise their investment after building the company over several years.")
        body2 = (f"{subject}, a director and shareholder, received proceeds of approximately "
                 f"GBP {_fmt(amount)} from the sale, reflecting their equity stake.")
        body3 = (f"The deal is expected to complete by end of {ctx.get('event_date','')[:7]}, "
                 f"subject to regulatory clearances.")
        financial_detail = (f"  Enterprise Value:   GBP {_fmt(amount * Decimal('1.2'))}\n"
                            f"  Equity Proceeds:    GBP {_fmt(amount)}")
        quote = "We are proud of what we built and excited about the next chapter"
        quote_attr = f"{subject}, outgoing director"

    base = {
        **ctx,
        "headline": headline,
        "byline": bloomberg_byline(seed),
        "dateline_city": dateline_city(seed),
        "article_date": ctx.get("event_date", ""),
        "article_time": f"{_h(seed+'at') % 12 + 8:02d}:{_h(seed+'ats') % 60:02d}",
        "article_year": ctx.get("event_date", "2022")[:4],
        "lead_paragraph": lead,
        "body_paragraph_1": body1,
        "body_paragraph_2": body2,
        "body_paragraph_3": body3,
        "financial_detail": financial_detail,
        "quote": quote,
        "quote_attribution": quote_attr,
        "closing_paragraph": (
            f"For more information on this story, contact Bloomberg News at "
            f"+1-212-617-2300."
        ),
        "related_links": [
            f"UK Mergers and Acquisitions",
            f"Private Equity Deals",
            f"Estate Planning and Wealth",
        ],
        "tags": f"{ctx.get('industry','Finance')}, M&A, UK, Wealth Management",
        "update_note": "additional details on transaction terms",
        "reported_amount": _fmt(amount),
        "total_pages": 1,
    }
    return [base]


def _ctx_ft_article(ctx: dict) -> list[dict]:
    # FT style — reuse bloomberg logic with different byline and framing
    pages = _ctx_bloomberg(ctx)
    seed  = ctx["doc_id"]
    pages[0]["byline"]   = ft_byline(seed)
    pages[0]["headline"] = "FT: " + pages[0]["headline"]
    return pages


def _ctx_companies_house(ctx: dict) -> list[dict]:
    seed = ctx["doc_id"]
    dist = Decimal(ctx["primary_amount"])
    co   = ctx.get("company_name", gen_company_name(seed))
    cnum = ctx.get("company_number", company_number(seed))
    turnover = (dist * Decimal("5")).quantize(_TWO)
    op       = (turnover * Decimal("0.20")).quantize(_TWO)
    pat      = (op * Decimal("0.81")).quantize(_TWO)
    sic_codes = [("6420", "Activities of holding companies"),
                 ("6499", "Other financial service activities"),
                 ("7022", "Business and other management consultancy")]
    sic = _pick(sic_codes, seed + "sic")
    directors = [{"name": ctx["subject_name"], "appointed": ctx.get("period_start", "")}]
    base = {
        **ctx,
        "company_name": co,
        "company_number": cnum,
        "company_address": ctx.get("company_address", company_address(seed)),
        "incorporation_date": ctx.get("incorporation_date", ctx.get("period_start", "")),
        "sic_code": sic[0],
        "sic_description": sic[1],
        "filing_date": ctx.get("period_end", ""),
        "accounts_year_end": ctx.get("period_end", ""),
        "accounts_type": "Total Exemption Full",
        "turnover": _fmt(turnover),
        "operating_profit": _fmt(op),
        "profit_after_tax": _fmt(pat),
        "distribution_amount": _fmt(dist),
        "director_name": ctx["subject_name"],
        "directors": directors,
        "ch_ref": f"CH{_h(seed+'ch') % 10000000:07d}",
        "extract_date": ctx.get("period_end", ""),
        "total_pages": 1,
    }
    return [base]


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_BUILDERS: dict[str, Callable[[dict], list[dict]]] = {
    "payslip":                   _ctx_payslip,
    "bank_statement":            _ctx_bank_statement,
    "bank_transfer_confirmation":_ctx_bank_transfer,
    "probate_grant":             _ctx_probate_grant,
    "will_extract":              _ctx_will_extract,
    "gift_deed":                 _ctx_gift_deed,
    "share_purchase_agreement":  _ctx_share_purchase,
    "company_accounts":          _ctx_company_accounts,
    "distribution_statement":    _ctx_distribution_statement,
    "employer_letter":           _ctx_employer_letter,
    "solicitor_letter":          _ctx_solicitor_letter,
    "email_thread":              _ctx_email_thread,
    "gift_letter":               _ctx_gift_letter,
    "bloomberg_article":         _ctx_bloomberg,
    "ft_article":                _ctx_ft_article,
    "companies_house_filing":    _ctx_companies_house,
}

_TEMPLATE_MAP: dict[str, list[str]] = {
    # format_type_value → [template path relative to templates root, ...]
    "payslip":                    ["structured/payslip_p1.j2","structured/payslip_p2.j2",
                                   "structured/payslip_p3.j2","structured/payslip_p4.j2"],
    "bank_statement":             ["structured/bank_statement_p1.j2"],
    "bank_transfer_confirmation": ["structured/bank_transfer_p1.j2"],
    "probate_grant":              ["structured/probate_grant_p1.j2"],
    "company_accounts":           ["structured/company_accounts_p1.j2",
                                   "structured/company_accounts_p1.j2"],
    "distribution_statement":     ["structured/distribution_statement_p1.j2"],
    "will_extract":               ["legal/will_extract_p1.j2"],
    "gift_deed":                  ["legal/gift_deed_p1.j2"],
    "share_purchase_agreement":   ["legal/share_purchase_p1.j2",
                                   "legal/share_purchase_p1.j2"],
    "employer_letter":            ["correspondence/employer_letter_p1.j2"],
    "solicitor_letter":           ["correspondence/solicitor_letter_p1.j2"],
    "email_thread":               ["correspondence/email_thread_p1.j2"],
    "gift_letter":                ["correspondence/gift_letter_p1.j2"],
    "bloomberg_article":          ["press/bloomberg_article_p1.j2"],
    "ft_article":                 ["press/bloomberg_article_p1.j2"],
    "companies_house_filing":     ["press/companies_house_p1.j2"],
}


# ---------------------------------------------------------------------------
# OcrPage assembly (shared)
# ---------------------------------------------------------------------------

def _line_to_ocr(text: str, page_width: float, line_index: int) -> OcrLine:
    y_top = 50.0 + line_index * 14.0
    y_bot = y_top + 12.0
    words, x = [], 30.0
    for token in text.split():
        w = len(token) * 6.5
        words.append(OcrWord(text=token, confidence=1.0,
                             polygon=[x, y_top, x+w, y_top, x+w, y_bot, x, y_bot]))
        x += w + 4.0
    return OcrLine(text=text, confidence=1.0,
                   polygon=[30.0, y_top, page_width-30.0, y_top,
                             page_width-30.0, y_bot, 30.0, y_bot],
                   words=words)


def _build_key_values(format_type: str, page_ctx: dict, page_num: int) -> list[KeyValue]:
    """Emit the primary-amount field plus entity fields as KeyValues."""
    spec   = FORMATS.get(format_type)
    kvs    = []
    currency = page_ctx.get("currency", "GBP")

    # Primary amount (varies by format)
    amt_field = spec.primary_amount_field if spec else "Amount"
    raw = page_ctx.get("primary_amount", "0")
    try:
        kvs.append(KeyValue(key=amt_field, value=f"{currency} {_fmt(Decimal(str(raw)))}"))
    except Exception:
        pass

    # Entity fields common to all formats
    for k, v in [
        ("Subject Name",  page_ctx.get("subject_name", "")),
        ("Document Type", format_type.replace("_", " ").title()),
        ("Page",          f"{page_num} of {page_ctx.get('total_pages', 1)}"),
    ]:
        if v:
            kvs.append(KeyValue(key=k, value=str(v)))
    return kvs


def _render_pages(format_type: str, page_contexts: list[dict]) -> list[OcrPage]:
    templates = _TEMPLATE_MAP.get(format_type, [])
    pages = []
    for i, page_ctx in enumerate(page_contexts):
        tpl_path = templates[min(i, len(templates) - 1)] if templates else None
        if tpl_path:
            try:
                tpl = _JINJA_ENV.get_template(tpl_path)
                rendered = tpl.render(**page_ctx)
            except Exception as e:
                rendered = f"[Render error for {format_type} p{i+1}: {e}]"
        else:
            rendered = f"[No template for {format_type}]"

        pw, ph = 595.0, 842.0
        lines = [_line_to_ocr(line, pw, li)
                 for li, line in enumerate(rendered.splitlines()) if line.strip()]
        kvs = _build_key_values(format_type, page_ctx, i + 1)
        pages.append(OcrPage(page_number=i+1, width=pw, height=ph,
                              lines=lines, key_values=kvs))
    return pages


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

from sow_synth.docplan import DocumentPlan  # noqa: E402 (avoid circular at module level)


def realize_document(plan: DocumentPlan, doc: Document) -> None:
    """Render plan → pages, mutate doc.pages in-place."""
    format_type = plan.template_context.get("format_type", "bank_statement")
    builder = _BUILDERS.get(format_type, _BUILDERS["bank_statement"])
    page_ctxs = builder(plan.template_context)
    doc.pages.extend(_render_pages(format_type, page_ctxs))


def realize_all(plans: list[DocumentPlan], documents: dict[str, Document]) -> None:
    for plan in plans:
        realize_document(plan, documents[plan.doc_id])
