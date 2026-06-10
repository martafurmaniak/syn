"""Stage 8 — Document body rendering (LLM or code fallback).

Each DocumentSpec is rendered to page_text via one of two paths:

  LLM path (when llm is provided):
    One structured-output call per document.  The model receives a prompt
    describing the document type, its evidential role, and ALL mandatory_facts
    verbatim.  It writes the full document text with natural layout variation.
    The response is a single `text` field.

  Code fallback (when llm is None — used in tests):
    A simple, readable fact-sheet layout assembled from mandatory_facts.
    Not template-based — just Python string formatting.

Post-rendering, verify.py checks that every mandatory_facts value appears in
the generated page_text.

Entry points:
  render_document(spec, llm=None) → str          (single document text)
  render_all_bundles(bundles, documents, llm=None) (mutates Document.pages in-place)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from sow_synth.formats import DOC_CATEGORY
from sow_synth.models import DocType, OcrPage

if TYPE_CHECKING:
    from sow_synth.docplan import CorroborationBundle, DocumentSpec
    from sow_synth.llm import LlmClient
    from sow_synth.models import Document

# ---------------------------------------------------------------------------
# LLM response schema
# ---------------------------------------------------------------------------

class _DocumentBody(BaseModel):
    text: str   # full document text — layout is the model's choice


# ---------------------------------------------------------------------------
# LLM prompt building
# ---------------------------------------------------------------------------

_DOC_TYPE_DESCRIPTIONS: dict[DocType, str] = {
    DocType.payslip:                   "P60 / end-of-year payslip issued by an employer to an employee",
    DocType.employer_letter:           "letter from an employer confirming an employee's salary and tenure",
    DocType.employment_contract:       "employment contract between employer and employee",
    DocType.bank_statement:            "bank account statement showing transactions and running balance",
    DocType.bank_transfer_confirmation:"bank confirmation of a specific electronic funds transfer",
    DocType.probate_grant:             "grant of probate issued by a probate registry confirming an estate",
    DocType.will_extract:              "extract from a will showing a specific bequest to a beneficiary",
    DocType.solicitor_letter:          "letter from a solicitor confirming a transaction or legal matter",
    DocType.gift_letter:               "letter from a donor confirming a financial gift",
    DocType.gift_deed:                 "executed deed of gift transferring funds from donor to donee",
    DocType.companies_house_filing:    "Companies House certificate of incorporation or filing",
    DocType.company_accounts:          "statutory company accounts filed at Companies House",
    DocType.distribution_statement:    "dividend or profit-distribution statement from a company",
    DocType.bloomberg_article:         "Bloomberg financial news article mentioning a transaction or individual",
    DocType.ft_article:                "Financial Times article mentioning a transaction or individual",
    DocType.email_thread:              "email exchange confirming a financial transaction or arrangement",
    DocType.client_history:            "internal client source-of-wealth history document",
}

_ROLE_DESCRIPTIONS: dict[str, str] = {
    "proves_entity_ownership":   "establishes who owns or controls a legal entity",
    "proves_transaction":        "evidences that a specific financial transfer occurred",
    "proves_employment_tenure":  "confirms employment period, role, and/or remuneration",
    "proves_income_amount":      "establishes total income or gross earnings",
    "proves_inheritance_right":  "confirms entitlement to and receipt of an inheritance",
    "proves_gift_transfer":      "evidences a voluntary financial gift between two parties",
    "proves_business_ownership": "establishes ownership stake in a business",
    "corroborates_narrative":    "provides external narrative corroboration for the claim",
}

_SYSTEM = (
    "You are generating a realistic document for a UK private-bank source-of-wealth "
    "case file. Produce a complete, plausible document that looks genuinely produced "
    "by the issuing organisation — vary the layout, heading style, and phrasing "
    "naturally. You MUST embed every value from the MANDATORY FACTS block verbatim "
    "in your document. Do not invent any figures, names, dates, or references not "
    "listed in the mandatory facts."
)


def _build_prompt(spec: "DocumentSpec") -> str:
    doc_label = _DOC_TYPE_DESCRIPTIONS.get(
        spec.doc_type,
        spec.doc_type.value.replace("_", " ").title(),
    )
    role_label = _ROLE_DESCRIPTIONS.get(spec.evidential_role.value, spec.evidential_role.value)
    category   = DOC_CATEGORY.get(spec.doc_type, "document")
    facts_block = "\n".join(f"  {k}: {v}" for k, v in spec.mandatory_facts.items())

    return (
        f"Document type:     {doc_label}\n"
        f"Category:          {category}\n"
        f"Evidential role:   {role_label}\n"
        f"Style:             {spec.style_hint}\n\n"
        f"MANDATORY FACTS — embed all of these verbatim:\n"
        f"{facts_block}\n\n"
        f"Generate the complete document text.  Include a realistic header, "
        f"body sections, and a signature or footer block appropriate for this "
        f"document type.  Vary the structure — do not use a formulaic layout."
    )


# ---------------------------------------------------------------------------
# Code fallback renderer (no LLM — used in tests and when llm=None)
# ---------------------------------------------------------------------------

_RULE = "=" * 68
_SEP  = "-" * 68


def _code_render(spec: "DocumentSpec") -> str:
    doc_label = _DOC_TYPE_DESCRIPTIONS.get(
        spec.doc_type,
        spec.doc_type.value.replace("_", " ").title(),
    )
    role_label = _ROLE_DESCRIPTIONS.get(spec.evidential_role.value, spec.evidential_role.value)

    lines = [
        doc_label.upper(),
        _RULE,
        f"Evidential role: {role_label}",
        "",
    ]

    # Group facts loosely: IDs and refs at top, amounts in middle, dates at bottom
    id_keys   = {"Document ID", "Reference", "Contract Ref", "Deed Reference",
                 "Probate Reference", "Our Ref", "PAYE Reference"}
    date_keys = {"Date", "Grant Date", "Date of Death", "Transfer Date", "Gift Date",
                 "Execution Date", "Publication Date", "Employment Period", "Period",
                 "Start Date", "Accounting Period", "Payment Date"}

    ids    = {k: v for k, v in spec.mandatory_facts.items() if k in id_keys}
    dates  = {k: v for k, v in spec.mandatory_facts.items() if k in date_keys}
    other  = {k: v for k, v in spec.mandatory_facts.items()
               if k not in id_keys and k not in date_keys}

    def _section(title: str, items: dict) -> list[str]:
        if not items:
            return []
        return [title, _SEP, *[f"  {k:<30} {v}" for k, v in items.items()], ""]

    lines += _section("DETAILS", other)
    lines += _section("DATES & PERIODS", dates)
    lines += _section("REFERENCES", ids)
    lines += [_RULE]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def render_document(spec: "DocumentSpec", llm: "LlmClient | None" = None) -> str:
    """Render one DocumentSpec to a text string.

    Uses LLM when provided; falls back to code rendering otherwise.
    """
    if llm is not None:
        try:
            body: _DocumentBody = llm.complete(
                step=f"render_{spec.doc_id}",
                system=_SYSTEM,
                user=_build_prompt(spec),
                response_model=_DocumentBody,
            )
            return body.text
        except Exception:
            pass  # fall through to code render on any LLM failure
    return _code_render(spec)


def render_all_bundles(
    bundles: "list[CorroborationBundle]",
    documents: "dict[str, Document]",
    llm: "LlmClient | None" = None,
) -> None:
    """Render every DocumentSpec in every bundle, mutating Document.pages in-place."""
    for bundle in bundles:
        for doc_spec in bundle.doc_specs:
            doc = documents.get(doc_spec.doc_id)
            if doc is None:
                continue
            text = render_document(doc_spec, llm=llm)
            doc.pages.append(OcrPage(page_number=1, page_text=f"<pre>{text}</pre>"))
