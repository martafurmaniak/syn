"""Format metadata: FormatType, PrecisionMode, FormatSpec.

A FormatSpec describes how a document *looks* (independent of what it
corroborates).  The combination of SowType (what) + FormatType (how) fully
determines the template family and verification rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class PrecisionMode(str, Enum):
    exact        = "exact"        # must match to the penny
    rounded      = "rounded"      # within £1
    approximate  = "approximate"  # within 10%
    narrative    = "narrative"    # figure may be absent; match by entity/date


FormatCategory = Literal["structured", "legal", "correspondence", "press"]


@dataclass(frozen=True)
class FormatSpec:
    doc_type_value: str             # matches DocType enum value
    category: FormatCategory
    precision_mode: PrecisionMode
    min_pages: int
    max_pages: int
    # Key-value field name in the rendered OcrPage that holds the primary amount
    primary_amount_field: str
    template_prefix: str            # templates are {prefix}_p{n}.j2


# ---------------------------------------------------------------------------
# Registry of all supported formats
# ---------------------------------------------------------------------------

FORMATS: dict[str, FormatSpec] = {
    # --- structured ---
    "payslip": FormatSpec(
        doc_type_value="payslip",
        category="structured",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=4,
        primary_amount_field="Gross Pay",
        template_prefix="payslip",
    ),
    "bank_statement": FormatSpec(
        doc_type_value="bank_statement",
        category="structured",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=3,
        primary_amount_field="Total Credits",
        template_prefix="bank_statement",
    ),
    "bank_transfer_confirmation": FormatSpec(
        doc_type_value="bank_transfer_confirmation",
        category="structured",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=1,
        primary_amount_field="Transfer Amount",
        template_prefix="bank_transfer",
    ),
    "company_accounts": FormatSpec(
        doc_type_value="company_accounts",
        category="structured",
        precision_mode=PrecisionMode.exact,
        min_pages=2, max_pages=4,
        primary_amount_field="Distribution Amount",
        template_prefix="company_accounts",
    ),
    "distribution_statement": FormatSpec(
        doc_type_value="distribution_statement",
        category="structured",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=2,
        primary_amount_field="Net Distribution",
        template_prefix="distribution_statement",
    ),
    "probate_grant": FormatSpec(
        doc_type_value="probate_grant",
        category="structured",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=2,
        primary_amount_field="Gross Estate Value",
        template_prefix="probate_grant",
    ),
    # --- legal ---
    "will_extract": FormatSpec(
        doc_type_value="will_extract",
        category="legal",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=2,
        primary_amount_field="Bequest Amount",
        template_prefix="will_extract",
    ),
    "gift_deed": FormatSpec(
        doc_type_value="gift_deed",
        category="legal",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=1,
        primary_amount_field="Gift Amount",
        template_prefix="gift_deed",
    ),
    "share_purchase_agreement": FormatSpec(
        doc_type_value="share_purchase_agreement",
        category="legal",
        precision_mode=PrecisionMode.exact,
        min_pages=2, max_pages=4,
        primary_amount_field="Purchase Price",
        template_prefix="share_purchase",
    ),
    # --- correspondence ---
    "employer_letter": FormatSpec(
        doc_type_value="employer_letter",
        category="correspondence",
        precision_mode=PrecisionMode.rounded,
        min_pages=1, max_pages=1,
        primary_amount_field="Confirmed Gross",
        template_prefix="employer_letter",
    ),
    "solicitor_letter": FormatSpec(
        doc_type_value="solicitor_letter",
        category="correspondence",
        precision_mode=PrecisionMode.rounded,
        min_pages=1, max_pages=2,
        primary_amount_field="Confirmed Amount",
        template_prefix="solicitor_letter",
    ),
    "email_thread": FormatSpec(
        doc_type_value="email_thread",
        category="correspondence",
        precision_mode=PrecisionMode.approximate,
        min_pages=1, max_pages=2,
        primary_amount_field="Stated Amount",
        template_prefix="email_thread",
    ),
    "gift_letter": FormatSpec(
        doc_type_value="gift_letter",
        category="correspondence",
        precision_mode=PrecisionMode.exact,
        min_pages=1, max_pages=1,
        primary_amount_field="Gift Amount",
        template_prefix="gift_letter",
    ),
    # --- press ---
    "bloomberg_article": FormatSpec(
        doc_type_value="bloomberg_article",
        category="press",
        precision_mode=PrecisionMode.approximate,
        min_pages=1, max_pages=2,
        primary_amount_field="Reported Amount",
        template_prefix="bloomberg_article",
    ),
    "ft_article": FormatSpec(
        doc_type_value="ft_article",
        category="press",
        precision_mode=PrecisionMode.approximate,
        min_pages=1, max_pages=2,
        primary_amount_field="Reported Amount",
        template_prefix="ft_article",
    ),
    "companies_house_filing": FormatSpec(
        doc_type_value="companies_house_filing",
        category="press",
        precision_mode=PrecisionMode.rounded,
        min_pages=1, max_pages=2,
        primary_amount_field="Profit After Tax",
        template_prefix="companies_house",
    ),
}
