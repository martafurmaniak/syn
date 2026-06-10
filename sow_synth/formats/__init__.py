"""Document type metadata — category mapping used for LLM prompting."""
from __future__ import annotations

from sow_synth.models import DocType

# Maps each DocType to its broad category (used in LLM prompts for context)
DOC_CATEGORY: dict[DocType, str] = {
    DocType.payslip:                     "structured financial",
    DocType.bank_statement:              "structured financial",
    DocType.bank_transfer_confirmation:  "structured financial",
    DocType.company_accounts:            "structured financial",
    DocType.distribution_statement:      "structured financial",
    DocType.probate_grant:               "structured legal",
    DocType.employment_contract:         "legal",
    DocType.will_extract:                "legal",
    DocType.gift_deed:                   "legal",
    DocType.share_purchase_agreement:    "legal",
    DocType.employer_letter:             "correspondence",
    DocType.solicitor_letter:            "correspondence",
    DocType.email_thread:                "correspondence",
    DocType.gift_letter:                 "correspondence",
    DocType.bloomberg_article:           "press",
    DocType.ft_article:                  "press",
    DocType.companies_house_filing:      "structured regulatory",
    DocType.client_history:              "internal",
}
