"""Weighted format selection per SoW type.

`select_format(sow_type, rng)` returns a FormatSpec drawn from the weighted
distribution for that claim type.  Because it consumes from `rng`, the result
is fully deterministic given the spec seed — varying only across seeds.
"""
from __future__ import annotations

import numpy as np

from sow_synth.formats import FORMATS, FormatSpec
from sow_synth.models import SowType

# ---------------------------------------------------------------------------
# Weighted tables  (format_key, weight)
# ---------------------------------------------------------------------------

_WEIGHTS: dict[SowType, list[tuple[str, float]]] = {
    SowType.employment: [
        ("payslip",          0.30),
        ("bank_statement",   0.25),
        ("employer_letter",  0.22),
        ("email_thread",     0.13),
        ("bloomberg_article",0.10),  # profile feature / remuneration survey
    ],
    SowType.inheritance: [
        ("will_extract",              0.25),
        ("probate_grant",             0.22),
        ("bank_transfer_confirmation",0.20),
        ("solicitor_letter",          0.18),
        ("bloomberg_article",         0.15),  # "estate of X valued at..."
    ],
    SowType.business_profits: [
        ("company_accounts",          0.22),
        ("distribution_statement",    0.20),
        ("bank_statement",            0.18),
        ("bloomberg_article",         0.20),  # deal / exit announcement
        ("companies_house_filing",    0.12),
        ("email_thread",              0.08),
    ],
    SowType.gift: [
        ("gift_letter",               0.30),
        ("gift_deed",                 0.22),
        ("bank_transfer_confirmation",0.28),
        ("email_thread",              0.12),
        ("solicitor_letter",          0.08),
    ],
}


def select_format(sow_type: SowType, rng: np.random.Generator) -> FormatSpec:
    """Draw a FormatSpec from the weighted distribution for this SoW type."""
    options = _WEIGHTS.get(sow_type, [("bank_statement", 1.0)])
    keys    = [k for k, _ in options]
    weights = np.array([w for _, w in options], dtype=float)
    weights /= weights.sum()
    chosen  = keys[int(rng.choice(len(keys), p=weights))]
    return FORMATS[chosen]
