"""Shared formatting utilities for document planning and rendering."""
from __future__ import annotations

import hashlib
from decimal import Decimal, ROUND_HALF_UP

_TWO = Decimal("0.01")

MONTHS = [
    "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "January", "February", "March",
]


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def pick(items: list, seed: str) -> str:
    return items[h(seed) % len(items)]


def fmt(amount: Decimal) -> str:
    """Comma-formatted two-decimal string, e.g. 1234567.89 → '1,234,567.89'."""
    return f"{amount:,.2f}"


def fmt_currency(currency: str, amount: Decimal) -> str:
    return f"{currency} {fmt(amount)}"


def words(amount: Decimal) -> str:
    """Very simplified number-to-words (whole pounds only)."""
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


def monthly_split(annual: Decimal, n: int = 12) -> list[Decimal]:
    """Split an annual amount into n approximately-equal monthly amounts."""
    base = (annual / n).quantize(_TWO, rounding=ROUND_HALF_UP)
    months = [base] * n
    months[-1] += annual - base * n
    return months
