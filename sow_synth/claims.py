"""Stage 4 — Claim projection.

Claims are typed aggregations over subsets of events.  Amounts are exact sums
of the covered events — never independently sampled.  The invariant
`Claim.amount == sum(covered event amounts)` is enforced here by construction.

Grouping logic:
- employment  : one claim per role (grouped by meta["role_id"])
- inheritance : one claim per inheritance event
- gift        : one claim per gift_received event
- business_profits: one claim per business (grouped by meta["business_id"])
"""
from __future__ import annotations

from decimal import Decimal
from itertools import count as _count

from sow_synth.models import Claim, Event, EventType, SowType


def _claim_id(sow_type: SowType, n: int) -> str:
    return f"claim_{sow_type.value}_{n:06d}"


def project_claims(events: list[Event]) -> list[Claim]:
    """Build all claims from the event list.  Returns a stable-ordered list."""
    counter = _count(1)
    claims: list[Claim] = []

    # --- employment: group by role_id ---
    role_buckets: dict[str, list[Event]] = {}
    for e in events:
        if e.type == EventType.employment_income:
            rid = e.meta.get("role_id", "unknown_role")
            role_buckets.setdefault(rid, []).append(e)
    for role_id in sorted(role_buckets):
        evts = role_buckets[role_id]
        total = sum(e.amount for e in evts)
        claims.append(Claim(
            claim_id=_claim_id(SowType.employment, next(counter)),
            sow_type=SowType.employment,
            amount=total,
            covered_event_ids=[e.event_id for e in sorted(evts, key=lambda x: x.date)],
        ))

    # --- inheritance: one claim per event ---
    for e in sorted(
        (e for e in events if e.type == EventType.inheritance),
        key=lambda x: x.date,
    ):
        claims.append(Claim(
            claim_id=_claim_id(SowType.inheritance, next(counter)),
            sow_type=SowType.inheritance,
            amount=e.amount,
            covered_event_ids=[e.event_id],
        ))

    # --- gift: one claim per event ---
    for e in sorted(
        (e for e in events if e.type == EventType.gift_received),
        key=lambda x: x.date,
    ):
        claims.append(Claim(
            claim_id=_claim_id(SowType.gift, next(counter)),
            sow_type=SowType.gift,
            amount=e.amount,
            covered_event_ids=[e.event_id],
        ))

    # --- business_profits: group by business_id ---
    biz_buckets: dict[str, list[Event]] = {}
    for e in events:
        if e.type == EventType.business_profit_distribution:
            bid = e.meta.get("business_id", "unknown_biz")
            biz_buckets.setdefault(bid, []).append(e)
    for biz_id in sorted(biz_buckets):
        evts = biz_buckets[biz_id]
        total = sum(e.amount for e in evts)
        claims.append(Claim(
            claim_id=_claim_id(SowType.business_profits, next(counter)),
            sow_type=SowType.business_profits,
            amount=total,
            covered_event_ids=[e.event_id for e in sorted(evts, key=lambda x: x.date)],
        ))

    return claims


def verify_claim_amounts(claims: list[Claim], events: list[Event]) -> None:
    """Assert the invariant: Claim.amount == sum of its covered events.

    Raises AssertionError on the first violation (used in tests and verify.py).
    """
    event_by_id = {e.event_id: e for e in events}
    for claim in claims:
        expected = sum(event_by_id[eid].amount for eid in claim.covered_event_ids)
        assert claim.amount == expected, (
            f"Claim {claim.claim_id}: amount={claim.amount} != "
            f"sum of covered events {expected}"
        )
