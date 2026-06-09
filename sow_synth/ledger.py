"""Stage 3 — Ledger evaluation.

Deterministically folds a chronologically-sorted event list into a balance
trajectory.  Net worth is computed from the ledger; the accounting identity
`net_worth = Σ inflows − Σ outflows` holds by construction.

Also provides `balance_to_target`, which injects a single investment_gain or
investment_loss event to bring net worth inside the target band.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np

from sow_synth.models import Event, EventType, INFLOW_TYPES, OUTFLOW_TYPES
from sow_synth.spec import ScenarioSpec


def compute_net_worth(events: list[Event], as_of: date) -> Decimal:
    """Σ inflows[date ≤ as_of] − Σ outflows[date ≤ as_of]."""
    net = Decimal("0")
    for e in events:
        if e.date <= as_of:
            if e.direction == "inflow":
                net += e.amount
            else:
                net -= e.amount
    return net


def balance_trajectory(events: list[Event]) -> list[tuple[date, Decimal]]:
    """Running balance after each event, sorted by date."""
    balance = Decimal("0")
    trajectory: list[tuple[date, Decimal]] = []
    for e in sorted(events, key=lambda x: x.date):
        if e.direction == "inflow":
            balance += e.amount
        else:
            balance -= e.amount
        trajectory.append((e.date, balance))
    return trajectory


def balance_to_target(
    events: list[Event],
    spec: ScenarioSpec,
    rng: np.random.Generator,
) -> list[Event]:
    """Append a balancing investment_gain or investment_loss event so that
    closing net worth falls inside spec.target_net_worth.

    Returns a new list (original is unchanged).
    """
    lo, hi = spec.target_net_worth
    current_nw = compute_net_worth(events, spec.as_of)
    target = spec.target_net_worth_midpoint

    gap = target - current_nw
    if lo <= current_nw <= hi:
        return list(events)  # already in band — no balancing event needed

    # Place the balancing event one day before as_of
    bal_date = spec.as_of - __import__("datetime").timedelta(days=1)

    # Unique ID that won't collide with events.py's counter scheme
    bal_id = f"bal_{spec.seed:08d}_000001"

    if gap > 0:
        bal_event = Event(
            event_id=bal_id,
            type=EventType.investment_gain,
            direction="inflow",
            date=bal_date,
            amount=gap,
            currency=spec.currency,
            sow_type=None,
            meta={"balancing": True},
        )
    else:
        bal_event = Event(
            event_id=bal_id,
            type=EventType.investment_loss,
            direction="outflow",
            date=bal_date,
            amount=-gap,
            currency=spec.currency,
            meta={"balancing": True},
        )

    result = sorted(list(events) + [bal_event], key=lambda e: e.date)
    return result
