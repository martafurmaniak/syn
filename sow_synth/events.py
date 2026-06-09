"""Stage 2 — Event timeline generation (code only, no LLM).

Every financial fact is a dated, typed Event.  Amounts come from realistic
log-normal distributions; causal ordering is enforced; tax is derived from
inflows rather than sampled independently.

The public entry point is `generate_events(profile, spec, rng)`.  After
calling this, run `ledger.balance_to_target` to add a single balancing event
that nudges net worth into the target band.
"""
from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from itertools import count

import numpy as np

from sow_synth.models import Event, EventType, Profile, SowType
from sow_synth.spec import ScenarioSpec

_counter = count(1)


def _event_id(prefix: str, n: int) -> str:
    return f"{prefix}_{n:06d}"


def _d(x: float) -> Decimal:
    """Convert float to Decimal via string to avoid float precision artefacts."""
    return Decimal(f"{x:.2f}")


def _rand_date_between(start: date, end: date, rng: np.random.Generator) -> date:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=int(rng.integers(0, delta)))


# ---------------------------------------------------------------------------
# Employment events
# ---------------------------------------------------------------------------

def _generate_employment_events(
    profile: Profile,
    spec: ScenarioSpec,
    n_claims: int,
    rng: np.random.Generator,
    id_seq: count,
) -> list[Event]:
    """One role per claim.  Each role produces annual payslip-style events."""
    if n_claims == 0:
        return []

    events: list[Event] = []
    career_end = spec.as_of
    # Divide career into n_claims consecutive roles
    total_days = max((career_end - profile.plausible_career_start).days, 1)
    role_length_days = total_days // n_claims

    # Base salary scales with target net worth (order-of-magnitude calibration)
    nw_mid = float(spec.target_net_worth_midpoint)
    base_salary = nw_mid / max(total_days / 365, 1) * 0.6  # employment ~ 60% of wealth
    base_salary = max(base_salary, 30_000.0)

    role_start = profile.plausible_career_start
    for role_idx in range(n_claims):
        role_id = f"role_{spec.seed}_{role_idx}"
        role_end = role_start + timedelta(days=role_length_days)
        if role_idx == n_claims - 1:
            role_end = career_end

        # Salary for this role: grows with seniority
        seniority_factor = 1.0 + 0.15 * role_idx
        salary = float(rng.lognormal(np.log(base_salary * seniority_factor), 0.3))
        salary = max(salary, 20_000.0)

        # One event per calendar year in the role
        year = role_start.year
        while True:
            pay_date = date(year, 12, 31)
            if pay_date < role_start:
                year += 1
                continue
            if pay_date > role_end:
                break
            # Pro-rate the final year
            year_start = date(year, 1, 1)
            year_end = date(year, 12, 31)
            effective_start = max(role_start, year_start)
            effective_end = min(role_end, year_end)
            fraction = (effective_end - effective_start).days / 365
            annual_amount = salary * fraction
            if annual_amount < 1.0:
                year += 1
                continue

            n = next(id_seq)
            events.append(Event(
                event_id=_event_id("emp", n),
                type=EventType.employment_income,
                direction="inflow",
                date=pay_date,
                amount=_d(annual_amount),
                currency=spec.currency,
                sow_type=SowType.employment,
                meta={"role_id": role_id, "role_index": role_idx, "annual_salary": round(salary, 2)},
            ))
            year += 1

        role_start = role_end + timedelta(days=1)

    return events


# ---------------------------------------------------------------------------
# Inheritance events
# ---------------------------------------------------------------------------

def _generate_inheritance_events(
    profile: Profile,
    spec: ScenarioSpec,
    n_claims: int,
    rng: np.random.Generator,
    id_seq: count,
) -> list[Event]:
    if n_claims == 0:
        return []

    nw_mid = float(spec.target_net_worth_midpoint)
    events: list[Event] = []

    for i in range(n_claims):
        # Parent plausibly dies when client is 40–65
        client_age_at_death = 40 + int(rng.integers(0, 26))
        death_year = profile.date_of_birth.year + client_age_at_death
        event_date = date(death_year, int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        # Clamp to [career_start, as_of]
        event_date = max(event_date, profile.plausible_career_start)
        event_date = min(event_date, spec.as_of)

        amount = float(rng.lognormal(np.log(max(nw_mid * 0.25, 50_000)), 0.6))
        amount = max(amount, 10_000.0)

        n = next(id_seq)
        events.append(Event(
            event_id=_event_id("inh", n),
            type=EventType.inheritance,
            direction="inflow",
            date=event_date,
            amount=_d(amount),
            currency=spec.currency,
            sow_type=SowType.inheritance,
            meta={"inheritance_index": i},
        ))

    return events


# ---------------------------------------------------------------------------
# Gift events
# ---------------------------------------------------------------------------

def _generate_gift_events(
    profile: Profile,
    spec: ScenarioSpec,
    n_claims: int,
    rng: np.random.Generator,
    id_seq: count,
) -> list[Event]:
    if n_claims == 0:
        return []

    nw_mid = float(spec.target_net_worth_midpoint)
    events: list[Event] = []

    for i in range(n_claims):
        event_date = _rand_date_between(profile.plausible_career_start, spec.as_of, rng)
        amount = float(rng.lognormal(np.log(max(nw_mid * 0.08, 10_000)), 0.7))
        amount = max(amount, 1_000.0)

        n = next(id_seq)
        events.append(Event(
            event_id=_event_id("gft", n),
            type=EventType.gift_received,
            direction="inflow",
            date=event_date,
            amount=_d(amount),
            currency=spec.currency,
            sow_type=SowType.gift,
            meta={"gift_index": i},
        ))

    return events


# ---------------------------------------------------------------------------
# Business profit distribution events
# ---------------------------------------------------------------------------

def _generate_business_events(
    profile: Profile,
    spec: ScenarioSpec,
    n_claims: int,
    rng: np.random.Generator,
    id_seq: count,
) -> list[Event]:
    """Each claim = one business; each business has 2–5 annual distributions."""
    if n_claims == 0:
        return []

    nw_mid = float(spec.target_net_worth_midpoint)
    events: list[Event] = []

    career_span_years = max((spec.as_of - profile.plausible_career_start).days // 365, 1)

    for b in range(n_claims):
        business_id = f"biz_{spec.seed}_{b}"
        # Business founded at a random point in career
        found_offset_years = int(rng.integers(2, max(career_span_years - 1, 3)))
        founding_date = profile.plausible_career_start + timedelta(days=found_offset_years * 365)
        founding_date = min(founding_date, spec.as_of - timedelta(days=365))

        n_distributions = int(rng.integers(2, 6))
        annual_profit = float(rng.lognormal(np.log(max(nw_mid * 0.2, 40_000)), 0.5))

        for d in range(n_distributions):
            dist_date = founding_date + timedelta(days=(d + 1) * 365)
            if dist_date > spec.as_of:
                break
            amount = annual_profit * float(rng.lognormal(0, 0.2))  # year-by-year variance
            amount = max(amount, 1_000.0)

            n = next(id_seq)
            events.append(Event(
                event_id=_event_id("biz", n),
                type=EventType.business_profit_distribution,
                direction="inflow",
                date=dist_date,
                amount=_d(amount),
                currency=spec.currency,
                sow_type=SowType.business_profits,
                meta={"business_id": business_id, "distribution_index": d},
            ))

    return events


# ---------------------------------------------------------------------------
# Outflow events (tax derived from inflows; living expenses annual)
# ---------------------------------------------------------------------------

_TAX_RATE_BY_TYPE: dict[EventType, float] = {
    EventType.employment_income: 0.30,
    EventType.inheritance: 0.00,        # UK: inheritance tax paid by estate, not recipient
    EventType.gift_received: 0.00,
    EventType.business_profit_distribution: 0.20,
    EventType.investment_gain: 0.20,
}


def _generate_outflow_events(
    profile: Profile,
    spec: ScenarioSpec,
    inflow_events: list[Event],
    rng: np.random.Generator,
    id_seq: count,
) -> list[Event]:
    outflows: list[Event] = []

    # Tax: grouped by calendar year, charged on Jan 31 of the following year
    tax_by_year: dict[int, float] = {}
    for e in inflow_events:
        rate = _TAX_RATE_BY_TYPE.get(e.type, 0.0)
        if rate > 0:
            tax_by_year[e.date.year] = tax_by_year.get(e.date.year, 0.0) + float(e.amount) * rate

    for year, tax_amount in sorted(tax_by_year.items()):
        tax_date = date(year + 1, 1, 31)
        if tax_date > spec.as_of:
            tax_date = spec.as_of
        if tax_amount < 1.0:
            continue
        n = next(id_seq)
        outflows.append(Event(
            event_id=_event_id("tax", n),
            type=EventType.tax,
            direction="outflow",
            date=tax_date,
            amount=_d(tax_amount),
            currency=spec.currency,
            meta={"tax_year": year},
        ))

    # Living expenses: one event per year, from career start to as_of
    # Calibrate to ~15% of average annual inflow
    total_inflows = sum(float(e.amount) for e in inflow_events)
    career_years = max((spec.as_of - profile.plausible_career_start).days / 365, 1)
    avg_annual_income = total_inflows / career_years
    annual_living = avg_annual_income * 0.15
    annual_living = max(annual_living, 15_000.0)

    year = profile.plausible_career_start.year
    while date(year, 12, 31) <= spec.as_of:
        amount = annual_living * float(rng.lognormal(0, 0.15))
        amount = max(amount, 10_000.0)
        n = next(id_seq)
        outflows.append(Event(
            event_id=_event_id("liv", n),
            type=EventType.living_expense,
            direction="outflow",
            date=date(year, 12, 31),
            amount=_d(amount),
            currency=spec.currency,
            meta={"expense_year": year},
        ))
        year += 1

    return outflows


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_events(
    profile: Profile,
    spec: ScenarioSpec,
    rng: np.random.Generator,
) -> list[Event]:
    """Generate the full event timeline for a given profile and spec.

    Returns events sorted chronologically.  Call `ledger.balance_to_target`
    afterward to inject the balancing investment event.
    """
    id_seq: count = count(1)

    inflows: list[Event] = []
    inflows += _generate_employment_events(
        profile, spec, spec.claims_per_sow_type.get(SowType.employment, 0), rng, id_seq
    )
    inflows += _generate_inheritance_events(
        profile, spec, spec.claims_per_sow_type.get(SowType.inheritance, 0), rng, id_seq
    )
    inflows += _generate_gift_events(
        profile, spec, spec.claims_per_sow_type.get(SowType.gift, 0), rng, id_seq
    )
    inflows += _generate_business_events(
        profile, spec, spec.claims_per_sow_type.get(SowType.business_profits, 0), rng, id_seq
    )

    outflows = _generate_outflow_events(profile, spec, inflows, rng, id_seq)

    all_events = sorted(inflows + outflows, key=lambda e: e.date)
    return all_events
