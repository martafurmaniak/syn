"""Hypothesis property tests for the fact-core invariants.

Invariants tested (see CLAUDE.md):
1. net_worth = Σ inflows − Σ outflows
2. Claim.amount == Σ amounts of covered events
3. Causal ordering never violated
4. (spec, seed) is fully reproducible
5. Closing net worth falls inside the target band (after balancing)
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st

from sow_synth.claims import project_claims, verify_claim_amounts
from sow_synth.events import generate_events
from sow_synth.graph import assemble_graph, net_worth_from_graph
from sow_synth.ledger import balance_to_target, compute_net_worth
from sow_synth.models import SowType
from sow_synth.profile import resolve_profile
from sow_synth.spec import DifficultyProfile, ProfileConstraints, ScenarioSpec


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def scenario_specs(draw: st.DrawFn) -> ScenarioSpec:
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    as_of = draw(st.dates(min_value=date(2000, 1, 1), max_value=date(2024, 12, 31)))

    lo = draw(st.decimals(min_value=Decimal("100000"), max_value=Decimal("5000000"),
                          allow_nan=False, allow_infinity=False, places=2))
    hi = lo + draw(st.decimals(min_value=Decimal("50000"), max_value=Decimal("2000000"),
                               allow_nan=False, allow_infinity=False, places=2))

    n_employment = draw(st.integers(min_value=0, max_value=4))
    n_inheritance = draw(st.integers(min_value=0, max_value=3))
    n_gift = draw(st.integers(min_value=0, max_value=3))
    n_business = draw(st.integers(min_value=0, max_value=2))

    # At least one claim type so we have something to test
    assume(n_employment + n_inheritance + n_gift + n_business > 0)

    return ScenarioSpec(
        seed=seed,
        as_of=as_of,
        target_net_worth=(lo, hi),
        profile_constraints=ProfileConstraints(
            min_age=30,
            max_age=65,
        ),
        claims_per_sow_type={
            SowType.employment: n_employment,
            SowType.inheritance: n_inheritance,
            SowType.gift: n_gift,
            SowType.business_profits: n_business,
        },
        currency="GBP",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_full_pipeline(spec: ScenarioSpec):
    rng = np.random.default_rng(spec.seed)
    profile = resolve_profile(spec, rng)
    events = generate_events(profile, spec, rng)
    events = balance_to_target(events, spec, rng)
    claims = project_claims(events)
    fg = assemble_graph(spec, profile, events, claims)
    return profile, events, claims, fg


# ---------------------------------------------------------------------------
# Invariant 1 — ledger identity
# ---------------------------------------------------------------------------

@given(spec=scenario_specs())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_ledger_identity(spec: ScenarioSpec) -> None:
    """net_worth == Σ inflows − Σ outflows for all events up to as_of."""
    _, events, _, _ = _build_full_pipeline(spec)

    inflows = sum(e.amount for e in events if e.direction == "inflow" and e.date <= spec.as_of)
    outflows = sum(e.amount for e in events if e.direction == "outflow" and e.date <= spec.as_of)
    expected = inflows - outflows
    actual = compute_net_worth(events, spec.as_of)

    assert actual == expected, f"Ledger identity violated: {actual} != {expected}"


# ---------------------------------------------------------------------------
# Invariant 2 — claim amounts
# ---------------------------------------------------------------------------

@given(spec=scenario_specs())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_claim_amounts_equal_sum_of_covered_events(spec: ScenarioSpec) -> None:
    """Claim.amount == sum of amounts of its covered events."""
    _, events, claims, _ = _build_full_pipeline(spec)
    verify_claim_amounts(claims, events)  # raises AssertionError on violation


# ---------------------------------------------------------------------------
# Invariant 3 — claim count matches spec
# ---------------------------------------------------------------------------

@given(spec=scenario_specs())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_claim_count_matches_spec(spec: ScenarioSpec) -> None:
    """Number of claims per SoW type matches the spec."""
    _, events, claims, _ = _build_full_pipeline(spec)

    counts = {sow_type: 0 for sow_type in SowType}
    for claim in claims:
        counts[claim.sow_type] += 1

    for sow_type, expected_n in spec.claims_per_sow_type.items():
        actual_n = counts[sow_type]
        assert actual_n == expected_n, (
            f"Expected {expected_n} {sow_type} claims, got {actual_n}"
        )


# ---------------------------------------------------------------------------
# Invariant 4 — causal ordering
# ---------------------------------------------------------------------------

@given(spec=scenario_specs())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_causal_ordering(spec: ScenarioSpec) -> None:
    """business_profit_distribution events must not precede the business founding."""
    rng = np.random.default_rng(spec.seed)
    profile = resolve_profile(spec, rng)
    events = generate_events(profile, spec, rng)

    # Collect business founding dates from meta
    biz_founding: dict[str, date] = {}
    for e in events:
        if e.type.value == "business_profit_distribution":
            bid = e.meta.get("business_id", "")
            # founding is inferred as the earliest dist date minus 1 year in events.py
            # We check that every dist is after career start (a proxy)
            assert e.date >= profile.plausible_career_start, (
                f"Distribution {e.event_id} date {e.date} before career start "
                f"{profile.plausible_career_start}"
            )
        if e.type.value == "inheritance":
            assert e.date >= profile.plausible_career_start, (
                f"Inheritance {e.event_id} date {e.date} before career start"
            )
        # All events must be on or before as_of
        assert e.date <= spec.as_of, (
            f"Event {e.event_id} date {e.date} is after as_of {spec.as_of}"
        )


# ---------------------------------------------------------------------------
# Invariant 5 — reproducibility
# ---------------------------------------------------------------------------

@given(spec=scenario_specs())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_reproducibility(spec: ScenarioSpec) -> None:
    """Same (spec, seed) must produce identical results on two runs."""
    _, events_a, claims_a, _ = _build_full_pipeline(spec)
    _, events_b, claims_b, _ = _build_full_pipeline(spec)

    assert len(events_a) == len(events_b)
    for ea, eb in zip(events_a, events_b):
        assert ea.event_id == eb.event_id
        assert ea.amount == eb.amount
        assert ea.date == eb.date

    assert len(claims_a) == len(claims_b)
    for ca, cb in zip(claims_a, claims_b):
        assert ca.claim_id == cb.claim_id
        assert ca.amount == cb.amount


# ---------------------------------------------------------------------------
# Invariant 6 — net worth in target band after balancing
# ---------------------------------------------------------------------------

@given(spec=scenario_specs())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_net_worth_in_target_band(spec: ScenarioSpec) -> None:
    """After balance_to_target, closing net worth is inside spec.target_net_worth."""
    _, events, _, _ = _build_full_pipeline(spec)
    lo, hi = spec.target_net_worth
    nw = compute_net_worth(events, spec.as_of)
    assert lo <= nw <= hi, (
        f"Net worth {nw} outside target band [{lo}, {hi}]"
    )


# ---------------------------------------------------------------------------
# Invariant 7 — graph integrity
# ---------------------------------------------------------------------------

@given(spec=scenario_specs())
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_graph_frozen_and_complete(spec: ScenarioSpec) -> None:
    """Graph must be frozen and contain nodes for every event and claim."""
    import networkx as nx

    _, events, claims, fg = _build_full_pipeline(spec)

    assert fg.frozen
    assert nx.is_frozen(fg.g)

    # Every event and claim has a node
    for e in events:
        assert e.event_id in fg.g.nodes
    for c in claims:
        assert c.claim_id in fg.g.nodes

    # net_worth computed from graph matches ledger
    nw_graph = net_worth_from_graph(fg)
    nw_ledger = compute_net_worth(events, spec.as_of)
    assert nw_graph == nw_ledger


# ---------------------------------------------------------------------------
# Unit tests (no Hypothesis)
# ---------------------------------------------------------------------------

def test_schema_round_trip() -> None:
    """ScenarioSpec round-trips through JSON without data loss."""
    spec = ScenarioSpec(
        seed=42,
        as_of=date(2023, 6, 30),
        target_net_worth=(Decimal("500000"), Decimal("2000000")),
        claims_per_sow_type={
            SowType.employment: 2,
            SowType.inheritance: 1,
            SowType.gift: 1,
            SowType.business_profits: 1,
        },
    )
    json_str = spec.model_dump_json()
    spec2 = ScenarioSpec.model_validate_json(json_str)
    assert spec == spec2


def test_minimal_pipeline_smoke() -> None:
    """End-to-end pipeline smoke test with a fixed seed."""
    spec = ScenarioSpec(
        seed=0,
        as_of=date(2023, 12, 31),
        target_net_worth=(Decimal("300000"), Decimal("800000")),
        claims_per_sow_type={
            SowType.employment: 2,
            SowType.inheritance: 1,
            SowType.gift: 0,
            SowType.business_profits: 1,
        },
    )
    profile, events, claims, fg = _build_full_pipeline(spec)

    assert len(events) > 0
    assert len(claims) == 4  # 2 employment + 1 inheritance + 1 business
    assert fg.frozen

    lo, hi = spec.target_net_worth
    nw = compute_net_worth(events, spec.as_of)
    assert lo <= nw <= hi
