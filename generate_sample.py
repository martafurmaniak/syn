"""CLI wrapper — generate one synthetic SoW sample and print a summary.

Usage examples:

    python generate_sample.py
    python generate_sample.py --seed 42 --net-worth-lo 500000 --net-worth-hi 2000000
    python generate_sample.py --seed 7 --employment 3 --inheritance 2 --gift 1 --business 1
    python generate_sample.py --seed 0 --output sample.json
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal

import numpy as np

from sow_synth.claims import project_claims
from sow_synth.events import generate_events
from sow_synth.graph import assemble_graph, net_worth_from_graph
from sow_synth.ledger import balance_to_target, compute_net_worth
from sow_synth.models import SowType
from sow_synth.profile import resolve_profile
from sow_synth.spec import ScenarioSpec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a synthetic SoW case file.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--as-of", type=date.fromisoformat, default=date(2023, 12, 31))
    p.add_argument("--net-worth-lo", type=float, default=500_000)
    p.add_argument("--net-worth-hi", type=float, default=2_000_000)
    p.add_argument("--employment",  type=int, default=2, metavar="N")
    p.add_argument("--inheritance", type=int, default=1, metavar="N")
    p.add_argument("--gift",        type=int, default=1, metavar="N")
    p.add_argument("--business",    type=int, default=1, metavar="N")
    p.add_argument("--output", type=str, default=None, metavar="FILE",
                   help="Write full JSON bundle to FILE (optional)")
    return p.parse_args()


def build_spec(args: argparse.Namespace) -> ScenarioSpec:
    return ScenarioSpec(
        seed=args.seed,
        as_of=args.as_of,
        target_net_worth=(Decimal(str(args.net_worth_lo)), Decimal(str(args.net_worth_hi))),
        claims_per_sow_type={
            SowType.employment:      args.employment,
            SowType.inheritance:     args.inheritance,
            SowType.gift:            args.gift,
            SowType.business_profits: args.business,
        },
    )


def run(spec: ScenarioSpec) -> dict:
    rng = np.random.default_rng(spec.seed)
    profile = resolve_profile(spec, rng)
    events  = generate_events(profile, spec, rng)
    events  = balance_to_target(events, spec, rng)
    claims  = project_claims(events)
    fg      = assemble_graph(spec, profile, events, claims)

    return {
        "spec": json.loads(spec.model_dump_json()),
        "profile": json.loads(profile.model_dump_json()),
        "events": [json.loads(e.model_dump_json()) for e in events],
        "claims": [json.loads(c.model_dump_json()) for c in claims],
        "net_worth": str(net_worth_from_graph(fg)),
    }


def print_summary(bundle: dict) -> None:
    p = bundle["profile"]
    lo, hi = bundle["spec"]["target_net_worth"]
    nw = Decimal(bundle["net_worth"])

    print("=" * 60)
    print("CLIENT")
    print(f"  Name        : {p['name']}")
    print(f"  DOB         : {p['date_of_birth']}  (age {p['age']})")
    print(f"  Occupation  : {p['occupation']}")
    print(f"  Domicile    : {p['domicile']}")
    print(f"  Industry    : {p['industry']}")
    print()
    print("NET WORTH")
    print(f"  Target band : £{Decimal(lo):,.0f} – £{Decimal(hi):,.0f}")
    print(f"  Computed    : £{nw:,.0f}")
    print()
    print(f"EVENTS  ({len(bundle['events'])} total)")
    by_type: dict[str, list] = {}
    for e in bundle["events"]:
        by_type.setdefault(e["type"], []).append(e)
    for etype, evts in sorted(by_type.items()):
        total = sum(Decimal(e["amount"]) for e in evts)
        print(f"  {etype:<35} {len(evts):>3} events   £{total:>14,.0f}")
    print()
    print(f"CLAIMS  ({len(bundle['claims'])} total)")
    for c in bundle["claims"]:
        n_evts = len(c["covered_event_ids"])
        print(f"  [{c['sow_type']:<18}]  £{Decimal(c['amount']):>14,.0f}  "
              f"({n_evts} event{'s' if n_evts != 1 else ''})")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    spec = build_spec(args)
    bundle = run(spec)
    print_summary(bundle)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, default=str)
        print(f"\nFull bundle written to: {args.output}")


if __name__ == "__main__":
    main()
