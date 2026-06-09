"""CLI wrapper — generate one synthetic SoW sample and print a summary.

Usage examples:

    python generate_sample.py
    python generate_sample.py --seed 42 --net-worth-lo 500000 --net-worth-hi 2000000
    python generate_sample.py --seed 7 --employment 3 --inheritance 2 --gift 1 --business 1
    python generate_sample.py --seed 0 --output sample.json
    python generate_sample.py --seed 0 --output sample.json --noise 0.1
    python generate_sample.py --seed 42 --export-dir ./output
    python generate_sample.py --seed 42 --client-history --export-dir ./output
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np

from sow_synth.claims import project_claims
from sow_synth.export import export_all
from sow_synth.docplan import plan_documents
from sow_synth.events import generate_events
from sow_synth.graph import assemble_graph, net_worth_from_graph
from sow_synth.ledger import balance_to_target
from sow_synth.models import SowType
from sow_synth.ocr import apply_noise
from sow_synth.profile import resolve_profile
from sow_synth.formats.realize import realize_all
from sow_synth.spec import ScenarioSpec
from sow_synth.telemetry import Telemetry
from sow_synth.verify import verify_all


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
    p.add_argument("--noise", type=float, default=0.05, metavar="LEVEL",
                   help="OCR noise level 0-1 (default 0.05)")
    p.add_argument("--output", type=str, default=None, metavar="FILE",
                   help="Write full JSON bundle to FILE (optional)")
    p.add_argument("--export-dir", type=str, default=None, metavar="DIR",
                   help="Write per-document HTML + JSON to DIR")
    p.add_argument("--client-history", action="store_true",
                   help="Generate client history document via LLM (requires Azure env vars)")
    return p.parse_args()


def build_spec(args: argparse.Namespace) -> ScenarioSpec:
    return ScenarioSpec(
        seed=args.seed,
        as_of=args.as_of,
        target_net_worth=(Decimal(str(args.net_worth_lo)), Decimal(str(args.net_worth_hi))),
        claims_per_sow_type={
            SowType.employment:       args.employment,
            SowType.inheritance:      args.inheritance,
            SowType.gift:             args.gift,
            SowType.business_profits: args.business,
        },
    )


def run(spec: ScenarioSpec, noise_level: float = 0.05,
        generate_client_history: bool = False) -> tuple[dict, Telemetry]:
    tel = Telemetry()
    rng = np.random.default_rng(spec.seed)

    # Stage 1: profile
    with tel.step("stage_1_profile"):
        profile = resolve_profile(spec, rng)

    # Stage 2: events
    with tel.step("stage_2_events"):
        events = generate_events(profile, spec, rng)

    # Stage 3: ledger balancing
    with tel.step("stage_3_ledger"):
        events = balance_to_target(events, spec, rng)

    # Stage 4: claim projection
    with tel.step("stage_4_claims"):
        claims = project_claims(events)

    # Stage 8a (LLM): client history document
    ch_doc = None
    if generate_client_history:
        from dotenv import load_dotenv
        load_dotenv()
        from sow_synth.llm import LlmClient
        from sow_synth.render.client_history import render_client_history
        llm = LlmClient(tel)
        # LLM calls are recorded directly into tel by LlmClient; no outer step needed
        ch_doc, claims = render_client_history(profile, events, claims, spec, llm, tel)

    # Stage 5: document planning (uses updated claims if asserted_text was set)
    with tel.step("stage_5_docplan"):
        doc_plans = plan_documents(profile, events, claims, spec, rng)

    # Stage 6: graph assembly + freeze
    with tel.step("stage_6_graph"):
        fg = assemble_graph(spec, profile, events, claims,
                            document_plans=doc_plans,
                            client_history=ch_doc)

    # Stage 8b: surface realization of corroboration docs (template-driven)
    with tel.step("stage_8_corroboration"):
        realize_all(doc_plans, fg._documents)

    # Stage 9: verify-and-repair
    with tel.step("stage_9_verify"):
        verify_errors = verify_all(fg, plans=doc_plans, repair=True)

    # Stage 10: OCR noise
    with tel.step("stage_10_ocr_noise"):
        noisy_docs = [apply_noise(doc, rng, noise_level=noise_level)
                      for doc in fg.documents]

    bundle = {
        "spec":              json.loads(spec.model_dump_json()),
        "profile":           json.loads(profile.model_dump_json()),
        "events":            [json.loads(e.model_dump_json()) for e in events],
        "claims":            [json.loads(c.model_dump_json()) for c in claims],
        "net_worth":         str(net_worth_from_graph(fg)),
        "documents_clean":   [json.loads(d.model_dump_json()) for d in fg.documents],
        "documents_noisy":   [json.loads(d.model_dump_json()) for d in noisy_docs],
        "verify_errors":     [str(e) for e in verify_errors],
        "corroborates":      [
            {"doc_id": src, "claim_id": dst}
            for src, dst, data in fg.g.edges(data=True)
            if data.get("edge_type") == "corroborates"
        ],
        "states":            [
            {"doc_id": src, "claim_id": dst}
            for src, dst, data in fg.g.edges(data=True)
            if data.get("edge_type") == "states"
        ],
    }
    return bundle, tel


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
    print(f"  Target band : \xa3{Decimal(lo):,.0f} – \xa3{Decimal(hi):,.0f}")
    print(f"  Computed    : \xa3{nw:,.0f}")
    print()
    print(f"EVENTS  ({len(bundle['events'])} total)")
    by_type: dict[str, list] = {}
    for e in bundle["events"]:
        by_type.setdefault(e["type"], []).append(e)
    for etype, evts in sorted(by_type.items()):
        total = sum(Decimal(e["amount"]) for e in evts)
        print(f"  {etype:<35} {len(evts):>3} events   \xa3{total:>14,.0f}")
    print()
    print(f"CLAIMS  ({len(bundle['claims'])} total)")
    for c in bundle["claims"]:
        n_evts = len(c["covered_event_ids"])
        has_text = "[text set]" if c.get("asserted_text") else ""
        print(f"  [{c['sow_type']:<18}]  \xa3{Decimal(c['amount']):>14,.0f}  "
              f"({n_evts} event{'s' if n_evts != 1 else ''})  {has_text}")
    print()

    docs = bundle.get("documents_clean", [])
    corr = bundle.get("corroborates", [])
    if docs:
        print(f"DOCUMENTS  ({len(docs)} total, {len(corr)} corroborates edges)")
        for d in docs:
            n_pages = len(d.get("pages", []))
            if d["role"] == "client_history":
                print(f"  [{d['doc_type']:<22}]  {n_pages} page(s)  [client history]")
            else:
                linked = [e["claim_id"] for e in corr if e["doc_id"] == d["doc_id"]]
                print(f"  [{d['doc_type']:<22}]  {n_pages} page(s)  -> {', '.join(linked) or 'none'}")
        errs = bundle.get("verify_errors", [])
        if errs:
            print(f"\n  VERIFY: {len(errs)} issue(s)")
            for e in errs:
                print(f"    {e}")
        else:
            print("\n  VERIFY: all figures trace to fact layer [OK]")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    spec = build_spec(args)
    bundle, tel = run(spec, noise_level=args.noise,
                      generate_client_history=args.client_history)
    print_summary(bundle)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, default=str)
        print(f"\nFull bundle written to: {args.output}")

    if args.export_dir:
        from sow_synth.models import Document
        clean_docs = [Document.model_validate(d) for d in bundle["documents_clean"]]
        noisy_docs = [Document.model_validate(d) for d in bundle["documents_noisy"]]
        export_dir = Path(args.export_dir)
        export_all(clean_docs, noisy_docs, export_dir)
        index = export_dir / "index.html"
        print(f"\nHTML + JSON exported to: {export_dir}")
        print(f"Open in browser:         {index}")

    tel.print_report()


if __name__ == "__main__":
    main()
