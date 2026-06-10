"""Generate one synthetic SoW sample.

Every run writes a timestamped subfolder under outputs/ with all artefacts.
No arguments are required — defaults produce a complete run immediately.

Usage examples:
    python generate_sample.py
    python generate_sample.py --seed 7 --employment 2 --inheritance 1
    python generate_sample.py --seed 42 --client-history
    python generate_sample.py --seed 42 --client-history --llm-docs
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from sow_synth.claims import project_claims
from sow_synth.export import export_all
from sow_synth.docplan import plan_bundles
from sow_synth.events import generate_events
from sow_synth.graph import assemble_graph, net_worth_from_graph
from sow_synth.ledger import balance_to_target
from sow_synth.models import Document, SowType
from sow_synth.ocr import apply_noise
from sow_synth.profile import resolve_profile
from sow_synth.render.document import render_all_bundles
from sow_synth.spec import ScenarioSpec
from sow_synth.telemetry import Telemetry
from sow_synth.verify import verify_all

_OUTPUTS_ROOT = Path(__file__).parent / "outputs"


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
    p.add_argument("--client-history", action="store_true",
                   help="Generate client history via LLM (requires Azure env vars)")
    p.add_argument("--llm-docs", action="store_true",
                   help="Generate corroboration documents via LLM (requires Azure env vars)")
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


def _run_dir(seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _OUTPUTS_ROOT / f"run_{ts}_seed{seed}"


def run(
    spec: ScenarioSpec,
    noise_level: float = 0.05,
    generate_client_history: bool = False,
    llm_docs: bool = False,
) -> tuple[dict, Telemetry]:
    tel = Telemetry()
    rng = np.random.default_rng(spec.seed)

    llm = None
    if generate_client_history or llm_docs:
        from dotenv import load_dotenv
        load_dotenv()
        from sow_synth.llm import LlmClient
        llm = LlmClient(tel)

    with tel.step("stage_1_profile"):
        profile = resolve_profile(spec, rng)

    with tel.step("stage_2_events"):
        events = generate_events(profile, spec, rng)

    with tel.step("stage_3_ledger"):
        events = balance_to_target(events, spec, rng)

    with tel.step("stage_4_claims"):
        claims = project_claims(events)

    ch_doc = None
    if generate_client_history and llm is not None:
        from sow_synth.render.client_history import render_client_history
        ch_doc, claims = render_client_history(profile, events, claims, spec, llm, tel)

    with tel.step("stage_5_docplan"):
        bundles = plan_bundles(profile, events, claims, spec, rng)

    with tel.step("stage_6_graph"):
        fg = assemble_graph(spec, profile, events, claims,
                            bundles=bundles,
                            client_history=ch_doc)

    with tel.step("stage_8_corroboration"):
        render_all_bundles(bundles, fg._documents, llm=llm if llm_docs else None)

    with tel.step("stage_9_verify"):
        verify_errors = verify_all(fg, bundles=bundles)

    with tel.step("stage_10_ocr_noise"):
        noisy_docs = [apply_noise(doc, rng, noise_level=noise_level)
                      for doc in fg.documents]

    bundle_summary = [
        {
            "bundle_id": b.bundle_id,
            "claim_id":  b.claim_id,
            "documents": [
                {"doc_id": s.doc_id,
                 "doc_type": s.doc_type.value,
                 "evidential_role": s.evidential_role.value}
                for s in b.doc_specs
            ],
        }
        for b in bundles
    ]

    result = {
        "spec":            json.loads(spec.model_dump_json()),
        "profile":         json.loads(profile.model_dump_json()),
        "events":          [json.loads(e.model_dump_json()) for e in events],
        "claims":          [json.loads(c.model_dump_json()) for c in claims],
        "net_worth":       str(net_worth_from_graph(fg)),
        "bundles":         bundle_summary,
        "documents_clean": [json.loads(d.model_dump_json()) for d in fg.documents],
        "documents_noisy": [json.loads(d.model_dump_json()) for d in noisy_docs],
        "verify_errors":   [str(e) for e in verify_errors],
        "corroborates":    [
            {"doc_id": src, "claim_id": dst,
             "evidential_role": data.get("evidential_role"),
             "bundle_id": data.get("bundle_id")}
            for src, dst, data in fg.g.edges(data=True)
            if data.get("edge_type") == "corroborates"
        ],
        "states": [
            {"doc_id": src, "claim_id": dst}
            for src, dst, data in fg.g.edges(data=True)
            if data.get("edge_type") == "states"
        ],
    }
    return result, tel


def save_outputs(bundle: dict, tel: Telemetry, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "spec.json",         bundle["spec"])
    _write_json(run_dir / "profile.json",      bundle["profile"])
    _write_json(run_dir / "events.json",       bundle["events"])
    _write_json(run_dir / "claims.json",       bundle["claims"])
    _write_json(run_dir / "net_worth.json",    {"net_worth": bundle["net_worth"]})
    _write_json(run_dir / "bundles.json",      bundle["bundles"])
    _write_json(run_dir / "corroborates.json", bundle["corroborates"])
    _write_json(run_dir / "states.json",       bundle["states"])
    if bundle["verify_errors"]:
        _write_json(run_dir / "verify_errors.json", bundle["verify_errors"])

    clean_docs = [Document.model_validate(d) for d in bundle["documents_clean"]]
    noisy_docs = [Document.model_validate(d) for d in bundle["documents_noisy"]]
    export_all(clean_docs, noisy_docs, run_dir / "documents")
    tel.save_report(run_dir / "telemetry.txt")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def print_summary(bundle: dict, run_dir: Path) -> None:
    p  = bundle["profile"]
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
    print(f"CLAIMS + BUNDLES  ({len(bundle['claims'])} claims)")
    bundle_map = {b["claim_id"]: b for b in bundle["bundles"]}
    for c in bundle["claims"]:
        n_evts = len(c["covered_event_ids"])
        b = bundle_map.get(c["claim_id"])
        print(f"  [{c['sow_type']:<18}]  \xa3{Decimal(c['amount']):>14,.0f}"
              f"  ({n_evts} event{'s' if n_evts != 1 else ''})")
        if b:
            for d in b["documents"]:
                print(f"      {d['doc_type']:<30} [{d['evidential_role']}]")
    print()
    docs = bundle.get("documents_clean", [])
    if docs:
        print(f"DOCUMENTS  ({len(docs)} total)")
        errs = bundle.get("verify_errors", [])
        if errs:
            print(f"  VERIFY: {len(errs)} missing mandatory fact(s)")
            for e in errs[:5]:
                print(f"    {e}")
        else:
            print("  VERIFY: all mandatory facts present [OK]")
    print("=" * 60)
    print(f"\nOutputs: {run_dir}")
    print(f"  documents/index.html  — browse all documents")
    print(f"  bundles.json          — bundle composition per claim")
    print(f"  telemetry.txt         — timing and token usage")


def main() -> None:
    args    = parse_args()
    spec    = build_spec(args)
    run_dir = _run_dir(spec.seed)

    result, tel = run(
        spec,
        noise_level=args.noise,
        generate_client_history=args.client_history,
        llm_docs=args.llm_docs,
    )

    save_outputs(result, tel, run_dir)
    print_summary(result, run_dir)
    tel.print_report()


if __name__ == "__main__":
    main()
