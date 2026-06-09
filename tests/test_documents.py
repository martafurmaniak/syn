"""Tests for document planning, rendering, verify, and OCR noise."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from sow_synth.claims import project_claims
from sow_synth.docplan import plan_documents
from sow_synth.events import generate_events
from sow_synth.graph import assemble_graph
from sow_synth.ledger import balance_to_target
from sow_synth.models import EventType, SowType
from sow_synth.ocr import apply_noise
from sow_synth.profile import resolve_profile
from sow_synth.formats.realize import realize_all
from sow_synth.spec import ScenarioSpec
from sow_synth.verify import verify_all


def _build_sample(seed: int = 0, employment: int = 2, inheritance: int = 0,
                  gift: int = 0, business: int = 0):
    spec = ScenarioSpec(
        seed=seed,
        as_of=date(2023, 12, 31),
        target_net_worth=(Decimal("300000"), Decimal("900000")),
        claims_per_sow_type={
            SowType.employment:       employment,
            SowType.inheritance:      inheritance,
            SowType.gift:             gift,
            SowType.business_profits: business,
        },
    )
    rng = np.random.default_rng(spec.seed)
    profile = resolve_profile(spec, rng)
    events  = balance_to_target(generate_events(profile, spec, rng), spec, rng)
    claims  = project_claims(events)
    plans   = plan_documents(profile, events, claims, spec, rng)
    fg      = assemble_graph(spec, profile, events, claims, document_plans=plans)
    realize_all(plans, fg._documents)
    return spec, profile, events, claims, plans, fg


# ---------------------------------------------------------------------------
# Document planning
# ---------------------------------------------------------------------------

def test_one_document_per_claim():
    """One document is planned per claim (not per event)."""
    _, _, _, claims, plans, _ = _build_sample(seed=1, employment=3)
    assert len(plans) == len(claims)


def test_corroborates_edges_created_by_construction():
    """Every document plan has corroborates edges pointing at expected claims."""
    _, _, _, claims, plans, fg = _build_sample(seed=2, employment=2)
    claim_ids = {c.claim_id for c in claims}

    corr_edges = {
        (src, dst)
        for src, dst, data in fg.g.edges(data=True)
        if data.get("edge_type") == "corroborates"
    }
    for plan in plans:
        for cid in plan.corroborates_claim_ids:
            assert (plan.doc_id, cid) in corr_edges
            assert cid in claim_ids


def test_derived_from_edges_exist():
    """Every plan has derived_from edges linking its doc to source events."""
    _, _, events, _, plans, fg = _build_sample(seed=3)
    event_ids = {e.event_id for e in events}

    df_edges = {
        (src, dst)
        for src, dst, data in fg.g.edges(data=True)
        if data.get("edge_type") == "derived_from"
    }
    for plan in plans:
        for eid in plan.source_event_ids:
            assert (plan.doc_id, eid) in df_edges
            assert eid in event_ids


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_rendered_documents_have_pages():
    """After realize_all, every planned document has at least one page."""
    _, _, _, _, _, fg = _build_sample(seed=4)
    for doc in fg.documents:
        assert len(doc.pages) >= 1, f"{doc.doc_id} has no pages after rendering"


def test_rendered_documents_have_key_values():
    """Every rendered document exposes at least one KeyValue on page 1."""
    _, _, _, _, _, fg = _build_sample(seed=5)
    for doc in fg.documents:
        kv_keys = {kv.key for kv in doc.pages[0].key_values}
        assert len(kv_keys) >= 1, f"{doc.doc_id} has no key_values"


def test_verify_hints_primary_field_present():
    """The primary_amount_field from verify_hints appears in the document's key_values."""
    _, _, _, _, plans, fg = _build_sample(seed=6)
    for plan in plans:
        doc = fg.get_document(plan.doc_id)
        all_kv_keys = {kv.key for page in doc.pages for kv in page.key_values}
        for hint in plan.verify_hints:
            key = hint["key"]
            if hint.get("precision") == "narrative":
                continue
            assert key in all_kv_keys, (
                f"{doc.doc_id} ({plan.doc_type}): hint key '{key}' not in key_values {all_kv_keys}"
            )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def test_verify_passes_for_clean_documents():
    """verify_all returns no errors for a cleanly rendered sample."""
    _, _, _, _, plans, fg = _build_sample(seed=7)
    errors = verify_all(fg, plans=plans, repair=False)
    assert errors == [], f"Unexpected verify errors: {errors}"


def test_verify_repairs_tampered_amount():
    """If a primary amount key_value is corrupted, verify repairs it."""
    _, _, _, _, plans, fg = _build_sample(seed=8)

    # Corrupt the first document that has verify_hints
    plan = next((p for p in plans if p.verify_hints and
                 p.verify_hints[0].get("precision") != "narrative"), None)
    if plan is None:
        pytest.skip("No verifiable documents in this sample")

    doc = fg.get_document(plan.doc_id)
    key = plan.verify_hints[0]["key"]
    for page in doc.pages:
        for kv in page.key_values:
            if kv.key == key:
                kv.value = "1.00"
                break

    errors = verify_all(fg, plans=plans, repair=True)
    repaired = [e for e in errors if e.repaired]
    assert len(repaired) >= 1

    errors_after = verify_all(fg, plans=plans, repair=False)
    assert errors_after == []


# ---------------------------------------------------------------------------
# OCR noise
# ---------------------------------------------------------------------------

def test_noise_produces_different_text():
    """Noisy document differs from clean at noise_level=0.3."""
    _, _, _, _, _, fg = _build_sample(seed=9)
    doc = next(iter(fg._documents.values()))
    rng = np.random.default_rng(99)
    noisy = apply_noise(doc, rng, noise_level=0.3)

    clean_text = " ".join(line.text for page in doc.pages for line in page.lines)
    noisy_text = " ".join(line.text for page in noisy.pages for line in page.lines)
    assert clean_text != noisy_text


def test_noise_zero_produces_identical_copy():
    """noise_level=0 returns a deep copy with identical content."""
    _, _, _, _, _, fg = _build_sample(seed=10)
    doc = next(iter(fg._documents.values()))
    rng = np.random.default_rng(0)
    copy = apply_noise(doc, rng, noise_level=0)

    for page_orig, page_copy in zip(doc.pages, copy.pages):
        for line_o, line_c in zip(page_orig.lines, page_copy.lines):
            assert line_o.text == line_c.text


def test_noise_preserves_key_values():
    """key_values are not text-corrupted (only confidence-degraded)."""
    _, _, _, _, _, fg = _build_sample(seed=11)
    doc = next(iter(fg._documents.values()))
    rng = np.random.default_rng(0)
    noisy = apply_noise(doc, rng, noise_level=0.5)

    for page_orig, page_noisy in zip(doc.pages, noisy.pages):
        for kv_o, kv_n in zip(page_orig.key_values, page_noisy.key_values):
            assert kv_o.key   == kv_n.key,   "key_value key was mutated"
            assert kv_o.value == kv_n.value,  "key_value value was mutated"
            assert kv_n.confidence <= kv_o.confidence + 1e-9


# ---------------------------------------------------------------------------
# Graph integrity after full pipeline
# ---------------------------------------------------------------------------

def test_graph_frozen_with_documents():
    """Graph remains frozen after rendering fills pages."""
    import networkx as nx
    _, _, _, _, _, fg = _build_sample(seed=12)
    assert fg.frozen
    assert nx.is_frozen(fg.g)


@given(seed=st.integers(min_value=0, max_value=999))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_full_pipeline_no_verify_errors(seed: int):
    """Property: verify passes for any seed with employment claims."""
    _, _, _, _, plans, fg = _build_sample(seed=seed, employment=2)
    errors = verify_all(fg, plans=plans, repair=False)
    assert errors == [], f"seed={seed}: verify errors: {errors}"
