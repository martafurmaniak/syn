"""Tests for bundle planning, rendering, verify, and OCR noise."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from sow_synth.claims import project_claims
from sow_synth.docplan import plan_bundles
from sow_synth.events import generate_events
from sow_synth.graph import assemble_graph
from sow_synth.ledger import balance_to_target
from sow_synth.models import EvidentialRole, SowType
from sow_synth.ocr import apply_noise
from sow_synth.profile import resolve_profile
from sow_synth.render.document import render_all_bundles
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
    bundles = plan_bundles(profile, events, claims, spec, rng)
    fg      = assemble_graph(spec, profile, events, claims, bundles=bundles)
    render_all_bundles(bundles, fg._documents)   # code fallback (no LLM)
    return spec, profile, events, claims, bundles, fg


# ---------------------------------------------------------------------------
# Bundle planning
# ---------------------------------------------------------------------------

def test_one_bundle_per_claim():
    """One CorroborationBundle is produced per claim."""
    _, _, _, claims, bundles, _ = _build_sample(seed=1, employment=3)
    assert len(bundles) == len(claims)
    assert {b.claim_id for b in bundles} == {c.claim_id for c in claims}


def test_bundles_have_at_least_one_doc():
    """Every bundle has at least one DocumentSpec."""
    _, _, _, _, bundles, _ = _build_sample(seed=2, employment=2, inheritance=1)
    for b in bundles:
        assert len(b.doc_specs) >= 1, f"Bundle {b.bundle_id} has no docs"


def test_bundle_doc_ids_are_unique():
    """No two DocumentSpecs in the same run share a doc_id."""
    _, _, _, _, bundles, _ = _build_sample(seed=3, employment=2, gift=1)
    all_ids = [s.doc_id for b in bundles for s in b.doc_specs]
    assert len(all_ids) == len(set(all_ids)), "Duplicate doc_ids found"


def test_evidential_roles_are_valid():
    """Every DocumentSpec carries a valid EvidentialRole."""
    _, _, _, _, bundles, _ = _build_sample(seed=4, employment=2, business=1)
    valid_roles = set(EvidentialRole)
    for b in bundles:
        for s in b.doc_specs:
            assert s.evidential_role in valid_roles


def test_mandatory_facts_contain_amounts():
    """Every DocumentSpec has at least one mandatory_fact value containing a number."""
    import re
    _, _, _, _, bundles, _ = _build_sample(seed=5, employment=2)
    for b in bundles:
        for s in b.doc_specs:
            has_number = any(
                re.search(r"\d", v)
                for v in s.mandatory_facts.values()
            )
            assert has_number, f"{s.doc_id} ({s.doc_type}) has no numeric fact"


# ---------------------------------------------------------------------------
# Graph edges
# ---------------------------------------------------------------------------

def test_corroborates_edges_have_role_and_bundle():
    """Every corroborates edge carries evidential_role and bundle_id."""
    _, _, _, _, bundles, fg = _build_sample(seed=6, employment=2)
    corr_edges = [
        (src, dst, data)
        for src, dst, data in fg.g.edges(data=True)
        if data.get("edge_type") == "corroborates"
    ]
    assert len(corr_edges) > 0
    for src, dst, data in corr_edges:
        assert "evidential_role" in data, f"Missing evidential_role on {src}→{dst}"
        assert "bundle_id" in data,       f"Missing bundle_id on {src}→{dst}"


def test_all_bundle_docs_have_corroborates_edges():
    """Every doc in every bundle has a corroborates edge to the right claim."""
    _, _, _, _, bundles, fg = _build_sample(seed=7, employment=2, gift=1)
    corr_pairs = {
        (src, dst)
        for src, dst, data in fg.g.edges(data=True)
        if data.get("edge_type") == "corroborates"
    }
    for b in bundles:
        for s in b.doc_specs:
            assert (s.doc_id, b.claim_id) in corr_pairs, (
                f"{s.doc_id} has no corroborates edge to {b.claim_id}"
            )


def test_derived_from_edges_exist():
    """Every DocumentSpec's source events have derived_from edges."""
    _, _, events, _, bundles, fg = _build_sample(seed=8)
    event_ids = {e.event_id for e in events}
    df_edges = {
        (src, dst)
        for src, dst, data in fg.g.edges(data=True)
        if data.get("edge_type") == "derived_from"
    }
    for b in bundles:
        for s in b.doc_specs:
            for eid in s.source_event_ids:
                assert (s.doc_id, eid) in df_edges
                assert eid in event_ids


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_rendered_documents_have_pages():
    """After render_all_bundles, every document has at least one page."""
    _, _, _, _, _, fg = _build_sample(seed=9)
    for doc in fg.documents:
        if doc.role == "client_history":
            continue
        assert len(doc.pages) >= 1, f"{doc.doc_id} has no pages"


def test_rendered_pages_have_page_text():
    """Every rendered page has non-empty page_text."""
    _, _, _, _, _, fg = _build_sample(seed=10)
    for doc in fg.documents:
        if doc.role == "client_history":
            continue
        for page in doc.pages:
            assert page.page_text.strip(), (
                f"{doc.doc_id} page {page.page_number} empty"
            )


def test_mandatory_facts_present_in_page_text():
    """All mandatory_facts values appear in the rendered page_text."""
    import re
    _, _, _, _, bundles, fg = _build_sample(seed=11)
    for b in bundles:
        for s in b.doc_specs:
            doc = fg.get_document(s.doc_id)
            text = " ".join(re.sub(r"<[^>]+>", " ", p.page_text) for p in doc.pages)
            for key, value in s.mandatory_facts.items():
                if value:
                    assert value in text, (
                        f"{doc.doc_id} ({s.doc_type}): fact '{key}' = {value!r} missing"
                    )


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def test_verify_passes_for_code_rendered_docs():
    """verify_all returns no errors for code-rendered documents."""
    _, _, _, _, bundles, fg = _build_sample(seed=12)
    errors = verify_all(fg, bundles=bundles)
    assert errors == [], f"Unexpected verify errors: {errors}"


# ---------------------------------------------------------------------------
# OCR noise
# ---------------------------------------------------------------------------

def test_noise_produces_different_text():
    """At least one document changes under noise_level=0.9."""
    _, _, _, _, _, fg = _build_sample(seed=13)
    rng = np.random.default_rng(99)
    changed = False
    for doc in fg.documents:
        if not doc.pages:
            continue
        noisy = apply_noise(doc, rng, noise_level=0.9)
        if any(p_orig.page_text != p_noisy.page_text
               for p_orig, p_noisy in zip(doc.pages, noisy.pages)):
            changed = True
            break
    assert changed, "No document changed under noise_level=0.9"


def test_noise_zero_produces_identical_copy():
    """noise_level=0 returns a deep copy with identical page_text."""
    _, _, _, _, _, fg = _build_sample(seed=14)
    doc = next(d for d in fg.documents if d.pages)
    rng = np.random.default_rng(0)
    copied = apply_noise(doc, rng, noise_level=0)
    for p_orig, p_copy in zip(doc.pages, copied.pages):
        assert p_orig.page_text == p_copy.page_text


def test_noise_preserves_page_numbers():
    """OCR noise does not alter page_number values."""
    _, _, _, _, _, fg = _build_sample(seed=15)
    doc = next(d for d in fg.documents if d.pages)
    rng = np.random.default_rng(0)
    noisy = apply_noise(doc, rng, noise_level=0.5)
    for p_orig, p_noisy in zip(doc.pages, noisy.pages):
        assert p_orig.page_number == p_noisy.page_number


# ---------------------------------------------------------------------------
# Graph integrity
# ---------------------------------------------------------------------------

def test_graph_frozen_with_documents():
    import networkx as nx
    _, _, _, _, _, fg = _build_sample(seed=16)
    assert fg.frozen
    assert nx.is_frozen(fg.g)


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------

@given(seed=st.integers(min_value=0, max_value=999))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_full_pipeline_no_verify_errors(seed: int):
    """Property: verify passes for any seed with employment claims."""
    _, _, _, _, bundles, fg = _build_sample(seed=seed, employment=2)
    errors = verify_all(fg, bundles=bundles)
    assert errors == [], f"seed={seed}: verify errors: {errors}"
