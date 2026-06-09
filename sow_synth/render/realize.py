"""Stage 8 — Surface realization (thin delegation wrapper).

All rendering logic now lives in sow_synth.formats.realize.
"""
from sow_synth.formats.realize import (  # noqa: F401
    realize_all,
    realize_document,
    _monthly_split,
    _fmt,
    _MONTHS,
    _n_pages_for_gross,
    _build_page1_ctx,
    _build_page2_ctx,
    _build_page3_ctx,
    _build_page4_ctx,
)
