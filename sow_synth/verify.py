"""Stage 9 — Verify mandatory facts.

After rendering, check that every value in DocumentSpec.mandatory_facts
appears in the document's concatenated page_text.  This ensures the renderer
(LLM or code fallback) embedded all required figures verbatim.

No repair: if a value is missing the error is recorded and the sample is
flagged.  Silent repair of LLM output is not safe — it would corrupt ground
truth without a trace.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sow_synth.graph import FactGraph

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sow_synth.docplan import CorroborationBundle
    from sow_synth.models import Document


@dataclass
class VerifyError:
    doc_id: str
    field: str
    expected: str

    def __str__(self) -> str:
        return f"[MISSING] {self.doc_id} / {self.field}: {self.expected!r} not in page_text"


def _full_text(doc: "Document") -> str:
    raw = " ".join(p.page_text for p in doc.pages)
    return re.sub(r"<[^>]+>", " ", raw)


def verify_document(
    doc: "Document",
    mandatory_facts: dict[str, str],
) -> list[VerifyError]:
    text = _full_text(doc)
    errors: list[VerifyError] = []
    for key, value in mandatory_facts.items():
        if value and value not in text:
            errors.append(VerifyError(doc_id=doc.doc_id, field=key, expected=value))
    return errors


def verify_all(
    fg: FactGraph,
    bundles: "list[CorroborationBundle] | None" = None,
    **_kwargs,  # absorb legacy keyword args (plans=, repair=) silently
) -> list[VerifyError]:
    """Verify every rendered corroboration document against its mandatory_facts."""
    if not bundles:
        return []

    all_errors: list[VerifyError] = []
    for bundle in bundles:
        for doc_spec in bundle.doc_specs:
            doc = fg._documents.get(doc_spec.doc_id)
            if doc is None or not doc.pages:
                continue
            all_errors.extend(verify_document(doc, doc_spec.mandatory_facts))
    return all_errors
