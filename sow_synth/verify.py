"""Stage 9 — Verify-and-repair.

After surface realization, check every rendered document against its
verify_hints (set in Stage 5, docplan).  Precision-aware comparison:
  exact       — must match to the penny
  rounded     — within £1 / $1
  approximate — within 10 %
  narrative   — skip numeric check (amount embedded in prose)

On mismatch, overwrite the offending key_value in-place (repair).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from sow_synth.docplan import DocumentPlan
from sow_synth.graph import FactGraph
from sow_synth.models import Document


@dataclass
class VerifyError:
    doc_id: str
    field: str
    found: str
    expected: str
    repaired: bool = False

    def __str__(self) -> str:
        status = "REPAIRED" if self.repaired else "ERROR"
        return f"[{status}] {self.doc_id} / {self.field}: found={self.found!r} expected={self.expected!r}"


def _parse_amount(value: str) -> Decimal | None:
    parts = value.strip().split()
    raw = parts[-1] if parts else value
    try:
        return Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return None


def _amounts_match(found: Decimal, expected: Decimal, precision: str) -> bool:
    if precision == "exact":
        return found == expected
    if precision == "rounded":
        return abs(found - expected) <= Decimal("1.00")
    if precision == "approximate":
        if expected == 0:
            return found == 0
        return abs(found - expected) / expected <= Decimal("0.10")
    # narrative — skip
    return True


def _repair_key_value(doc: Document, key: str, correct_value: str) -> None:
    for page in doc.pages:
        for kv in page.key_values:
            if kv.key == key:
                old = kv.value
                kv.value = correct_value
                for line in page.lines:
                    if key in line.text and old in line.text:
                        line.text = line.text.replace(old, correct_value)
                return


def verify_document(
    doc: Document,
    plan: DocumentPlan,
    repair: bool = True,
) -> list[VerifyError]:
    """Check doc's key_values against plan.verify_hints."""
    errors: list[VerifyError] = []

    # Build a flat key_value map from all pages (first occurrence wins)
    kv_map: dict[str, str] = {}
    for page in doc.pages:
        for kv in page.key_values:
            kv_map.setdefault(kv.key, kv.value)

    for hint in plan.verify_hints:
        key = hint["key"]
        expected_str = hint["expected"]
        precision = hint.get("precision", "exact")

        if precision == "narrative":
            continue

        found_str = kv_map.get(key)
        if found_str is None:
            continue  # field not present in this doc format; skip

        found = _parse_amount(found_str)
        expected = _parse_amount(expected_str)
        if found is None or expected is None:
            errors.append(VerifyError(doc.doc_id, key, found_str or "", expected_str))
            continue

        if not _amounts_match(found, expected, precision):
            err = VerifyError(
                doc_id=doc.doc_id,
                field=key,
                found=str(found),
                expected=expected_str,
                repaired=False,
            )
            if repair:
                _repair_key_value(doc, key, expected_str)
                err.repaired = True
            errors.append(err)

    return errors


def verify_all(
    fg: FactGraph,
    plans: list[DocumentPlan] | None = None,
    repair: bool = True,
) -> list[VerifyError]:
    """Verify all rendered documents.  Requires plans list for hint lookup."""
    plan_map: dict[str, DocumentPlan] = {}
    if plans:
        plan_map = {p.doc_id: p for p in plans}

    all_errors: list[VerifyError] = []
    for doc in fg.documents:
        if not doc.pages:
            continue
        plan = plan_map.get(doc.doc_id)
        if plan is None:
            continue
        all_errors.extend(verify_document(doc, plan, repair=repair))
    return all_errors
