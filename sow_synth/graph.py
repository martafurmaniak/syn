"""Stage 6 — Canonical fact graph assembly and freezing.

Assembles a NetworkX MultiDiGraph from Profile, Events, Claims, and Documents.
Edge types: 'states', 'covers', 'corroborates', 'derived_from'.

The graph is frozen after assembly; no stage downstream may mutate it (except
the perturbation stage, which works on a deep copy with explicit mutation logging).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from sow_synth.models import Claim, Document, Event, Profile
from sow_synth.spec import ScenarioSpec

# Avoid a circular import: docplan imports models but not graph
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sow_synth.docplan import DocumentPlan


# Node-type labels stored as a node attribute so consumers can filter easily
_NODE_TYPE = "node_type"


@dataclass
class FactGraph:
    """Wrapper around a NetworkX MultiDiGraph with typed accessors."""
    g: nx.MultiDiGraph
    spec: ScenarioSpec
    profile: Profile
    frozen: bool = False

    # Convenience index caches (populated during assembly)
    _events: dict[str, Event] = field(default_factory=dict)
    _claims: dict[str, Claim] = field(default_factory=dict)
    _documents: dict[str, Document] = field(default_factory=dict)

    # -----------------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------------

    def get_event(self, event_id: str) -> Event:
        return self._events[event_id]

    def get_claim(self, claim_id: str) -> Claim:
        return self._claims[claim_id]

    def get_document(self, doc_id: str) -> Document:
        return self._documents[doc_id]

    @property
    def events(self) -> list[Event]:
        return list(self._events.values())

    @property
    def claims(self) -> list[Claim]:
        return list(self._claims.values())

    @property
    def documents(self) -> list[Document]:
        return list(self._documents.values())

    def corroborating_docs_for_claim(self, claim_id: str) -> list[Document]:
        """Documents that have a 'corroborates' edge to this claim."""
        docs = []
        for src, dst, data in self.g.edges(data=True):
            if data.get("edge_type") == "corroborates" and dst == claim_id:
                docs.append(self._documents[src])
        return docs

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _add_node(self, node_id: str, node_type: str, data: Any) -> None:
        self.g.add_node(node_id, node_type=node_type, data=data)

    def _add_edge(self, src: str, dst: str, edge_type: str, **attrs: Any) -> None:
        self.g.add_edge(src, dst, edge_type=edge_type, **attrs)


def assemble_graph(
    spec: ScenarioSpec,
    profile: Profile,
    events: list[Event],
    claims: list[Claim],
    document_plans: "list[DocumentPlan] | None" = None,
    client_history: "Document | None" = None,
) -> FactGraph:
    """Build the canonical fact graph and freeze it.

    document_plans may be empty at Phase 1 (no documents yet).  When provided,
    Document nodes are created with empty pages, and corroborates / derived_from
    edges are wired in by construction.  Surface realization (Stage 8) later
    populates the pages in-place; NetworkX freeze does not prevent mutating
    node attribute objects, only structural graph changes.

    client_history is an already-rendered Document (role='client_history').
    When provided, it is added as a node with 'states' edges to every claim.
    """
    if document_plans is None:
        document_plans = []

    g = nx.MultiDiGraph()
    fg = FactGraph(g=g, spec=spec, profile=profile)

    # -- Profile node --
    fg._add_node(profile.client_id, "profile", profile)

    # -- Event nodes --
    for e in events:
        fg._add_node(e.event_id, "event", e)
        fg._events[e.event_id] = e

    # -- Claim nodes + covers edges --
    for claim in claims:
        fg._add_node(claim.claim_id, "claim", claim)
        fg._claims[claim.claim_id] = claim
        for eid in claim.covered_event_ids:
            fg._add_edge(claim.claim_id, eid, "covers")

    # -- states edges: client_history → claim (preferred) or profile → claim (fallback) --
    if client_history is not None:
        fg._add_node(client_history.doc_id, "document", client_history)
        fg._documents[client_history.doc_id] = client_history
        for claim in claims:
            fg._add_edge(client_history.doc_id, claim.claim_id, "states")
    else:
        for claim in claims:
            fg._add_edge(profile.client_id, claim.claim_id, "states")

    # -- Corroboration document nodes + corroborates + derived_from edges --
    for plan in document_plans:
        doc = plan.to_document()           # empty pages — rendered later
        fg._add_node(doc.doc_id, "document", doc)
        fg._documents[doc.doc_id] = doc
        for eid in plan.source_event_ids:
            fg._add_edge(doc.doc_id, eid, "derived_from")
        for cid in plan.corroborates_claim_ids:
            fg._add_edge(doc.doc_id, cid, "corroborates")

    # -- Freeze graph structure --
    nx.freeze(g)
    fg.frozen = True

    return fg


def net_worth_from_graph(fg: FactGraph) -> "Decimal":  # noqa: F821
    """Recompute net worth directly from the graph's event nodes."""
    from sow_synth.ledger import compute_net_worth
    return compute_net_worth(fg.events, fg.spec.as_of)
