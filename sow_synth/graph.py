"""Stage 6 — Canonical fact graph assembly and freezing.

Assembles a NetworkX MultiDiGraph from Profile, Events, Claims, and Documents.
Edge types: 'states', 'covers', 'corroborates', 'derived_from'.

corroborates edges carry two extra attributes:
  evidential_role — what the document proves (EvidentialRole enum value)
  bundle_id       — which CorroborationBundle this document belongs to

The graph is frozen after assembly; no stage downstream may mutate it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from sow_synth.models import Claim, Document, Event, Profile
from sow_synth.spec import ScenarioSpec

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sow_synth.docplan import CorroborationBundle

_NODE_TYPE = "node_type"


@dataclass
class FactGraph:
    g: nx.MultiDiGraph
    spec: ScenarioSpec
    profile: Profile
    frozen: bool = False

    _events:    dict[str, Event]    = field(default_factory=dict)
    _claims:    dict[str, Claim]    = field(default_factory=dict)
    _documents: dict[str, Document] = field(default_factory=dict)

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
        docs = []
        for src, dst, data in self.g.edges(data=True):
            if data.get("edge_type") == "corroborates" and dst == claim_id:
                docs.append(self._documents[src])
        return docs

    def bundle_id_for_claim(self, claim_id: str) -> str | None:
        for _, dst, data in self.g.edges(data=True):
            if data.get("edge_type") == "corroborates" and dst == claim_id:
                return data.get("bundle_id")
        return None

    def _add_node(self, node_id: str, node_type: str, data: Any) -> None:
        self.g.add_node(node_id, node_type=node_type, data=data)

    def _add_edge(self, src: str, dst: str, edge_type: str, **attrs: Any) -> None:
        self.g.add_edge(src, dst, edge_type=edge_type, **attrs)


def assemble_graph(
    spec: ScenarioSpec,
    profile: Profile,
    events: list[Event],
    claims: list[Claim],
    bundles: "list[CorroborationBundle] | None" = None,
    client_history: "Document | None" = None,
) -> FactGraph:
    """Build the canonical fact graph and freeze it.

    bundles — list of CorroborationBundle from plan_bundles().  corroborates
    edges carry evidential_role and bundle_id attributes.

    client_history — already-rendered Document (role='client_history').
    """
    if bundles is None:
        bundles = []

    g = nx.MultiDiGraph()
    fg = FactGraph(g=g, spec=spec, profile=profile)

    fg._add_node(profile.client_id, "profile", profile)

    for e in events:
        fg._add_node(e.event_id, "event", e)
        fg._events[e.event_id] = e

    for claim in claims:
        fg._add_node(claim.claim_id, "claim", claim)
        fg._claims[claim.claim_id] = claim
        for eid in claim.covered_event_ids:
            fg._add_edge(claim.claim_id, eid, "covers")

    if client_history is not None:
        fg._add_node(client_history.doc_id, "document", client_history)
        fg._documents[client_history.doc_id] = client_history
        for claim in claims:
            fg._add_edge(client_history.doc_id, claim.claim_id, "states")
    else:
        for claim in claims:
            fg._add_edge(profile.client_id, claim.claim_id, "states")

    for bundle in bundles:
        for doc_spec in bundle.doc_specs:
            doc = doc_spec.to_document()
            fg._add_node(doc.doc_id, "document", doc)
            fg._documents[doc.doc_id] = doc
            for eid in doc_spec.source_event_ids:
                fg._add_edge(doc.doc_id, eid, "derived_from")
            fg._add_edge(
                doc.doc_id, bundle.claim_id, "corroborates",
                evidential_role=doc_spec.evidential_role.value,
                bundle_id=bundle.bundle_id,
            )

    nx.freeze(g)
    fg.frozen = True
    return fg


def net_worth_from_graph(fg: FactGraph) -> "Decimal":  # noqa: F821
    from sow_synth.ledger import compute_net_worth
    return compute_net_worth(fg.events, fg.spec.as_of)
