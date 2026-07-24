"""Structural invariants of the identity graphs (`identity/*.graph.json`).

The loader (`sandbox.latent._adjacency`) skips pure ``{"_comment": ...}`` divider objects inside
the ``nodes``/``edges`` arrays. These tests are the guard that makes that skip safe: every skipped
entry must carry ONLY the ``_comment`` key (a typo'd real node cannot vanish silently), and the
loader's node census must equal the raw-JSON census. The remaining invariants (referential
integrity, connectedness, unique relation triples) are what `_spectral_embed` and the shuffle
control assume; they hold for every graph under `identity/`, including control fixtures.
Invariants, not exact counts — the graphs grow with authored content.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sandbox.latent import load_identity_anchors

D = 8
GRAPHS = sorted((Path(__file__).resolve().parents[1] / "identity").glob("*.graph.json"))


def _load(graph_path: Path) -> dict:
    data = json.loads(graph_path.read_text())
    return {
        "nodes": [n for n in data["nodes"] if "id" in n],
        "edges": [e for e in data["edges"] if "src" in e and "dst" in e],
        "node_comments": [n for n in data["nodes"] if "id" not in n],
        "edge_comments": [e for e in data["edges"] if "src" not in e or "dst" not in e],
    }


@pytest.fixture(params=GRAPHS, ids=[p.name for p in GRAPHS])
def graph_path(request) -> Path:
    return request.param


def test_graphs_discovered():
    assert GRAPHS, "no identity/*.graph.json files found"


def test_graph_parses_nonempty(graph_path):
    g = _load(graph_path)
    assert g["nodes"] and g["edges"]


def test_real_nodes_unique_ids(graph_path):
    g = _load(graph_path)
    ids = [n["id"] for n in g["nodes"]]
    assert len(ids) >= 22  # the pre-enrichment floor; grows with authored content
    assert len(ids) == len(set(ids))
    assert all(isinstance(i, str) and i for i in ids)


def test_comment_entries_carry_only_comment_key(graph_path):
    """The guard that makes the loader's skip-filter safe: only pure divider objects are ever
    skipped. A real node/edge with a missing or typo'd key must fail HERE, not vanish."""
    g = _load(graph_path)
    for entry in g["node_comments"] + g["edge_comments"]:
        assert set(entry) == {"_comment"}, f"non-comment entry would be silently skipped: {entry}"


def test_edge_endpoints_exist_no_self_loops(graph_path):
    g = _load(graph_path)
    ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["src"] in ids and e["dst"] in ids, f"dangling edge {e['src']}→{e['dst']}"
        assert e["src"] != e["dst"], f"self-loop at {e['src']}"


def test_relation_triples_unique(graph_path):
    """Duplicate (src, dst) pairs with *different* relations are allowed (intended reinforcement —
    they accumulate adjacency weight); an exact duplicate triple is an authoring error."""
    g = _load(graph_path)
    triples = [(e["src"], e["dst"], e.get("relation", "")) for e in g["edges"]]
    assert len(triples) == len(set(triples))


def test_connected_undirected(graph_path):
    """`_spectral_embed` skips exactly one trivial eigenvector; a disconnected graph has a
    degenerate zero eigenspace, so connectedness is load-bearing for the embedding."""
    g = _load(graph_path)
    ids = [n["id"] for n in g["nodes"]]
    adj: dict[str, set[str]] = {i: set() for i in ids}
    for e in g["edges"]:
        adj[e["src"]].add(e["dst"])
        adj[e["dst"]].add(e["src"])
    seen: set[str] = set()
    stack = [ids[0]]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(adj[u] - seen)
    assert len(seen) == len(ids), f"graph is disconnected: reached {len(seen)}/{len(ids)} nodes"


def test_loader_skips_comment_entries(graph_path):
    """The loader contract, tied to the raw-JSON census: one anchor per real node, all finite."""
    g = _load(graph_path)
    ids, anchors = load_identity_anchors(D, graph_path=graph_path)
    assert len(ids) == len(g["nodes"])
    assert anchors.shape == (len(ids), D)
    assert np.all(np.isfinite(anchors))
