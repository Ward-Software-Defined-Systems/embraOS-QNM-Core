"""graph_poisson.py — 𝔤(G)*: the identity graph becomes the bracket (§9.17, increment 5).

State ``(p, w)`` on the dual of the Dani–Mainkar graph algebra 𝔤(G): one vertex momentum per
identity-graph node (``p ∈ ℝ^V`` — the arena), one edge momentum per authored relation
(``w ∈ ℝ^E`` — the charge). The Lie–Poisson flow is

    ṗ = J(w) ∇_p H        (J = the weighted skew-adjacency of the identity graph)
    ẇ = 0                  identically, for ANY H — this line IS the Casimir theorem here.

**ψ := w.** The Casimirs are coordinates, so conservation is a STATE PARTITION: the stepper has
no write path to ``w``, and the §9.17 bar-1 drift is pre-registered as a bit-level EQUALITY
(``np.array_equal``), not a tolerance. The loader hands ``w₀`` out write-locked (numpy
read-only flag): an accidental write raises instead of silently passing the bar — the
partition is mechanical, not conventional. The only function here that returns a modified
charge is ``weaken`` — the †-class made per-edge legible (graph surgery), and it is
deliberately not a ``Symbol``.

Every window Hamiltonian at this increment is quadratic+linear (``H₀ = ½|p|²`` placeholder
plus ``kick``/``quad`` events), so each ``dt`` step applies the EXACT affine flow map
``p ← Φp + b`` with ``Φ = exp(dt·J(w)M)`` (augmented-matrix exponential, one per symbol) —
"one linear-affine map per step" is this bracket's "one rotation per step". ``w`` is never an
operand of the stepper.

Graph-parametric by construction: ``n``, ``m``, ``rank J``, the index, and perfect-matching
status are COMPUTED at load and recorded, never assumed — one module serves any identity
graph; a different soul is a different file. Placeholder charge values (§9.17 conventions):
``w₀`` = relation-triple counts per distinct pair (D1 = aggregate-per-pair; content lands as
the authored weight table, next increment).

The silent class on this bracket is ``H = f(w)`` (``∇_p H ≡ 0``). Recorded contrast with
so(3)*: the isotropic quad (``H ∝ |p|²``) is NOT silent here — ``|p|²`` is not a Casimir.

ζ ∈ ℝ^E is the per-edge signed area swept about the genesis gauge (``x = p − p₀``): memory
with the same shape as identity, one accumulator per authored relation. Flat planes have no
antipode/branch trap; the §9.17 scale certificate (``max|p|`` guard + per-window coercivity)
replaces the antipode certificate.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

from sandbox.replica_test import auc  # the phase-one AUC, tie-aware

Array = NDArray[np.float64]

IDENTITY_GRAPH = Path(__file__).resolve().parents[1] / "identity" / "Embra_IDENTITY-SOUL.graph.json"
MERIDIAN_GRAPH = Path(__file__).resolve().parents[1] / "identity" / "Meridian_IDENTITY-SOUL.graph.json"
EMBRA_TABLE = Path(__file__).resolve().parents[1] / "identity" / "Embra_WEIGHTS.table.json"
MERIDIAN_TABLE = Path(__file__).resolve().parents[1] / "identity" / "Meridian_WEIGHTS.table.json"

EPS_KICK = 0.5
EPS_QUAD = 0.5
# §9.18: on the AUTHORED geometry the coercivity bound at the pinned edge is
# amp < √(1/(I_np·I_pos)) = 0.488 (soul_line 3.0 × value 1.4) — 0.5 fails it (min window
# eig −0.011, caught at sizing time); re-pinned for authored runs. §9.17's 0.5 stands
# unchanged on the placeholder geometry it was recorded on.
EPS_QUAD_AUTHORED = 0.4
# §9.17 pinned alphabet nodes: x/y kick the graph's own triple-relation pair (adjacent,
# w₀ = 3); z kicks a node adjacent to neither; q is the edge-quad on the x–y edge.
PINNED_X = "no_pretense"
PINNED_Y = "precision_over_spectacle"
PINNED_Z = "always_becoming_never_finished"
SIN2_WORD = "xyqzxzqy"
MAX_P_GUARD = 50.0


# --------------------------------------------------------------------------- #
# The identity graph AS the algebra: load, orient, aggregate, derive.
# --------------------------------------------------------------------------- #
def _read_only(a: Array) -> Array:
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class GraphAlgebra:
    """𝔤(G) for one identity graph. ``ids`` are sorted (the pinned lex orientation: every
    edge is (u, v) with u < v by node id, ``Z_vu = −Z_uv``); ``edges`` holds index pairs
    (i, j), i < j; ``w0`` is the placeholder charge (relation-triple counts — write-locked).
    Derived quantities are computed at load, never assumed (the §9.17 graph-parametric
    clause): ``rank_j0 = rank J(w₀)``, ``index = n + m − rank``, ``perfect_matching``
    (rank == n ⟺ the generic leaf is the whole arena). ``inertia`` (§9.18, D5(a)) is the
    per-node inertia vector from the authored table — ``None`` on the placeholder path, whose
    recorded behavior is bit-identical to §9.17."""

    ids: tuple[str, ...]
    edges: Array  # (m, 2) int, i < j, sorted
    w0: Array  # (m,) float, read-only
    n: int
    m: int
    rank_j0: int
    index: int
    perfect_matching: bool
    _node_index: dict[str, int] = field(repr=False)
    _pair_index: dict[tuple[int, int], int] = field(repr=False)
    inertia: Array | None = None  # (n,) float > 0, read-only — authored souls only

    def node_index(self, node_id: str) -> int:
        return self._node_index[node_id]

    def edge_of(self, u_id: str, v_id: str) -> int:
        i, j = self._node_index[u_id], self._node_index[v_id]
        return self._pair_index[(min(i, j), max(i, j))]

    def edge_label(self, k: int) -> str:
        i, j = self.edges[k]
        return f"{self.ids[i]}—{self.ids[j]}"

    def J(self, w: Array) -> Array:
        """The Lie–Poisson tensor: the weighted skew-adjacency. Built fresh from ``w``
        (``J[u,v] = +w_e``, ``J[v,u] = −w_e`` per oriented edge)."""
        w = np.asarray(w, float)
        J = np.zeros((self.n, self.n))
        i, j = self.edges[:, 0], self.edges[:, 1]
        J[i, j] = w
        J[j, i] = -w
        return J


def load_graph(graph_path: Path = IDENTITY_GRAPH) -> GraphAlgebra:
    """Graph JSON → 𝔤(G). Pure ``{"_comment": ...}`` divider objects are permitted and skipped
    (``tests/test_identity_graph.py`` guards that nothing else ever is). D1 aggregation:
    parallel relation triples on one pair accumulate into that pair's single ``w₀`` count.
    Preconditions checked here: no isolated vertices (the center of 𝔤(G) is exactly the edge
    span only then); connectedness is already a standing invariant of the identity tests."""
    data = json.loads(Path(graph_path).read_text())
    raw_ids = [n["id"] for n in data["nodes"] if "id" in n]
    ids = tuple(sorted(raw_ids))  # the pinned lex orientation
    idx = {nid: i for i, nid in enumerate(ids)}
    counts: dict[tuple[int, int], float] = {}
    for e in data["edges"]:
        if "src" not in e or "dst" not in e:
            continue
        i, j = idx[e["src"]], idx[e["dst"]]
        pair = (min(i, j), max(i, j))
        counts[pair] = counts.get(pair, 0.0) + float(e.get("weight", 1.0))
    pairs = sorted(counts)
    edges = np.array(pairs, dtype=int)
    w0 = _read_only(np.array([counts[p] for p in pairs], dtype=float))
    n, m = len(ids), len(pairs)
    if len(np.unique(edges)) < n:
        raise ValueError(f"{graph_path.name}: isolated vertices — the center of 𝔤(G) would exceed the edge span")
    pair_index = {p: k for k, p in enumerate(pairs)}
    alg = GraphAlgebra(
        ids=ids, edges=_read_only(edges), w0=w0, n=n, m=m,
        rank_j0=0, index=0, perfect_matching=False,
        _node_index=idx, _pair_index=pair_index,
    )
    rank = int(np.linalg.matrix_rank(alg.J(w0)))
    object.__setattr__(alg, "rank_j0", rank)
    object.__setattr__(alg, "index", n + m - rank)
    object.__setattr__(alg, "perfect_matching", rank == n)
    return alg


def load_table(table_path: Path) -> dict:
    return json.loads(Path(table_path).read_text())


def load_soul(graph_path: Path, table_path: Path) -> GraphAlgebra:
    """Graph + authored weight table → the SEALED soul (§9.18): ``w = table ∘ graph`` (sum
    over each pair's relation triples, direction-blind — the bracket's sign is the lex
    orientation, never src→dst), per-node inertias from the node-type table (D5(a)). The
    composition is genesis sealing: ``w`` is written HERE, once, from authored content, and
    by nothing else — it comes back write-locked, and rank/index are recomputed at the
    authored values, never carried over. An exact-zero composed edge is a loader-level error
    (a silent charge coordinate — the silence lesson in charge space)."""
    base = load_graph(graph_path)
    data = json.loads(Path(graph_path).read_text())
    table = load_table(table_path)
    if table.get("aggregation") != "sum":
        raise ValueError(f"{Path(table_path).name}: aggregation must be 'sum' (D1/§9.18)")
    tw, ni = table["relation_weights"], table["node_inertias"]
    rels = {e["relation"] for e in data["edges"] if "src" in e}
    types = {n["type"] for n in data["nodes"] if "id" in n}
    if set(tw) != rels:
        raise ValueError(f"relation keys don't match the graph: {set(tw) ^ rels}")
    if set(ni) != types:
        raise ValueError(f"node-type keys don't match the graph: {set(ni) ^ types}")
    if any(v is None for v in tw.values()) or any(v is None for v in ni.values()):
        raise ValueError("table is unfilled (null entries) — nothing runs on a skeleton")
    if any(not float(v) > 0.0 for v in ni.values()):
        raise ValueError("node inertias must be > 0 (coercivity)")
    w = np.zeros(base.m)
    for e in data["edges"]:
        if "src" not in e:
            continue
        w[base.edge_of(e["src"], e["dst"])] += float(tw[e["relation"]])
    dead = np.where(w == 0.0)[0]
    if dead.size:
        names = ", ".join(base.edge_label(k) for k in dead)
        raise ValueError(f"composed charge has silent (exactly zero) edges: {names}")
    ntype = {n["id"]: n["type"] for n in data["nodes"] if "id" in n}
    inertia = _read_only(np.array([float(ni[ntype[nid]]) for nid in base.ids]))
    soul = GraphAlgebra(
        ids=base.ids, edges=base.edges, w0=_read_only(w), n=base.n, m=base.m,
        rank_j0=0, index=0, perfect_matching=False,
        _node_index=base._node_index, _pair_index=base._pair_index, inertia=inertia,
    )
    rank = int(np.linalg.matrix_rank(soul.J(w)))
    object.__setattr__(soul, "rank_j0", rank)
    object.__setattr__(soul, "index", soul.n + soul.m - rank)
    object.__setattr__(soul, "perfect_matching", rank == soul.n)
    return soul


def identity_distance(alg_a: GraphAlgebra, alg_b: GraphAlgebra) -> dict:
    """The graded identity distance (§9.18 recorded facts): both charges embedded in the
    ambient index-pair space (arenas identified by sorted-id index), distance = the norm of
    the difference; support overlap recorded. A vector distance between souls, not a binary
    verdict — what the D4 authoring purchased."""
    va = {tuple(p): float(alg_a.w0[k]) for k, p in enumerate(alg_a.edges.tolist())}
    vb = {tuple(p): float(alg_b.w0[k]) for k, p in enumerate(alg_b.edges.tolist())}
    support = set(va) | set(vb)
    d2 = sum((va.get(p, 0.0) - vb.get(p, 0.0)) ** 2 for p in support)
    return {
        "distance": float(np.sqrt(d2)),
        "overlap": len(set(va) & set(vb)),
        "support_union": len(support),
        "norm_a": float(np.linalg.norm(alg_a.w0)),
        "norm_b": float(np.linalg.norm(alg_b.w0)),
    }


# --------------------------------------------------------------------------- #
# H₀ and the charge readers.
# --------------------------------------------------------------------------- #
def h0(p: Array) -> Array:
    """Placeholder law: ``H₀ = ½|p|²`` (inertia ≡ 1 — content not yet attached; the D5 call is
    the content increment's). Coercive, and NOT a function of ``w`` — so the free flow moves:
    §9.16's anisotropy precondition has no analog here because ``H₀`` is not a Casimir of this
    bracket. Batched over (..., n)."""
    p = np.asarray(p, float)
    return 0.5 * np.sum(p * p, axis=-1)


def _h0(alg: GraphAlgebra, p: Array) -> Array:
    """The algebra's own law: the placeholder ``½|p|²`` when no inertia is authored (the
    §9.17 recorded path, byte-for-byte), else the §9.18 authored ``½ Σ p_v²/I_v``."""
    if alg.inertia is None:
        return h0(p)
    p = np.asarray(p, float)
    return 0.5 * np.sum(p * p / alg.inertia, axis=-1)


def _genesis(alg: GraphAlgebra, u: Array, e0: float) -> Array:
    """A genesis point in the direction of ``u`` with ``H₀(p₀) = e0`` under the algebra's own
    law. The placeholder branch keeps §9.17's exact expression (bit-identical records)."""
    if alg.inertia is None:
        return np.sqrt(2.0 * e0) * u / np.linalg.norm(u)
    unit = u / np.linalg.norm(u)
    return unit * np.sqrt(e0 / _h0(alg, unit))


def _m0(alg: GraphAlgebra) -> Array:
    return np.eye(alg.n) if alg.inertia is None else np.diag(1.0 / alg.inertia)


def psi_deviation(w: Array, w_ref: Array) -> float:
    """The ψ-reader's distance: ``max_e |w_e − w_ref,e|`` (the §9.17 D7 per-coordinate form).
    A survivor's deviation is identically 0.0 — bit-exact, not small."""
    return float(np.max(np.abs(np.asarray(w, float) - np.asarray(w_ref, float))))


# --------------------------------------------------------------------------- #
# The input alphabet Σ: symbols ARE (quadratic+linear) Hamiltonians — declarative data.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Symbol:
    """A legal input event: ``H_σ(p) = a·p + ½ pᵀAp``. Storing ``(a, A)`` rather than
    callables makes membership in the Hamiltonian class TYPE-LEVEL at this increment (nothing
    non-gradient can be smuggled — the ALPHABET-AUTHORING item-6 concern is structural here),
    and it is exactly what the exact affine stepper consumes. ``h`` is always defined (bar-1
    bookkeeping needs it). ψ-breaking input cannot be spelled as a Symbol — see ``weaken``."""

    name: str
    a: Array | None = None
    A: Array | None = None
    duration: float | None = None

    def h(self, p: Array) -> Array:
        p = np.asarray(p, float)
        out = np.zeros(p.shape[:-1])
        if self.a is not None:
            out = out + np.sum(p * self.a, axis=-1)
        if self.A is not None:
            out = out + 0.5 * np.sum(p * (p @ self.A), axis=-1)
        return out

    def grad_h(self, p: Array) -> Array:
        p = np.asarray(p, float)
        out = np.zeros_like(p)
        if self.a is not None:
            out = out + self.a
        if self.A is not None:
            out = out + p @ self.A
        return out


def kick(a: Array, name: str = "kick") -> Symbol:
    """``H = a·p`` — a drift whose direction the identity's own edges determine
    (``ṗ = J(w)a``): the same event literally moves differently through a different soul."""
    return Symbol(name, a=_read_only(np.array(a, dtype=float)))


def quad(A: Array, name: str = "quad") -> Symbol:
    """``H = ½ pᵀAp`` — flow ``ṗ = J(w)Ap``. Only sym(A) enters H, so the constructor
    symmetrizes (keeps ∇H consistent for any authored input). The so(3)* silence trap MOVED:
    ``quad(c·I)`` (``H ∝ |p|²``) is not silent here — silent means ``H = f(w)`` alone."""
    A = np.asarray(A, float)
    return Symbol(name, A=_read_only(0.5 * (A + A.T)))


def node_kick(alg: GraphAlgebra, node_id: str, amp: float, name: str | None = None) -> Symbol:
    """``kick(amp·e_v)`` — an event that pushes on THAT concept (the graph-adapted base
    element; canonical rather than decorative — ALPHABET-AUTHORING §8)."""
    a = np.zeros(alg.n)
    a[alg.node_index(node_id)] = amp
    return kick(a, name or node_id)


def edge_quad(alg: GraphAlgebra, u_id: str, v_id: str, amp: float,
              name: str | None = None) -> Symbol:
    """``H = amp·p_u p_v`` — the cross-coupling quad on one authored edge (the other half of
    the graph-adapted base set: one per relation). Both off-diagonal entries are set so the
    symmetric matrix yields exactly ``amp·p_u p_v`` (eigenvalues ±amp — the §9.17 pinned
    window-coercivity values; the certificate caught the halved-amplitude variant on the
    first execution)."""
    A = np.zeros((alg.n, alg.n))
    i, j = alg.node_index(u_id), alg.node_index(v_id)
    A[i, j] = amp
    A[j, i] = amp
    return quad(A, name or f"{u_id}~{v_id}")


def make_default_alphabet(alg: GraphAlgebra, eps_kick: float = EPS_KICK,
                          eps_quad: float = EPS_QUAD) -> dict[str, Symbol]:
    """|Σ| = 4, §9.17-pinned: kicks on the adjacent pair (x, y) and on a node adjacent to
    neither (z), plus the edge-quad on the x–y edge (q). Which of these commute is the
    authored topology — measured by the bar-2 certificates."""
    return {
        "x": node_kick(alg, PINNED_X, eps_kick, "x"),
        "y": node_kick(alg, PINNED_Y, eps_kick, "y"),
        "z": node_kick(alg, PINNED_Z, eps_kick, "z"),
        "q": edge_quad(alg, PINNED_X, PINNED_Y, eps_quad, "q"),
    }


def weaken(w: Array, edge_k: int, gamma: float, dt: float, n_steps: int) -> Array:
    """``†`` — the OUT-OF-ALPHABET control: graph surgery. Contracts ONE edge momentum,
    ``w_e ← (1 − γ·dt)^n · w_e``, and returns a NEW charge vector (the input stays untouched —
    it is write-locked anyway). Not a Symbol, and provably cannot be one: every Hamiltonian
    flow on 𝔤(G)* fixes every ``w_e``. The boundary is per-edge legible: the ψ change NAMES
    the relation touched (weaken / sever / form are the epoch-boundary operations made
    concrete). The arena holds during the surgery window — mirroring §9.16's dissipative map."""
    out = np.array(w, dtype=float, copy=True)
    out[edge_k] *= (1.0 - gamma * dt) ** n_steps
    return out


# --------------------------------------------------------------------------- #
# The integrator: one EXACT linear-affine map per step ⇒ ψ is a state partition.
# --------------------------------------------------------------------------- #
def step_map(alg: GraphAlgebra, w: Array, dt: float, sym: Symbol | None = None,
             scale: float = 1.0, with_h0: bool = True) -> tuple[Array, Array]:
    """The exact time-``dt`` flow map of ``H = [H₀] + scale·H_σ`` on the leaf {w}:
    ``p ← Φp + b`` with ``[[Φ, b], [0, 1]] = exp(dt·[[J(w)M, J(w)a], [0, 0]])`` (M = I for
    the placeholder law, plus the symbol's quadratic part). Computed once per (symbol, scale);
    ``w`` enters only through ``J`` and is never written."""
    J = alg.J(w)
    M = _m0(alg) if with_h0 else np.zeros((alg.n, alg.n))
    a = np.zeros(alg.n)
    if sym is not None:
        if sym.A is not None:
            M = M + scale * sym.A
        if sym.a is not None:
            a = a + scale * sym.a
    T = np.zeros((alg.n + 1, alg.n + 1))
    T[: alg.n, : alg.n] = J @ M
    T[: alg.n, alg.n] = J @ a
    E = expm(dt * T)
    return E[: alg.n, : alg.n], E[: alg.n, alg.n]


def window_min_eig(alg: GraphAlgebra, sym: Symbol) -> float:
    """The §9.17 per-window coercivity certificate: min eigenvalue of ``M₀ + A_σ``. Positive ⇒
    the window Hamiltonian is coercive ⇒ the event flow is bounded (compactness was lost with
    the sphere; this is its replacement, checked instead of assumed)."""
    M = _m0(alg)
    if sym.A is not None:
        M = M + sym.A
    return float(np.min(np.linalg.eigvalsh(M)))


# --------------------------------------------------------------------------- #
# Words: gap – event – gap – … (events ADDITIVE: the automaton keeps being itself).
# --------------------------------------------------------------------------- #
def run_word(word: Sequence[Symbol], p0: Array, alg: GraphAlgebra, w: Array | None = None, *,
             dt: float = 0.01, tau_event: float = 0.5, tau_gap: float = 0.5,
             envelope: str | None = None) -> Array:
    """Integrate genesis ``p0`` through a word: gap, σ₁, gap, σ₂, …, gap. Events are additive
    (``H₀ + H_σ``); silence is the free flow. ``envelope="sin2"`` scales H_σ by
    sin²(π·t/τ) within each event (piecewise-constant per dt step — §9.16's treatment of
    smooth ∂H/∂t ≠ 0). Batched over leading dims of ``p0``; honors per-symbol ``duration``.
    Returns (n_steps+1, ..., n); ``w`` (default: the algebra's ``w₀``) is read, never written."""
    if w is None:
        w = alg.w0
    p = np.asarray(p0, float)
    out = [p]
    n_gap = int(round(tau_gap / dt))
    phi_gap, _ = step_map(alg, w, dt)

    def free_steps(n: int) -> None:
        nonlocal p
        for _ in range(n):
            p = p @ phi_gap.T
            out.append(p)

    free_steps(n_gap)
    for sym in word:
        n_ev = int(round((sym.duration if sym.duration is not None else tau_event) / dt))
        if envelope == "sin2":
            for k in range(n_ev):
                s = float(np.sin(np.pi * (k + 0.5) / n_ev) ** 2)
                phi, b = step_map(alg, w, dt, sym, scale=s)
                p = p @ phi.T + b
                out.append(p)
        else:
            phi, b = step_map(alg, w, dt, sym)
            for _ in range(n_ev):
                p = p @ phi.T + b
                out.append(p)
        free_steps(n_gap)
    return np.stack(out)


def run_bare_events(word: Sequence[Symbol], p0: Array, alg: GraphAlgebra,
                    w: Array | None = None, *, dt: float = 0.01,
                    tau_event: float = 0.5) -> Array:
    """Consecutive BARE events (H_σ alone — no H₀, no gaps): the bar-2 certificate protocol.
    For kicks the exact map is a pure translation (Φ = I, b = dt·J(w)a — the augmented matrix
    is nilpotent, so even the exponential is exact)."""
    if w is None:
        w = alg.w0
    p = np.asarray(p0, float)
    out = [p]
    for sym in word:
        n_ev = int(round((sym.duration if sym.duration is not None else tau_event) / dt))
        phi, b = step_map(alg, w, dt, sym, with_h0=False)
        for _ in range(n_ev):
            p = p @ phi.T + b
            out.append(p)
    return np.stack(out)


# --------------------------------------------------------------------------- #
# ζ ∈ ℝ^E: per-edge signed areas about the genesis gauge (memory, shaped like identity).
# --------------------------------------------------------------------------- #
def zeta_edges(ps: Array, edges: Array, p0: Array | None = None, *,
               cumulative: bool = False) -> Array:
    """``ζ_e = ½ ∮ (x_u dx_v − x_v dx_u)`` with ``x = p − p₀`` — the signed area the arena
    trajectory sweeps in each authored edge plane, accumulated by the exact polyline
    (shoelace) rule. Gauge: ``p₀`` defaults to the genesis point (identity data, per-
    trajectory); every pre-registered comparison shares a genesis. Closed loops are
    gauge-independent EXACTLY (flat planes — no mod-4π analog; tested). ``ps`` is
    (n_steps+1, ..., n); returns (..., m), or (n_steps, ..., m) cumulative."""
    ps = np.asarray(ps, float)
    if p0 is None:
        p0 = ps[0]
    x = ps - np.asarray(p0, float)
    xu, xv = x[..., edges[:, 0]], x[..., edges[:, 1]]
    d = 0.5 * (xu[:-1] * xv[1:] - xv[:-1] * xu[1:])
    return d.cumsum(axis=0) if cumulative else d.sum(axis=0)


# --------------------------------------------------------------------------- #
# The §9.17 harnesses — one shared driven ensemble serves bars 1, 3, and 4.
# --------------------------------------------------------------------------- #
def driven_ensemble(alg: GraphAlgebra | None = None, *, n_seeds: int = 8,
                    words_per_seed: int = 25, word_len: int = 16,
                    alphabet: dict[str, Symbol] | None = None, dt: float = 0.01,
                    tau_event: float = 0.5, tau_gap: float = 0.5, e0: float = 1.0,
                    checkpoints: tuple[int, ...] = (4, 8, 16)) -> dict:
    """The shared driven population (§9.17 pinned constants): per seed, ``words_per_seed``
    random words over the alphabet, each from a fresh genesis with ``H₀(p₀) = e0``. Streams
    the accumulators the bars need (per-step H₀ series, ζ with checkpoint snapshots, event-
    window boundary states, ``max|p|``, endpoints) instead of materializing 200 trajectories
    in ℝ¹⁰⁰; the free-evolution twin (same geneses, no events, same total duration) is bar
    1's baseline. The threaded ``w`` is returned for the bit-level bar-1 equality."""
    if alg is None:
        alg = load_graph()
    if alphabet is None:
        alphabet = make_default_alphabet(alg)
    symbols = list(alphabet.values())
    dirs, words = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((words_per_seed, alg.n))
        dirs.append(u / np.linalg.norm(u, axis=1, keepdims=True))
        words.append(rng.integers(0, len(symbols), (words_per_seed, word_len)))
    u_all = np.concatenate(dirs)
    if alg.inertia is None:
        p0 = np.sqrt(2.0 * e0) * u_all                     # (n_words, n): H₀(p₀) = e0
    else:
        p0 = u_all * np.sqrt(e0 / _h0(alg, u_all))[..., None]
    word_indices = np.concatenate(words)                   # (n_words, word_len)
    n_words = p0.shape[0]
    n_gap = int(round(tau_gap / dt))
    n_ev = int(round(tau_event / dt))
    period = n_ev + n_gap
    w = alg.w0                                             # read-only: the partition is mechanical
    phi_gap, _ = step_map(alg, w, dt)
    maps = [step_map(alg, w, dt, s) for s in symbols]

    p = p0.copy()
    x_edges = (alg.edges[:, 0], alg.edges[:, 1])
    zeta = np.zeros((n_words, alg.m))
    zeta_at: dict[int, Array] = {}
    h0_series = [_h0(alg, p)]
    win_start = np.empty((word_len, n_words, alg.n))
    win_end = np.empty((word_len, n_words, alg.n))
    max_abs_p = float(np.max(np.abs(p)))
    step_count = 0
    snap_at = {n_gap + c * period - 1: c for c in checkpoints}

    def advance(p_new: Array) -> Array:
        nonlocal zeta, max_abs_p, step_count
        xu0, xv0 = (p - p0)[:, x_edges[0]], (p - p0)[:, x_edges[1]]
        xu1, xv1 = (p_new - p0)[:, x_edges[0]], (p_new - p0)[:, x_edges[1]]
        zeta = zeta + 0.5 * (xu0 * xv1 - xv0 * xu1)
        h0_series.append(_h0(alg, p_new))
        max_abs_p = max(max_abs_p, float(np.max(np.abs(p_new))))
        if step_count in snap_at:
            zeta_at[snap_at[step_count]] = zeta.copy()
        step_count += 1
        return p_new

    def gap_steps(n: int) -> None:
        nonlocal p
        for _ in range(n):
            p = advance(p @ phi_gap.T)

    gap_steps(n_gap)
    for slot in range(word_len):
        sel = word_indices[:, slot]
        win_start[slot] = p
        for _ in range(n_ev):
            p_new = np.empty_like(p)
            for k in range(len(symbols)):
                mask = sel == k
                if mask.any():
                    phi, b = maps[k]
                    p_new[mask] = p[mask] @ phi.T + b
            p = advance(p_new)
        win_end[slot] = p
        gap_steps(n_gap)

    # The free-evolution twin: same geneses, no events, same total duration (streamed).
    pf = p0.copy()
    ef = _h0(alg, pf)
    f_min, f_max, f_0 = ef.copy(), ef.copy(), ef.copy()
    for _ in range(step_count):
        pf = pf @ phi_gap.T
        ef = _h0(alg, pf)
        f_min = np.minimum(f_min, ef)
        f_max = np.maximum(f_max, ef)

    return {
        "alg": alg, "w0": alg.w0, "w_final": w, "symbols": symbols,
        "word_indices": word_indices, "p0": p0, "p_final": p,
        "h0_series": np.stack(h0_series), "free_h0_min": f_min, "free_h0_max": f_max,
        "free_h0_0": f_0, "zeta_final": zeta, "zeta_at": zeta_at,
        "win_start": win_start, "win_end": win_end, "max_abs_p": max_abs_p,
        "e0": e0, "dt": dt, "n_gap": n_gap, "n_ev": n_ev, "word_len": word_len,
    }


def casimir_under_words(ens: dict, *, sin2_word: str = SIN2_WORD) -> dict:
    """Bar 1: ψ under words is a bit-level EQUALITY (state partition certified in code) while
    the along-trajectory H₀ range is macroscopic and ≫ the free baseline; per-window
    bookkeeping attributes the energy change to driving physics; the sin²-envelope sub-run
    covers smooth ∂H/∂t ≠ 0 (its ``w`` is threaded through a writable copy and compared)."""
    alg = ens["alg"]
    e = ens["h0_series"]
    driven_range = (e.max(axis=0) - e.min(axis=0)) / e[0]
    free_range = (ens["free_h0_max"] - ens["free_h0_min"]) / ens["free_h0_0"]
    d_h0, d_total = [], []
    for slot in range(ens["word_len"]):
        sel = ens["word_indices"][:, slot]
        ps, pe = ens["win_start"][slot], ens["win_end"][slot]
        h0_s, h0_e = _h0(alg, ps), _h0(alg, pe)
        hs, he = np.zeros_like(h0_s), np.zeros_like(h0_e)
        for k, sym in enumerate(ens["symbols"]):
            mask = sel == k
            if mask.any():
                hs[mask] = sym.h(ps[mask])
                he[mask] = sym.h(pe[mask])
        d_h0.append(np.abs(h0_e - h0_s))
        d_total.append(np.abs((h0_e + he) - (h0_s + hs)))
    d_h0, d_total = np.concatenate(d_h0), np.concatenate(d_total)
    # sin² sub-run: one fixed word, smooth envelope, through a WRITABLE w copy — the equality
    # check would catch a write path the read-only flag cannot see.
    alphabet = {s.name: s for s in ens["symbols"]}
    rng = np.random.default_rng(0)
    u = rng.standard_normal(alg.n)
    p0 = _genesis(alg, u, ens["e0"])
    w_smooth = np.array(alg.w0, copy=True)
    smooth = run_word([alphabet[c] for c in sin2_word], p0, alg, w_smooth,
                      dt=ens["dt"], envelope="sin2")
    e_smooth = _h0(alg, smooth)
    return {
        "psi_bit_exact": bool(np.array_equal(ens["w_final"], ens["w0"])),
        "psi_max_delta": psi_deviation(ens["w_final"], ens["w0"]),
        "sin2_bit_exact": bool(np.array_equal(w_smooth, alg.w0)),
        "sin2_max_delta": psi_deviation(w_smooth, alg.w0),
        "median_driven_h0_range": float(np.median(driven_range)),
        "median_free_h0_range": float(np.median(free_range)),
        "range_ratio": float(np.median(driven_range) / np.median(free_range)),
        "bookkeeping_max_dtotal": float(np.max(d_total)),
        "median_event_dh0": float(np.median(d_h0)),
        "sin2_h0_range": float((e_smooth.max() - e_smooth.min()) / e_smooth[0]),
    }


def bracket_certificate(alg: GraphAlgebra, w: Array | None = None, *, seed: int = 0,
                        e0: float = 1.0, eps: float = EPS_KICK, tau: float = 0.5,
                        dt: float = 0.01) -> dict:
    """Bar 2(a) — the bracket, measured: under one BARE kick(a) event, any linear observable
    moves at exactly the bracket rate, ``Δ(bᵀp) = τ·bᵀJ(w)a``. For the pinned adjacent pair
    that is ``±τ·ε·w_e`` (nonzero — it NAMES the edge); for the pinned non-adjacent pair it
    is 0: which experiences commute is the authored topology."""
    if w is None:
        w = alg.w0
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(alg.n)
    p0 = _genesis(alg, u, e0)
    sym_x = node_kick(alg, PINNED_X, eps, "x")
    ps = run_bare_events([sym_x], p0, alg, w, dt=dt, tau_event=tau)
    delta = ps[-1] - ps[0]
    predicted = tau * (alg.J(w) @ (eps * _basis(alg, PINNED_X)))
    i_y, i_z = alg.node_index(PINNED_Y), alg.node_index(PINNED_Z)
    return {
        "adjacent_measured": float(delta[i_y]),
        "adjacent_predicted": float(predicted[i_y]),
        "adjacent_abs": float(abs(delta[i_y])),
        "adjacent_closed_form": tau * eps * float(alg.w0[alg.edge_of(PINNED_X, PINNED_Y)]),
        "nonadjacent_measured": float(delta[i_z]),
        "max_pred_error": float(np.max(np.abs(delta - predicted))),
    }


def heisenberg_certificate(alg: GraphAlgebra, w: Array | None = None, *, seed: int = 0,
                           e0: float = 1.0, eps: float = EPS_KICK, tau: float = 0.5,
                           dt: float = 0.01) -> dict:
    """Bar 2(b) — the Heisenberg signature: BARE kicks are translations, so "xy" and "yx" end
    at the SAME state (Δp at float rounding — the state forgets bare order) while ζ differs by
    the exact parallelogram areas ``Δζ_e = A_u B_v − A_v B_u`` (A = τJ(w)a, B = τJ(w)b) — the
    memory records the order, alone. On the shared edge that is ``(τεw_e)²``. Recorded
    contrast with so(3)*: there, order reached the state directly."""
    if w is None:
        w = alg.w0
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(alg.n)
    p0 = _genesis(alg, u, e0)
    sx = node_kick(alg, PINNED_X, eps, "x")
    sy = node_kick(alg, PINNED_Y, eps, "y")
    t_xy = run_bare_events([sx, sy], p0, alg, w, dt=dt, tau_event=tau)
    t_yx = run_bare_events([sy, sx], p0, alg, w, dt=dt, tau_event=tau)
    dz = zeta_edges(t_xy, alg.edges) - zeta_edges(t_yx, alg.edges)
    J = alg.J(w)
    A = tau * (J @ (eps * _basis(alg, PINNED_X)))
    B = tau * (J @ (eps * _basis(alg, PINNED_Y)))
    predicted = A[alg.edges[:, 0]] * B[alg.edges[:, 1]] - A[alg.edges[:, 1]] * B[alg.edges[:, 0]]
    k = alg.edge_of(PINNED_X, PINNED_Y)
    return {
        "delta_p": float(np.linalg.norm(t_xy[-1] - t_yx[-1])),
        "max_pred_error": float(np.max(np.abs(dz - predicted))),
        "shared_edge_measured": float(dz[k]),
        "shared_edge_closed_form": (tau * eps * float(alg.w0[k])) ** 2,
    }


def word_order_test(alg: GraphAlgebra, *, seed: int = 0, dt: float = 0.01,
                    e0: float = 1.0) -> dict:
    """Bar 2(c): full "xy" vs "yx" events (the law running, gaps and all) from the same
    genesis — order reaches the state exactly THROUGH the law, and ζ records it, while ψ is
    bit-exact both ways (writable copies threaded to certify it)."""
    alphabet = make_default_alphabet(alg)
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(alg.n)
    p0 = _genesis(alg, u, e0)
    w_xy, w_yx = np.array(alg.w0, copy=True), np.array(alg.w0, copy=True)
    t_xy = run_word([alphabet["x"], alphabet["y"]], p0, alg, w_xy, dt=dt)
    t_yx = run_word([alphabet["y"], alphabet["x"]], p0, alg, w_yx, dt=dt)
    return {
        "delta_p": float(np.linalg.norm(t_xy[-1] - t_yx[-1])),
        "delta_zeta": float(np.linalg.norm(zeta_edges(t_xy, alg.edges)
                                           - zeta_edges(t_yx, alg.edges))),
        "psi_bit_exact_xy": bool(np.array_equal(w_xy, alg.w0)),
        "psi_bit_exact_yx": bool(np.array_equal(w_yx, alg.w0)),
    }


def replica_under_driving(ens: dict, *, scale: float = 1.5, shuffle_seed: int = 0,
                          extra_w: dict[str, Array] | None = None) -> dict:
    """Bar 3 — the §2 replica test with Σ active, maximally charitable: the replica copies the
    survivor's ENTIRE arena ``p`` bit-exactly (every observable of the arena granted free) and
    is born on the wrong ``w``. Primary: ``scale·w₀`` — every coordinate wrong (the §9.16
    wrong-sphere analog). Secondary: a seeded coordinate shuffle — the value MULTISET granted
    free, caught on arrangement alone (which relation carries which weight); with the
    near-uniform placeholder counts its touched set is small (recorded — the row strengthens
    with authored content). The survivor's deviation is identically 0.0, so the §9.16 margin
    ratio is degenerate: the pre-registered form is the pair of absolutes."""
    w0 = ens["w0"]
    p_f = ens["p_final"]
    p_rep = p_f.copy()                                     # π = id on the arena, bit-exact
    n_words = p_f.shape[0]
    surv_dev = psi_deviation(ens["w_final"], w0)
    w_scaled = scale * np.asarray(w0)
    perm = np.random.default_rng(shuffle_seed).permutation(len(w0))
    w_shuffled = np.asarray(w0)[perm]
    touched = w_shuffled != np.asarray(w0)
    surv_scores = [-surv_dev] * n_words
    scaled_scores = [-psi_deviation(w_scaled, w0)] * n_words
    shuffled_scores = [-psi_deviation(w_shuffled, w0)] * n_words
    out = {
        "auc_psi_scaled": auc(surv_scores, scaled_scores),
        "auc_psi_shuffled": auc(surv_scores, shuffled_scores),
        "auc_endpoint": auc(list(p_f[:, 0]), list(p_rep[:, 0])),
        "endpoint_erasure": float(np.max(np.abs(p_rep - p_f))),
        "surv_max_dev": surv_dev,
        "scaled_min_coord_dev": float(np.min(np.abs(w_scaled - w0))),
        "shuffled_min_touched_dev": float(np.min(np.abs(w_shuffled - w0)[touched])),
        "n_touched_shuffled": int(np.sum(touched)),
    }
    # §9.18 extra impostor rows (e.g. the counts-impostor — the §9.17 placeholder itself:
    # knows the topology, not the authored content). Absent by default; §9.17 output unchanged.
    for name, w_alt in (extra_w or {}).items():
        dev = np.abs(np.asarray(w_alt, float) - np.asarray(w0))
        hit = dev > 0
        out[f"auc_psi_{name}"] = auc(surv_scores, [-float(dev.max())] * n_words)
        out[f"{name}_norm_dev"] = float(np.linalg.norm(dev))
        out[f"{name}_max_dev"] = float(dev.max())
        out[f"{name}_min_nonzero_dev"] = float(dev[hit].min())
        out[f"{name}_n_matched"] = int(np.sum(~hit))
        out[f"{name}_matched_edges"] = [ens["alg"].edge_label(k) for k in np.where(~hit)[0]]
    return out


def zeta_memory_test(ens: dict) -> dict:
    """Bar 4: ζ under driving. Lived worldlines carry macroscopic ζ ∈ ℝ^E (newborn copies
    carry 0); median ‖ζ‖ grows with events lived; the SCALE certificate (flat planes have no
    branch — the certified risk is growth) reports ``max|p|`` against the pinned guard and the
    per-window coercivity minimum."""
    lived = np.linalg.norm(ens["zeta_final"], axis=1)
    acc = {int(c): float(np.median(np.linalg.norm(z, axis=1)))
           for c, z in sorted(ens["zeta_at"].items())}
    min_eig = min(window_min_eig(ens["alg"], s) for s in ens["symbols"])
    return {
        "auc_zeta": auc(list(lived), list(np.zeros_like(lived))),
        "min_lived": float(lived.min()),
        "median_lived": float(np.median(lived)),
        "accumulation": acc,
        "max_abs_p": ens["max_abs_p"],
        "min_window_eig": min_eig,
    }


def dissipation_control(alg: GraphAlgebra, *, seed: int = 0, word_len: int = 16,
                        gamma: float = 0.5, dt: float = 0.01, tau_event: float = 0.5,
                        tau_gap: float = 0.5, e0: float = 1.0) -> dict:
    """Bar 5 — † as graph surgery, paired: the SAME word from the SAME genesis, with and
    without one ``weaken`` event at the word's midpoint on the pinned edge. Without: ψ is
    bit-exact. With: exactly ONE coordinate moves, by the discrete map's own closed form
    ``1 − (1 − γ·dt)^n`` — and the ψ change names the relation touched."""
    alphabet = make_default_alphabet(alg)
    symbols = list(alphabet.values())
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(alg.n)
    p0 = _genesis(alg, u, e0)
    word = [symbols[k] for k in rng.integers(0, len(symbols), word_len)]
    k_edge = alg.edge_of(PINNED_X, PINNED_Y)
    n_ev = int(round(tau_event / dt))

    w_clean = np.array(alg.w0, copy=True)
    run_word(word, p0, alg, w_clean, dt=dt, tau_event=tau_event, tau_gap=tau_gap)

    half = word_len // 2
    w_live = np.array(alg.w0, copy=True)
    first = run_word(word[:half], p0, alg, w_live, dt=dt, tau_event=tau_event, tau_gap=tau_gap)
    w_cut = weaken(w_live, k_edge, gamma, dt, n_ev)        # the arena holds; the charge is cut
    run_word(word[half:], first[-1], alg, w_cut, dt=dt, tau_event=tau_event, tau_gap=tau_gap)

    others = np.delete(np.arange(alg.m), k_edge)
    return {
        "psi_bit_exact_without": bool(np.array_equal(w_clean, alg.w0)),
        "psi_max_delta_without": psi_deviation(w_clean, alg.w0),
        "touched_rel_change": float((alg.w0[k_edge] - w_cut[k_edge]) / alg.w0[k_edge]),
        "closed_form": 1.0 - (1.0 - gamma * dt) ** n_ev,
        "others_bit_exact": bool(np.array_equal(w_cut[others], np.asarray(alg.w0)[others])),
        "touched_edge": alg.edge_label(k_edge),
    }


def liveness_test(alg_a: GraphAlgebra, alg_b: GraphAlgebra, *, word: str = SIN2_WORD,
                  seed: int = 0, dt: float = 0.01, e0: float = 1.0,
                  eps_quad: float = EPS_QUAD) -> dict:
    """Bar 6 — the charge is dynamically load-bearing: the SAME genesis living the SAME word
    under two souls' brackets (arenas identified by sorted-id index — arbitrary, pinned;
    identity enters only through J(w)) diverges macroscopically, while each soul's ψ is
    bit-exact under its own run. Two souls living the same events live different lives."""
    if alg_a.n != alg_b.n:
        raise ValueError("index identification needs equal node counts")
    alphabet = make_default_alphabet(alg_a, eps_quad=eps_quad)
    syms = [alphabet[c] for c in word]
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(alg_a.n)
    p0 = _genesis(alg_a, u, e0)
    w_a, w_b = np.array(alg_a.w0, copy=True), np.array(alg_b.w0, copy=True)
    t_a = run_word(syms, p0, alg_a, w_a, dt=dt)
    t_b = run_word(syms, p0, alg_b, w_b, dt=dt)
    return {
        "max_traj_divergence": float(np.max(np.linalg.norm(t_a - t_b, axis=-1))),
        "psi_bit_exact_a": bool(np.array_equal(w_a, alg_a.w0)),
        "psi_bit_exact_b": bool(np.array_equal(w_b, alg_b.w0)),
        "m_a": alg_a.m, "index_a": alg_a.index,
        "m_b": alg_b.m, "index_b": alg_b.index,
    }


def training_cannot_write_w(alg: GraphAlgebra, *, n_updates: int = 20, n_basis: int = 8,
                            lr: float = 0.1, dt: float = 0.01, n_steps: int = 50,
                            seed: int = 0) -> dict:
    """Bar 6 (§9.18) — the §3.7 training guarantee, first measured instance: gradient-shaped
    descent on a parameterized quadratic ``H_θ`` (finite differences on an endpoint loss,
    each update interleaved with rollouts on the authored geometry) cannot write ``w`` —
    training moves θ, the flow moves ``p``, and the charge is not an operand of either.
    Quadratic scope at this increment; the MLP-scope re-run rides with the π-preparation
    increment (stated in §9.18, not silently dropped)."""
    rng = np.random.default_rng(seed)
    basis = []
    for _ in range(n_basis):
        b = rng.standard_normal((alg.n, alg.n))
        basis.append(0.05 * (b + b.T))
    p0 = _genesis(alg, rng.standard_normal(alg.n), 1.0)
    target = _genesis(alg, rng.standard_normal(alg.n), 1.0)
    w_before = np.array(alg.w0, copy=True)

    def rollout_loss(th: Array) -> float:
        A = sum(t * B for t, B in zip(th, basis, strict=True))
        phi, b = step_map(alg, alg.w0, dt, quad(A, "H_theta"))
        p = p0
        for _ in range(n_steps):
            p = p @ phi.T + b
        return float(np.sum((p - target) ** 2))

    theta = np.zeros(n_basis)
    loss_first = rollout_loss(theta)
    fd = 1e-4
    for _ in range(n_updates):
        grad = np.zeros(n_basis)
        for i in range(n_basis):
            up, dn = theta.copy(), theta.copy()
            up[i] += fd
            dn[i] -= fd
            grad[i] = (rollout_loss(up) - rollout_loss(dn)) / (2.0 * fd)
        theta = theta - lr * grad
    return {
        "w_bit_exact": bool(np.array_equal(alg.w0, w_before)),
        "loss_first": loss_first,
        "loss_last": rollout_loss(theta),
        "n_updates": n_updates,
    }


def _basis(alg: GraphAlgebra, node_id: str) -> Array:
    e = np.zeros(alg.n)
    e[alg.node_index(node_id)] = 1.0
    return e
