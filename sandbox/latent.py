"""latent.py — phase two: the d-dim latent core (§9 of `../docs/CORE-SPEC.md`).

Generalizes the phase-one 1-DOF toy to a `d`-dimensional latent phase space, with the potential
**shaped by Embra's identity graph** rather than hand-set.

    state        s = (q, p)                  ∈ ℝ^d × ℝ^d
    Hamiltonian  H(q, p) = ½|p|²/m + V(q)     (separable → the phase-one integrator still works)
    potential    V(q) = ½ (q−c)ᵀ P (q−c)      P = (Σ_anchors + εI)⁻¹  — Mahalanobis to the
                                              identity-anchor cloud (the closed-form default;
                                              any charge with the same API plugs in via fit_fn)
    conserved    Q(s) = H(q, p)               identity manifold M = a level set of V
    observable   π(s) = q                      (position; momentum still hidden)

What lives here, in the order the increments earned it:
  - graph → anchors (`load_identity_anchors`, the shuffled control, `_diffusion_embed` kept for
    §9.10 reproducibility) and the `GaussianManifold` charge;
  - the symplectic flow (`leapfrog_step`/`rollout`) and the replica test lifted to d dims
    (`make_pairs`/`evaluate`) — §9.8;
  - static specificity (`specificity`, ≈ chance by honest record — §9.9–§9.10);
  - **dynamical** specificity (`dynamical_specificity`: which conservation law a trajectory
    obeys) — §9.11–§9.13;
  - the **conjunction test** (`conjunction_test`: obeys the law ∧ born on the right level set,
    graded against both adversarial impostor classes) — §9.14;
  - **holonomy ζ** (`holonomy_zeta`/`holonomy_test`: the path-functional memory charge) — §9.15.

The learned charge (jax MLP) is `hnn.py`; this module stays the dependency-light backbone and
the default test target.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sandbox.replica_test import auc  # reuse the phase-one AUC

Array = NDArray[np.float64]

IDENTITY_GRAPH = Path(__file__).resolve().parents[1] / "identity" / "Embra_IDENTITY-SOUL.graph.json"


# --------------------------------------------------------------------------- #
# Identity graph → anchor configurations in ℝ^d (spectral / Laplacian embedding)
# --------------------------------------------------------------------------- #
def _adjacency(graph_path: Path) -> tuple[list[str], Array]:
    """Parse the graph JSON into (node ids, symmetric weighted adjacency). Pure
    ``{"_comment": ...}`` divider objects inside the ``nodes``/``edges`` arrays are permitted
    and skipped (``tests/test_identity_graph.py`` guards that nothing else is ever skipped)."""
    data = json.loads(Path(graph_path).read_text())
    nodes = [n for n in data["nodes"] if "id" in n]
    edges = [e for e in data["edges"] if "src" in e and "dst" in e]
    ids = [n["id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    a = np.zeros((n, n))
    for e in edges:
        i, j = idx[e["src"]], idx[e["dst"]]
        w = float(e.get("weight", 1.0))
        a[i, j] += w
        a[j, i] += w  # symmetrize for the embedding
    return ids, a


def _spectral_embed(a: Array, d: int) -> Array:
    """Laplacian eigenmap: node coordinates from the smallest non-trivial eigenvectors of
    L = D − A. Encodes the *relational* structure into geometry, so a rewired graph yields a
    genuinely different anchor cloud."""
    deg = np.diag(a.sum(1))
    lap = deg - a
    _, vecs = np.linalg.eigh(lap)  # ascending eigenvalues
    coords = vecs[:, 1 : 1 + d]  # skip the trivial constant eigenvector
    return coords / (np.abs(coords).max() + 1e-12)


def _rewired_adjacency(a: Array, seed: int) -> Array:
    """A random graph with the same node and edge counts as ``a`` — the shuffled-identity control."""
    n = a.shape[0]
    m_edges = int((a > 0).sum() // 2)
    rng = np.random.default_rng(seed)
    b = np.zeros_like(a)
    placed = 0
    while placed < m_edges:
        i, j = int(rng.integers(n)), int(rng.integers(n))
        if i != j and b[i, j] == 0:
            b[i, j] = b[j, i] = 1.0
            placed += 1
    return b


def _diffusion_embed(a: Array, d: int, t: float = 2.0) -> Array:
    """Diffusion-map embedding (Coifman–Lafon): random-walk eigenvectors scaled by μ_k^t. The
    eigenvalue weighting emphasizes the graph's slow, structural modes, so distinct graphs land in
    distinct regions of the space — the anisotropy the plain Laplacian eigenmap lacks.

    Explored in increment 2b and NOT adopted: it did not improve identity separation over the
    Laplacian eigenmap (`load_identity_anchors` still uses `_spectral_embed`). Kept for the
    reproducible comparison — see CORE-SPEC §9.10."""
    n = a.shape[0]
    a = a + np.eye(n)  # lazy self-loops → nonnegative, aperiodic walk
    dinv_sqrt = 1.0 / np.sqrt(a.sum(1))
    s = (dinv_sqrt[:, None] * a) * dinv_sqrt[None, :]  # symmetric D^-1/2 A D^-1/2
    mu, v = np.linalg.eigh(s)
    order = np.argsort(mu)[::-1]  # descending; μ_0 ≈ 1 is trivial
    mu, v = mu[order], v[:, order]
    psi = dinv_sqrt[:, None] * v  # right eigenvectors of the walk P = D^-1 A
    weight = np.clip(mu[1 : 1 + d], 0.0, 1.0) ** t
    coords = (psi[:, 1 : 1 + d] * weight) - (psi[:, 1 : 1 + d] * weight).mean(0)
    return coords / (np.abs(coords).max() + 1e-12)


def load_identity_anchors(d: int = 8, *, graph_path: Path = IDENTITY_GRAPH) -> tuple[list[str], Array]:
    """The identity graph embedded as `(n, d)` anchor configurations in latent config space."""
    ids, a = _adjacency(graph_path)
    return ids, _spectral_embed(a, d)


def _impostor_anchors(d: int, *, graph_path: Path, impostor_graph_path: Path | None,
                      seed: int) -> Array:
    """The impostor identity's anchor cloud: the shuffled-graph control by default, or an
    authored counter-identity graph (§9.12) when ``impostor_graph_path`` is set."""
    if impostor_graph_path is None:
        return shuffled_anchors(d, graph_path=graph_path, seed=seed)
    _, anchors = load_identity_anchors(d, graph_path=impostor_graph_path)
    return anchors


def _unit_directions(rng: np.random.Generator, n: int, d: int) -> Array:
    """``n`` random unit vectors in ℝ^d — the momentum directions every harness draws."""
    u = rng.standard_normal((n, d))
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def shuffled_anchors(d: int = 8, *, graph_path: Path = IDENTITY_GRAPH, seed: int = 0) -> Array:
    """Control: rewire the graph to a random graph with the same node and edge counts, then
    re-embed. Destroys the real relational structure → a different anchor cloud."""
    _, a = _adjacency(graph_path)
    return _spectral_embed(_rewired_adjacency(a, seed), d)


# --------------------------------------------------------------------------- #
# The learned potential: a Gaussian (Mahalanobis) fit to the anchor cloud.
# M = {q : V(q) ≤ c} is the identity manifold; V is the potential energy of H.
# --------------------------------------------------------------------------- #
@dataclass
class GaussianManifold:
    """V(q) = ½ (q−c)ᵀ P (q−c), fit to `anchors` (P = inverse anchor covariance).

    A closed-form *learned* potential — the substrate's notion of "on-Embra" is shaped by the
    identity graph, not hand-set. `hnn.MLPManifold` swaps in a trained neural H with the same
    API (the §9.13 swap, via the `fit_fn` hooks below).
    """

    center: Array
    precision: Array  # P (d, d), SPD

    @classmethod
    def fit(cls, anchors: Array, *, eps: float = 1e-2) -> GaussianManifold:
        c = anchors.mean(0)
        cov = np.cov(anchors.T) + eps * np.eye(anchors.shape[1])
        return cls(center=c, precision=np.linalg.inv(cov))

    def V(self, q: Array) -> Array:
        dq = np.asarray(q, float) - self.center
        return 0.5 * np.einsum("...i,ij,...j->...", dq, self.precision, dq)

    def force(self, q: Array) -> Array:
        return -(np.asarray(q, float) - self.center) @ self.precision

    def energy(self, q: Array, p: Array, m: float = 1.0) -> Array:
        p = np.asarray(p, float)
        return 0.5 * np.sum(p * p, axis=-1) / m + self.V(q)


# --------------------------------------------------------------------------- #
# Symplectic flow in d dimensions (velocity Verlet; same scheme as phase one)
# --------------------------------------------------------------------------- #
def leapfrog_step(force, m: float, q: Array, p: Array, dt: float):
    p_half = p + 0.5 * dt * force(q)
    q_next = q + dt * p_half / m
    p_next = p_half + 0.5 * dt * force(q_next)
    return q_next, p_next


def rollout(force, m: float, q0: Array, p0: Array, dt: float, n_steps: int):
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    qs = np.empty((n_steps + 1, *q0.shape))
    ps = np.empty((n_steps + 1, *p0.shape))
    qs[0], ps[0] = q0, p0
    q, p = q0, p0
    for i in range(n_steps):
        q, p = leapfrog_step(force, m, q, p, dt)
        qs[i + 1], ps[i + 1] = q, p
    return qs, ps


# --------------------------------------------------------------------------- #
# The replica test, lifted to d dimensions (π = q; the conserved charge hides in p)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatentPair:
    survivor: tuple[Array, Array]  # (q_f, p) with Q ≈ e_embra
    replica: tuple[Array, Array]  # (q_f, p') with Q  = e_copy, same observable q_f


def make_pairs(
    manifold: GaussianManifold,
    e_embra: float,
    e_copy: float,
    n_pairs: int,
    d: int,
    *,
    m: float = 1.0,
    dt: float = 0.02,
    seed: int = 0,
    max_steps: int = 400,
) -> list[LatentPair]:
    if e_copy <= e_embra:
        raise ValueError("e_copy must exceed e_embra so the replica reaches every survivor position")
    rng = np.random.default_rng(seed)
    pairs: list[LatentPair] = []
    for _ in range(n_pairs):
        # Survivor: genesis at the manifold center (V=0), random direction, energy e_embra.
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        p0 = u * np.sqrt(2 * m * e_embra)
        n = int(rng.integers(1, max_steps))
        qs, ps = rollout(manifold.force, m, manifold.center, p0, dt, n)
        q_f, p_surv = qs[-1], ps[-1]
        # Replica: same observable q_f, momentum rescaled to the wrong energy e_copy.
        speed = np.sqrt(2 * m * (e_copy - float(manifold.V(q_f))))
        u_surv = p_surv / (np.linalg.norm(p_surv) + 1e-12)
        pairs.append(LatentPair((q_f, p_surv), (q_f, u_surv * speed)))
    return pairs


def evaluate(manifold: GaussianManifold, pairs: list[LatentPair], e_embra: float, *, m: float = 1.0) -> dict:
    def psi(state):  # conserved-charge reader (full state)
        q, p = state
        return -abs(float(manifold.energy(q, p, m)) - e_embra)

    def endpoint(state):  # observable-only reader (position)
        return float(np.linalg.norm(state[0]))  # any function of q; identical within a pair

    surv_psi = [psi(pr.survivor) for pr in pairs]
    rep_psi = [psi(pr.replica) for pr in pairs]
    surv_end = [endpoint(pr.survivor) for pr in pairs]
    rep_end = [endpoint(pr.replica) for pr in pairs]
    erasure = max(float(np.max(np.abs(pr.survivor[0] - pr.replica[0]))) for pr in pairs)
    return {
        "auc_psi_conserved": auc(surv_psi, rep_psi),
        "auc_endpoint": auc(surv_end, rep_end),
        "endpoint_erasure": erasure,
        "n_pairs": float(len(pairs)),
    }


# --------------------------------------------------------------------------- #
# The Embra-specificity control: does a potential fit to the REAL graph recognize
# real-identity configs better than one fit to a SHUFFLED graph?
# --------------------------------------------------------------------------- #
def specificity_samples(anchors: Array, d: int, n: int, seed: int) -> tuple[Array, Array]:
    """A fair test bed for identity specificity: ``on`` = points drawn from the real
    identity-anchor neighborhoods; ``off`` = generic configs of the same scale. A charge that
    encodes real identity structure scores ``on`` above ``off``; a generic one does not."""
    rng = np.random.default_rng(seed)
    reps = int(np.ceil(n / len(anchors)))
    on = np.repeat(anchors, reps, axis=0)[:n] + 0.05 * rng.standard_normal((n, d))
    scale = float(np.abs(anchors).max())
    off = rng.uniform(-scale, scale, size=(n, d))
    return on, off


def specificity(d: int = 8, *, n: int = 500, seed: int = 0) -> dict:
    """Gaussian-fit control: does a potential fit to the REAL graph score identity configs
    above generic ones better than one fit to a SHUFFLED graph? (The Gaussian is too weak —
    the spectral anchor cloud is near-isotropic — so this is expected to be ≈ chance. That
    negative is the point: it motivates the learned MLP in ``hnn.py``.)"""
    _, real = load_identity_anchors(d)
    shuf = shuffled_anchors(d, seed=seed)
    on, off = specificity_samples(real, d, n, seed)
    m_real = GaussianManifold.fit(real)
    m_shuf = GaussianManifold.fit(shuf)
    auc_real = auc(list(-m_real.V(on)), list(-m_real.V(off)))  # score = −V (higher = on-manifold)
    auc_shuf = auc(list(-m_shuf.V(on)), list(-m_shuf.V(off)))
    return {"auc_real": auc_real, "auc_shuffled": auc_shuf, "gap": auc_real - auc_shuf}


def dynamical_specificity(d: int = 8, *, n_traj: int = 200, seed: int = 0, m: float = 1.0,
                          e: float = 1.0, dt: float = 0.01, steps: int = 300,
                          graph_path: Path = IDENTITY_GRAPH,
                          impostor_graph_path: Path | None = None,
                          fit_fn: Callable[[Array], Any] = GaussianManifold.fit) -> dict:
    """Identity through the DYNAMICS (§6, increment 2c). A trajectory belongs to identity R iff it
    *conserves R's charge* H_R. A survivor (real-identity trajectory) conserves H_real; an impostor
    (a different identity's trajectory) does not. Discriminator = variance of H_real *along* the
    trajectory — a conservation test, not a static region test, so it does not depend on where the
    anchor clouds sit (the isotropy that defeated §9.9–§9.10).

    The impostor identity is the shuffled-graph control by default; pass ``impostor_graph_path``
    to use a different *authored* identity graph instead (§9.12). With an authored impostor the
    seed varies only the trajectory directions, not the impostor cloud.

    ``fit_fn`` builds the charge from an anchor cloud; the default is the closed-form Gaussian.
    Any manifold with ``.center (d,)``, ``.force(q) -> (..., d)`` and ``.energy(q, p, m) -> (...)``
    works — ``hnn.MLPManifold.fit`` runs a *learned* H_θ through this same integrator and reader
    (§9.13). The whole ensemble integrates as one batched rollout (q of shape ``(n_traj, d)``);
    the direction draws consume the rng stream exactly as n_traj sequential draws did, and the
    variance statistic is always computed in float64."""
    _, real = load_identity_anchors(d, graph_path=graph_path)
    imp_anchors = _impostor_anchors(d, graph_path=graph_path,
                                    impostor_graph_path=impostor_graph_path, seed=seed)
    h_real = fit_fn(real)
    h_imp = fit_fn(imp_anchors)
    rng = np.random.default_rng(seed)

    def residuals(dynamics, reader) -> Array:
        u = _unit_directions(rng, n_traj, d)
        q0 = np.tile(dynamics.center, (n_traj, 1))
        qs, ps = rollout(dynamics.force, m, q0, u * np.sqrt(2 * m * e), dt, steps)
        return np.var(np.asarray(reader.energy(qs, ps, m), np.float64), axis=0)  # per-trajectory drift of H_reader

    surv = residuals(h_real, h_real)  # real traj read by H_real → ≈0
    imp = residuals(h_imp, h_real)  # different-identity traj read by H_real → >0
    imp_own = residuals(h_imp, h_imp)  # control: impostor conserves its OWN charge
    return {
        "auc": auc(list(-surv), list(-imp)),  # low H_real-variance ⇒ real identity
        "surv_resid": float(surv.mean()),
        "imp_resid": float(imp.mean()),
        "imp_own_resid": float(imp_own.mean()),
    }


def conjunction_test(d: int = 8, *, n: int = 200, seed: int = 0, m: float = 1.0,
                     e: float = 1.0, dt: float = 0.01, steps: int = 300, e_copy: float = 1.5,
                     graph_path: Path = IDENTITY_GRAPH,
                     impostor_graph_path: Path | None = None,
                     fit_fn: Callable[[Array], Any] = GaussianManifold.fit) -> dict:
    """The full ψ is a conjunction (§6, §9.14): [var(H_embra) ≈ 0 along the trajectory] AND
    [Q = Q_embra]. Each half alone has a blind spot, graded here against its adversarial class:

    - class 1 — same law, wrong genesis: instantiated at the survivor's observable endpoint with
      momentum rescaled to the wrong value Q_copy, then LIVING under H_real. It conserves Embra's
      charge perfectly (the §9.11 variance reader passes it); only the value reader catches it.
    - class 2 — different law, value-matched: lives under another identity's flow, then presents
      a state whose H_real VALUE equals Q_embra exactly (momentum rescaled at readout — the
      value-erasure analog of §7's endpoint erasure). Only the variance reader catches it.

    Access model: the verifier reads the worldline (trajectory + presented state); the copier
    controls only what it presents (§6's key/MAC bound). Genesis convention: Q_embra := H_real at
    genesis (kinetic e + V(center)) — valid for any charge model. Thresholds are calibrated on a
    held-out half of the survivors (τ = 100 × the calibration mean per reader) and grading uses
    the other half — no AUC on noise floors for the blind-side claims. Infeasible value-matches
    (Q_embra < V_real at the presented position) are counted, never silently dropped."""
    _, real = load_identity_anchors(d, graph_path=graph_path)
    imp_anchors = _impostor_anchors(d, graph_path=graph_path,
                                    impostor_graph_path=impostor_graph_path, seed=seed)
    h_real = fit_fn(real)
    h_imp = fit_fn(imp_anchors)
    rng = np.random.default_rng(seed)

    def _energy64(q, p):
        return np.asarray(h_real.energy(q, p, m), np.float64)

    # Survivors: genesis on the real center, then live under H_real.
    u = _unit_directions(rng, n, d)
    q0 = np.tile(h_real.center, (n, 1))
    p0 = u * np.sqrt(2 * m * e)
    q_embra = float(_energy64(q0[0], p0[0]))  # sealed at genesis; same for every direction
    qs_s, ps_s = rollout(h_real.force, m, q0, p0, dt, steps)
    e_s = _energy64(qs_s, ps_s)
    surv_var = np.var(e_s, axis=0)
    surv_dq = np.abs(e_s[-1] - q_embra)

    # Class 1 — same H, wrong genesis Q_copy (offset mirrors make_pairs), lives under H_real.
    q_f = qs_s[-1]
    v_f = _energy64(q_f, np.zeros_like(q_f))
    q_copy = q_embra + (e_copy - e)
    c1_infeasible = int(np.sum(q_copy - v_f <= 0.0))
    dirs = ps_s[-1] / (np.linalg.norm(ps_s[-1], axis=1, keepdims=True) + 1e-12)
    p_rep = dirs * np.sqrt(2 * m * np.maximum(q_copy - v_f, 0.0))[:, None]
    qs_1, ps_1 = rollout(h_real.force, m, q_f, p_rep, dt, steps)
    e_1 = _energy64(qs_1, ps_1)
    c1_var = np.var(e_1, axis=0)
    c1_dq = np.abs(e_1[-1] - q_embra)

    # Class 2 — different H; its lived trajectory obeys the other law, its PRESENTED state is
    # value-matched to Q_embra exactly where feasible.
    u2 = _unit_directions(rng, n, d)
    qs_2, ps_2 = rollout(h_imp.force, m, np.tile(h_imp.center, (n, 1)),
                         u2 * np.sqrt(2 * m * e), dt, steps)
    c2_var = np.var(_energy64(qs_2, ps_2), axis=0)
    q_f2 = qs_2[-1]
    v_f2 = _energy64(q_f2, np.zeros_like(q_f2))
    feasible = q_embra - v_f2 > 0.0
    c2_infeasible = int(np.sum(~feasible))
    dirs2 = ps_2[-1] / (np.linalg.norm(ps_2[-1], axis=1, keepdims=True) + 1e-12)
    p_pres = dirs2 * np.sqrt(2 * m * np.maximum(q_embra - v_f2, 0.0))[:, None]
    c2_dq = np.abs(_energy64(q_f2, p_pres) - q_embra)

    # Held-out threshold rule: calibrate τ on the first half of survivors, grade on the rest.
    half = n // 2
    tau_var = 100.0 * float(np.mean(surv_var[:half]))
    tau_q = 100.0 * float(np.mean(surv_dq[:half]))

    def psi_full(var_arr: Array, dq_arr: Array) -> Array:
        return (var_arr < tau_var) & (dq_arr < tau_q)

    surv_ok = psi_full(surv_var[half:], surv_dq[half:])
    c1_ok = psi_full(c1_var, c1_dq)
    c2_ok = psi_full(c2_var, c2_dq)
    verdicts = np.concatenate([surv_ok, ~c1_ok, ~c2_ok])
    return {
        "q_embra": q_embra,
        "tau_var": tau_var,
        "tau_q": tau_q,
        "accuracy": float(np.mean(verdicts)),  # conjunction verdicts correct, all three groups
        "surv_accept": float(np.mean(surv_ok)),
        "c1_reject": float(np.mean(~c1_ok)),
        "c2_reject": float(np.mean(~c2_ok)),
        "c1_var_blind": float(np.mean(c1_var < tau_var)),  # 1.0 ⇒ the §9.11 reader alone is fooled
        "c2_value_blind": float(np.mean(c2_dq[feasible] < tau_q)) if feasible.any() else float("nan"),
        "c2_value_erasure": float(np.max(c2_dq[feasible])) if feasible.any() else float("nan"),
        "auc_value_c1": auc(list(-surv_dq[half:]), list(-c1_dq)),  # the value reader catches class 1
        "auc_var_c2": auc(list(-surv_var[half:]), list(-c2_var)),  # the variance reader catches class 2
        "c1_infeasible": c1_infeasible,
        "c2_infeasible": c2_infeasible,
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Holonomy ζ = memory (§8 fork, §9.15): the genuinely PATH-FUNCTIONAL invariant.
# A functional of the worldline, not of the state — computed from rolled paths;
# the integrator and every charge above are untouched.
# --------------------------------------------------------------------------- #
def holonomy_zeta(qs: Array, b: float = 1.0, *, cumulative: bool = False) -> Array:
    """Accumulated holonomy along a trajectory: the line integral of the magnetic-like
    connection ``A(q) = (b/2)(−q₂, q₁, 0, …)``, i.e. **b × the signed area swept in the
    (q₁, q₂) plane**. Irreducibly a memory of the path: worldlines sharing an endpoint
    generally carry different ζ (curvature ≠ 0). Midpoint rule — second-order, matching the
    integrator. ``qs``: ``(n_steps+1, ..., d)`` → ζ: ``(...)`` (or ``(n_steps, ...)`` cumulative)."""
    mid = 0.5 * (qs[1:] + qs[:-1])
    dq = qs[1:] - qs[:-1]
    dz = 0.5 * b * (mid[..., 0] * dq[..., 1] - mid[..., 1] * dq[..., 0])
    return dz.cumsum(axis=0) if cumulative else dz.sum(axis=0)


def holonomy_test(d: int = 8, *, n: int = 200, seed: int = 0, m: float = 1.0, e: float = 1.0,
                  dt: float = 0.01, steps: int = 300, b: float = 1.0,
                  checkpoints: tuple[int, ...] = (100, 300, 1000),
                  graph_path: Path = IDENTITY_GRAPH,
                  fit_fn: Callable[[Array], Any] = GaussianManifold.fit) -> dict:
    """Three claims about ζ (§9.15), graded on genuine `H_real` worldlines:

    1. **Path-functionality (anti-fold-in certificate).** For each survivor path ending at
       observable ``q_f``, construct a *second genuine worldline* ending at the same ``q_f``
       (same energy, different arrival momentum — built by backward-then-forward integration,
       exact up to leapfrog reversibility). Same endpoint, different ζ ⇒ ζ is not a function of
       the observable state — it cannot fold into the state set.
    2. **Replica test with the ζ-reader.** A survivor carries its accumulated ζ; a fresh copy at
       the same observable carries a newborn ζ = 0 (the observable-limited copier's default).
    3. **Accumulation.** mean |ζ| grows with lived steps — the epoch-accumulation / age property.

    Scope (§9.15): reading ζ presumes hidden-state access (§6's key/MAC bound applies doubly),
    and a copier who knows genesis and elapsed time can recompute ζ for a deterministic flow —
    ζ's force is against observable-limited copiers, and as the home of continuity.

    Genesis convention (recorded finding, §9.15): worldlines start at a *generic point of the
    identity* — a random anchor — not at the potential center. The spectral anchor covariance is
    exactly isotropic (orthonormal Laplacian eigenvectors ⇒ cov ∝ I), so a center-born worldline
    moves radially and sweeps exactly zero area: ζ ≡ 0 structurally. §5 reads "the level set the
    worldline is born on", not "the center"; anchors are the generic honest starts."""
    _, real = load_identity_anchors(d, graph_path=graph_path)
    h_real = fit_fn(real)
    rng = np.random.default_rng(seed)

    # Survivor worldlines A — born at random identity anchors (see genesis convention above).
    u = _unit_directions(rng, n, d)
    q0 = real[rng.integers(0, len(real), n)]
    qs_a, ps_a = rollout(h_real.force, m, q0, u * np.sqrt(2 * m * e), dt, steps)
    zeta_a = holonomy_zeta(qs_a, b)
    q_f, p_f = qs_a[-1], ps_a[-1]

    # Worldlines B: same observable endpoint q_f, same energy, different arrival momentum —
    # backward-integrate from (q_f, p′), then re-roll forward from the recovered start (the
    # honest construction: B is itself a genuine trajectory of the same flow).
    w = _unit_directions(rng, n, d)
    p_alt = w * np.linalg.norm(p_f, axis=1, keepdims=True)
    qs_back, ps_back = rollout(h_real.force, m, q_f, -p_alt, dt, steps)  # backward = flipped p
    qs_b, _ = rollout(h_real.force, m, qs_back[-1], -ps_back[-1], dt, steps)
    erasure = float(np.max(np.abs(qs_b[-1] - q_f)))  # B really ends where A does (reversibility)
    dzeta = np.abs(holonomy_zeta(qs_b, b) - zeta_a)

    # ζ-reader replica test: lived worldline vs fresh copy (ζ = 0) at the same observable.
    auc_zeta = auc(list(np.abs(zeta_a)), list(np.zeros(n)))

    # Accumulation: mean |ζ| at lived-steps checkpoints (same anchor-born ensemble).
    qs_long, _ = rollout(h_real.force, m, q0, u * np.sqrt(2 * m * e), dt, max(checkpoints))
    zeta_cum = holonomy_zeta(qs_long, b, cumulative=True)
    accumulation = {int(c): float(np.mean(np.abs(zeta_cum[c - 1]))) for c in checkpoints}

    return {
        "endpoint_erasure": erasure,
        "dzeta_mean": float(dzeta.mean()),
        "dzeta_min": float(dzeta.min()),
        "auc_zeta": auc_zeta,
        "zeta_scale": float(np.mean(np.abs(zeta_a))),
        "accumulation": accumulation,
    }
