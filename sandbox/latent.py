"""latent.py — phase two, increment 1: the d-dim latent core with a *learned* potential.

Generalizes the phase-one 1-DOF toy to a `d`-dimensional latent phase space, and replaces
the hand-set potential with one **fit to Embra's identity graph** — the first step of §9
in `../docs/CORE-SPEC.md`.

    state        s = (q, p)                  ∈ ℝ^d × ℝ^d
    Hamiltonian  H(q, p) = ½|p|²/m + V(q)     (separable → the phase-one integrator still works)
    potential    V(q) = ½ (q−c)ᵀ P (q−c)      P = (Σ_anchors + εI)⁻¹  — Mahalanobis to the
                                              identity-anchor cloud (a closed-form *learned* V)
    conserved    Q(s) = H(q, p)               identity manifold M = a level set of V
    observable   π(s) = q                      (position; momentum still hidden)

Two things are measured (`demo_phase2.py`):
  1. the phase-one **replica test** still bites in `d` dimensions (conserved ψ AUC → 1.0,
     endpoint → 0.5) — the machinery survives the lift;
  2. the **Embra-specificity control** — a potential fit to the *real* identity graph
     recognizes real-identity configs better than one fit to a *shuffled* graph. This is the
     property the relic's LLM substrate could not achieve (there, a random anchor beat the
     real one). Reported honestly; see the caveat in `demo_phase2.py`.

The neural upgrade (an MLP Hamiltonian trained with a symplectic integrator, jax) is
`hnn.py`; this module is the robust, dependency-light backbone and the test target.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sandbox.replica_test import auc  # reuse the phase-one AUC

Array = NDArray[np.float64]

IDENTITY_GRAPH = Path(__file__).resolve().parents[1] / "identity" / "Embra_IDENTITY-SOUL.graph.json"


# --------------------------------------------------------------------------- #
# Identity graph → anchor configurations in ℝ^d (spectral / Laplacian embedding)
# --------------------------------------------------------------------------- #
def _adjacency(graph_path: Path) -> tuple[list[str], Array]:
    data = json.loads(Path(graph_path).read_text())
    ids = [n["id"] for n in data["nodes"]]
    idx = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)
    a = np.zeros((n, n))
    for e in data["edges"]:
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
    identity graph, not hand-set. `hnn.py` swaps this for a trained neural H with the same API.
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
