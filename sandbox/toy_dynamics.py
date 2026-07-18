"""toy_dynamics.py — the smallest substrate with a *native* conserved charge.

A 1-DOF Hamiltonian system:

    state        s = (q, p)                     ∈ phase space ℝ²
    Hamiltonian  H(q, p) = p²/2m + V(q),        V(q) = ½k q² + ¼ε q⁴
    conserved    Q(s)   = H(q, p)               (energy — constant along the flow)
    observable   π(s)   = q                      (position only; momentum is HIDDEN)

The point of the toy: `Q` is conserved *by the dynamics* (a symplectic integrator
preserves it), and it lives in the part of the state the observable erases. That is
the whole thesis of `docs/CORE-SPEC.md` in miniature — identity as an invariant of
motion, native to the substrate, hidden behind the readout. ``ε = 0`` is the exact
harmonic oscillator; ``ε > 0`` is a Duffing-type anharmonic well, included so the
demonstration does not secretly depend on linearity.

Next phase replaces this hand-built `Q` with one *learned* on the identity manifold
(a Hamiltonian Neural Network); the machinery here — flow, conservation, readout —
is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Hamiltonian:
    """H(q, p) = p²/2m + V(q), with V(q) = ½k q² + ¼ε q⁴.

    All methods are vectorized: ``q``/``p`` may be scalars or arrays of any shape.
    """

    m: float = 1.0
    k: float = 1.0
    eps: float = 0.0

    def V(self, q) -> Array:
        q = np.asarray(q, float)
        return 0.5 * self.k * q**2 + 0.25 * self.eps * q**4

    def dVdq(self, q) -> Array:
        q = np.asarray(q, float)
        return self.k * q + self.eps * q**3

    def force(self, q) -> Array:
        return -self.dVdq(q)

    def kinetic(self, p) -> Array:
        p = np.asarray(p, float)
        return p**2 / (2.0 * self.m)

    def energy(self, q, p) -> Array:
        """The conserved charge Q = H(q, p)."""
        return self.kinetic(p) + self.V(q)


def conserved_charge(ham: Hamiltonian, q, p) -> Array:
    """Q(s) — the identity charge. Alias for the energy, named for the spec."""
    return ham.energy(q, p)


def observable(q, p) -> Array:
    """π(s): the externally visible readout — position only.

    Momentum, and therefore the conserved charge, is the hidden complement. A reader
    restricted to π cannot recover Q — which is exactly why an endpoint-only ψ is
    blind to the survivor/replica distinction.
    """
    return np.asarray(q, float)


# --------------------------------------------------------------------------- #
# Symplectic flow (velocity Verlet / Störmer–Verlet): time-reversible and
# volume-preserving, so H is bounded (no secular energy drift) — the numerical
# reason "conservation beats checking" is realizable rather than aspirational.
# --------------------------------------------------------------------------- #
def leapfrog_step(ham: Hamiltonian, q, p, dt: float):
    """One symplectic step of the flow Φ_H."""
    p_half = p + 0.5 * dt * ham.force(q)
    q_next = q + dt * p_half / ham.m
    p_next = p_half + 0.5 * dt * ham.force(q_next)
    return q_next, p_next


def rollout(ham: Hamiltonian, q0, p0, dt: float, n_steps: int):
    """Integrate ``n_steps`` of the flow from (q0, p0).

    Returns ``(qs, ps)`` each of shape ``(n_steps + 1, *shape(q0))`` — the full
    trajectory including the initial point. ``q0``/``p0`` may be scalars or arrays
    (arrays integrate many independent trajectories at once).
    """
    q0 = np.asarray(q0, float)
    p0 = np.asarray(p0, float)
    qs = np.empty((n_steps + 1, *q0.shape), float)
    ps = np.empty((n_steps + 1, *p0.shape), float)
    qs[0], ps[0] = q0, p0
    q, p = q0, p0
    for i in range(n_steps):
        q, p = leapfrog_step(ham, q, p, dt)
        qs[i + 1], ps[i + 1] = q, p
    return qs, ps


def turning_point(ham: Hamiltonian, E: float) -> float:
    """Outer turning point q > 0 where V(q) = E (so p = 0 there): the natural place
    to start a trajectory on the energy-E orbit."""
    if ham.eps == 0.0:
        return float(np.sqrt(2.0 * E / ham.k))
    # Quartic in u = q²:  (ε/4) u² + (k/2) u − E = 0  → take the positive root.
    a, b, c = ham.eps / 4.0, ham.k / 2.0, -E
    u = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    return float(np.sqrt(u))


def momentum_for_energy(ham: Hamiltonian, E: float, q, sign: float = 1.0) -> Array:
    """The momentum p such that H(q, p) = E — i.e. the state on the energy-E orbit at
    position ``q``. This is how a *replica* is placed at a given observable but on the
    wrong (non-genesis) energy level."""
    kinetic = np.asarray(E - ham.V(q), float)
    if np.any(kinetic < -1e-12):
        raise ValueError("energy below potential at q (classically forbidden turning region)")
    return sign * np.sqrt(np.clip(2.0 * ham.m * kinetic, 0.0, None))
