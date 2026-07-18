"""The substrate must actually conserve its charge — else "conservation beats
checking" is a slogan, not a mechanism. These tests pin the numerics of the flow."""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.toy_dynamics import (
    Hamiltonian,
    momentum_for_energy,
    rollout,
    turning_point,
)


def test_turning_point_matches_energy():
    for ham in [Hamiltonian(k=1.0, eps=0.0), Hamiltonian(k=1.5, eps=0.3)]:
        for e in (0.5, 1.0, 2.0):
            q = turning_point(ham, e)
            assert np.isclose(float(ham.V(q)), e)


def test_momentum_for_energy_roundtrip():
    ham = Hamiltonian(k=1.0, eps=0.2)
    e = 1.3
    qs = np.linspace(-0.5, 0.5, 11)
    p = momentum_for_energy(ham, e, qs)
    assert np.allclose(ham.energy(qs, p), e)


@pytest.mark.parametrize("eps,tol", [(0.0, 1e-4), (0.2, 1e-2)])
def test_energy_conserved_over_long_rollout(eps, tol):
    """Symplectic (velocity-Verlet) flow keeps H bounded — no secular drift."""
    ham = Hamiltonian(m=1.0, k=1.0, eps=eps)
    e = 1.0
    qs, ps = rollout(ham, turning_point(ham, e), 0.0, dt=0.01, n_steps=20_000)
    energies = ham.energy(qs, ps)
    rel_drift = (energies.max() - energies.min()) / abs(energies.mean())
    assert rel_drift < tol


def test_time_reversibility():
    """Flip the momentum and re-integrate → you retrace the path back to genesis.
    A defining property of a symplectic, time-symmetric integrator."""
    ham = Hamiltonian(k=1.0, eps=0.2)
    q0, p0 = 0.7, 0.3
    n, dt = 5_000, 0.01
    qs, ps = rollout(ham, q0, p0, dt, n)
    qs2, ps2 = rollout(ham, qs[-1], -ps[-1], dt, n)
    assert np.isclose(qs2[-1], q0, atol=1e-5)
    assert np.isclose(ps2[-1], -p0, atol=1e-5)
