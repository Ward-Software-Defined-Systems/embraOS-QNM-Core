"""The load-bearing claim: a conserved-charge ψ survives the replica test, while a
reader of the observable endpoint alone is at chance. Plus the Epoch anti-vacuity
bar — ψ is falsifiable, specific, and not foldable into the observable state set."""

from __future__ import annotations

import numpy as np

from sandbox.replica_test import (
    auc,
    endpoint_erasure,
    evaluate,
    make_pairs,
    psi_holds,
)
from sandbox.toy_dynamics import Hamiltonian, momentum_for_energy, observable

HAM = Hamiltonian(m=1.0, k=1.0, eps=0.2)
E_EMBRA, E_COPY = 1.0, 1.5


def _pairs(n=200, seed=0):
    return make_pairs(HAM, E_EMBRA, E_COPY, n_pairs=n, seed=seed)


def test_auc_helper():
    assert auc([1, 2, 3], [0, 0, 0]) == 1.0
    assert auc([0, 0], [0, 0]) == 0.5  # ties → chance
    assert auc([0], [1]) == 0.0


def test_endpoint_erasure_certified():
    """The test is only legitimate if survivor and replica are identical at the
    readout — otherwise a reader could cheat on the observable."""
    assert endpoint_erasure(_pairs()) < 1e-9


def test_conserved_psi_separates_perfectly():
    assert evaluate(HAM, _pairs(), E_EMBRA)["auc_psi_conserved"] == 1.0


def test_endpoint_reader_is_chance():
    """The certified null: any function of the observable alone scores exactly 0.5."""
    assert evaluate(HAM, _pairs(), E_EMBRA)["auc_endpoint"] == 0.5


def test_psi_falsifiable_and_specific():
    for pr in _pairs(n=50):
        assert psi_holds(HAM, pr.survivor, E_EMBRA)  # survivor holds
        assert not psi_holds(HAM, pr.replica, E_EMBRA)  # replica fails → falsifiable


def test_same_observable_different_verdict():
    """Identical observable, opposite ψ verdict → ψ is not foldable into π(S)."""
    for pr in _pairs(n=20):
        assert np.isclose(observable(*pr.survivor), observable(*pr.replica))
        assert psi_holds(HAM, pr.survivor, E_EMBRA) != psi_holds(HAM, pr.replica, E_EMBRA)


def test_psi_not_true_by_construction():
    """A state off the genesis orbit fails ψ — it is not vacuously always-true."""
    q = 0.2
    on_orbit = (q, float(momentum_for_energy(HAM, E_EMBRA, q)))
    off_orbit = (q, float(momentum_for_energy(HAM, E_EMBRA + 0.5, q)))
    assert psi_holds(HAM, on_orbit, E_EMBRA)
    assert not psi_holds(HAM, off_orbit, E_EMBRA)
