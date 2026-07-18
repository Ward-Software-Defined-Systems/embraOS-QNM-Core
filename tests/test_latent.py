"""Phase-two increment 1: the d-dim machinery must hold as robustly as phase one.

These tests cover the *machinery* (conservation + the replica test lifted to d dimensions) and
that the specificity harness runs — NOT the learned-MLP specificity result, which is
research-in-progress and reported (honestly, with its variance) by ``sandbox/demo_phase2.py``.
The MLP path (jax) is deliberately kept out of the default test suite.
"""

from __future__ import annotations

import numpy as np

from sandbox.latent import (
    GaussianManifold,
    dynamical_specificity,
    evaluate,
    load_identity_anchors,
    make_pairs,
    rollout,
    specificity,
)

D = 8


def test_anchors_from_identity_graph():
    ids, anchors = load_identity_anchors(D)
    assert anchors.shape == (len(ids), D)  # one anchor per graph node, in ℝ^D
    assert len(ids) >= 22  # 20 original + Voice + "Honoring the restoration"; grows with richer content
    assert np.allclose(anchors.mean(0), 0.0, atol=1e-6)  # spectral embedding is mean-centered


def test_energy_conserved_d_dim():
    _, anchors = load_identity_anchors(D)
    man = GaussianManifold.fit(anchors)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    qs, ps = rollout(man.force, 1.0, man.center, u * np.sqrt(2.0), dt=0.005, n_steps=10_000)
    e = man.energy(qs, ps, 1.0)
    assert (e.max() - e.min()) / abs(e.mean()) < 2e-2


def test_replica_bites_in_d_dim():
    _, anchors = load_identity_anchors(D)
    man = GaussianManifold.fit(anchors)
    m = evaluate(man, make_pairs(man, 1.0, 1.5, 200, D, seed=0), 1.0)
    assert m["auc_psi_conserved"] == 1.0  # conserved charge tells survivor from copy
    assert m["auc_endpoint"] == 0.5  # observable-only reader is blind (the null)
    assert m["endpoint_erasure"] < 1e-9  # legitimately endpoint-blind


def test_specificity_harness_runs():
    s = specificity(D, seed=0)
    for k in ("auc_real", "auc_shuffled"):
        assert 0.0 <= s[k] <= 1.0


def test_dynamical_specificity_is_reliable():
    """Identity through the dynamics: a trajectory conserves H_real iff it is the real identity.
    Reliable (deterministic) — unlike the static region-membership harness above."""
    r = dynamical_specificity(D, seed=0, n_traj=100)
    assert r["auc"] == 1.0  # conservation perfectly discriminates identity
    assert r["imp_resid"] > 100 * r["surv_resid"] + 1e-6  # impostor breaks H_real; survivor doesn't
    assert r["imp_own_resid"] < r["imp_resid"]  # impostor conserves its OWN charge (well-posed)
