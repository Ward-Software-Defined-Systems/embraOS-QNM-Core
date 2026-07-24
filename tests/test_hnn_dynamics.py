"""Increment 3b (§9.13): the dynamical-specificity protocol under a LEARNED H_θ.

Mirrors ``test_latent.test_dynamical_specificity_is_reliable`` with ``MLPManifold`` swapped in
for the closed-form Gaussian — same symplectic integrator, same conservation reader, same bars.
The training configuration tested here (800 steps) is the recorded/demo configuration. Requires
the ``learn`` extra (jax/optax); skips cleanly without it.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

pytest.importorskip("jax")

from sandbox.hnn import MLPManifold  # noqa: E402
from sandbox.latent import (  # noqa: E402
    dynamical_specificity,
    holonomy_test,
    load_identity_anchors,
    rollout,
)

D = 8


def test_learned_h_dynamical_specificity():
    """The §9.11 bars must survive the Gaussian → learned-H_θ swap."""
    r = dynamical_specificity(D, seed=0, n_traj=100, fit_fn=partial(MLPManifold.fit, seed=0))
    assert r["auc"] == 1.0  # conservation still perfectly discriminates identity
    assert r["imp_resid"] > 100 * r["surv_resid"] + 1e-6  # impostor breaks H_θ; survivor doesn't
    assert r["imp_own_resid"] < r["imp_resid"]  # impostor conserves its OWN learned charge


def test_learned_h_conservation_smoke():
    """Leapfrog must conserve the learned H_θ too (float32 V_θ, float64 statistics) — the analog
    of ``test_energy_conserved_d_dim`` for the trained potential."""
    _, anchors = load_identity_anchors(D)
    man = MLPManifold.fit(anchors, seed=0)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    qs, ps = rollout(man.force, 1.0, man.center, u * np.sqrt(2.0), dt=0.005, n_steps=10_000)
    e = man.energy(qs, ps, 1.0)
    assert (e.max() - e.min()) / abs(e.mean()) < 2e-2


def test_learned_h_holonomy():
    """§9.15's learned-charge row, guarded: ζ's bars must hold under the trained potential too —
    the regime where the sweep rate is NOT conserved and ζ is genuinely history-integral."""
    r = holonomy_test(D, seed=0, n=100, fit_fn=partial(MLPManifold.fit, seed=0))
    assert r["endpoint_erasure"] < 1e-6
    assert r["dzeta_min"] > 1e-6
    assert r["dzeta_mean"] > 0.1 * r["zeta_scale"]
    assert r["auc_zeta"] == 1.0
    acc = r["accumulation"]
    ks = sorted(acc)
    assert all(acc[a] < acc[b] for a, b in zip(ks, ks[1:], strict=False))


def test_manifold_adapter_contract():
    """The duck-typed contract ``dynamical_specificity`` relies on: shapes, dtypes, finiteness —
    for single states, ensemble batches, and full trajectory arrays."""
    _, anchors = load_identity_anchors(D)
    man = MLPManifold.fit(anchors, seed=0, steps=50)  # tiny train — contract only, not quality
    assert man.center.shape == (D,)
    f = man.force(np.zeros((5, D)))
    assert f.shape == (5, D) and f.dtype == np.float64
    en = man.energy(np.zeros((3, 5, D)), np.ones((3, 5, D)), 1.0)
    assert en.shape == (3, 5) and en.dtype == np.float64
    assert np.all(np.isfinite(en))
    assert np.all(man.V(anchors) >= 0.0)  # softplus potential is nonnegative by construction
