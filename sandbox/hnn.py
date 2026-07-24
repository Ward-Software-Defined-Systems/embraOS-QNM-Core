"""hnn.py — a learned neural Hamiltonian potential (jax), phase two (§9 of CORE-SPEC).

Where the closed-form Gaussian fit (``latent.py``) cannot carve an identity-specific level set
— the spectral anchor cloud is *exactly* isotropic at the covariance level (§9.15), so
mean+covariance throws the identity away — a small MLP potential can. `V_θ(q) =
softplus(MLP(q))` is trained contrastively so identity-anchor neighborhoods are low-energy
(on-manifold) and generic configs are high-energy. The Hamiltonian stays separable,
`H = ½|p|² + V_θ`, so the phase-one symplectic integrator conserves the charge; `force = −∇V_θ`
comes from autodiff. ``MLPManifold`` puts the trained potential behind the ``GaussianManifold``
API, so every `latent.py` harness (dynamical specificity, the conjunction test, holonomy) runs
the learned charge through the same integrator and readers (§9.13–§9.15).

Requires the ``learn`` extra (jax, optax); `tests/test_hnn_dynamics.py` covers it and skips
cleanly without jax.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from sandbox.latent import Array, load_identity_anchors, shuffled_anchors, specificity_samples
from sandbox.replica_test import auc


def _init(d: int, hidden: int, key) -> dict:
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "W1": jax.random.normal(k1, (d, hidden)) / jnp.sqrt(d), "b1": jnp.zeros(hidden),
        "W2": jax.random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden), "b2": jnp.zeros(hidden),
        "W3": jax.random.normal(k3, (hidden, 1)) / jnp.sqrt(hidden), "b3": jnp.zeros(1),
    }


def _mlp(params: dict, q):
    h = jnp.tanh(q @ params["W1"] + params["b1"])
    h = jnp.tanh(h @ params["W2"] + params["b2"])
    return (h @ params["W3"] + params["b3"])[..., 0]


def potential_V(params: dict, q):
    """V_θ(q) = softplus(MLP(q)) ≥ 0."""
    return jax.nn.softplus(_mlp(params, q))


def train_potential(anchors, d: int, *, hidden: int = 64, steps: int = 800,
                    margin: float = 2.0, noise: float = 0.05, seed: int = 0) -> dict:
    """Contrastive fit: pull identity-anchor neighborhoods to low V, push generic configs to
    V ≥ margin. The learned well *is* the identity manifold M."""
    params = _init(d, hidden, jax.random.PRNGKey(seed))
    anchors_j = jnp.asarray(anchors, jnp.float32)
    scale = float(np.abs(anchors).max())
    opt = optax.adam(3e-3)
    state = opt.init(params)

    def loss(params, pos, neg):
        vp = potential_V(params, pos)
        vn = potential_V(params, neg)
        wd = sum(jnp.sum(params[w] ** 2) for w in ("W1", "W2", "W3"))
        return jnp.mean(vp) + jnp.mean(jax.nn.relu(margin - vn)) + 1e-4 * wd

    @jax.jit
    def step(params, state, pos, neg):
        val, grad = jax.value_and_grad(loss)(params, pos, neg)
        updates, state = opt.update(grad, state)
        return optax.apply_updates(params, updates), state, val

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        pos = anchors_j + noise * jnp.asarray(rng.standard_normal(anchors.shape), jnp.float32)
        neg = jnp.asarray(rng.uniform(-scale, scale, (128, d)), jnp.float32)
        params, state, _ = step(params, state, pos, neg)
    return params


@jax.jit
def _force_fn(params: dict, q):
    """−∇_q V_θ, per row: the gradient of the *summed* batched potential (rows are independent),
    so one jitted call serves both a single ``(d,)`` state and an ensemble ``(n, d)`` batch."""
    return -jax.grad(lambda qq: jnp.sum(potential_V(params, qq)))(q)


@dataclass
class MLPManifold:
    """The trained MLP potential ``V_θ`` behind the ``GaussianManifold`` API (``.center`` /
    ``.force`` / ``.energy`` / ``.V``), so ``latent.dynamical_specificity`` runs a *learned* H_θ
    through the SAME symplectic integrator and conservation reader (§9.13) — the swap the
    ``GaussianManifold`` docstring promises. numpy float64 at the boundary, jax float32 inside;
    the kinetic term and the variance statistic never touch float32."""

    params: dict
    center: Array  # rollout start — the anchor centroid, mirroring GaussianManifold.center

    @classmethod
    def fit(cls, anchors: Array, *, hidden: int = 64, steps: int = 800, margin: float = 2.0,
            noise: float = 0.05, seed: int = 0) -> MLPManifold:
        anchors = np.asarray(anchors, float)
        params = train_potential(anchors, anchors.shape[1], hidden=hidden, steps=steps,
                                 margin=margin, noise=noise, seed=seed)
        return cls(params=params, center=anchors.mean(0))

    def V(self, q: Array) -> Array:
        return np.asarray(potential_V(self.params, jnp.asarray(q, jnp.float32)), np.float64)

    def force(self, q: Array) -> Array:
        return np.asarray(_force_fn(self.params, jnp.asarray(q, jnp.float32)), np.float64)

    def energy(self, q: Array, p: Array, m: float = 1.0) -> Array:
        p = np.asarray(p, np.float64)
        return 0.5 * np.sum(p * p, axis=-1) / m + self.V(q)


def _score(params, x) -> list[float]:
    return list(-np.asarray(potential_V(params, jnp.asarray(x, jnp.float32))))  # −V: higher = on-M


def specificity_mlp(d: int = 8, *, n: int = 500, seed: int = 0, steps: int = 800) -> dict:
    """The learned-H version of the Embra-specificity control: train V_θ on the real graph and
    on a shuffled graph, then see which recognizes real-identity configs. Real ≫ shuffled = the
    substrate carries an identity-specific charge (the property the LLM lacked)."""
    _, real = load_identity_anchors(d)
    shuf = shuffled_anchors(d, seed=seed)
    on, off = specificity_samples(real, d, n, seed)
    p_real = train_potential(real, d, steps=steps, seed=seed)
    p_shuf = train_potential(shuf, d, steps=steps, seed=seed)
    auc_real = auc(_score(p_real, on), _score(p_real, off))
    auc_shuf = auc(_score(p_shuf, on), _score(p_shuf, off))
    return {"auc_real": auc_real, "auc_shuffled": auc_shuf, "gap": auc_real - auc_shuf}


# --------------------------------------------------------------------------- #
# Increment 2 — the harder bar: does the learned charge GENERALIZE to identity
# configs it was never trained on? Train on samples of the identity manifold (the
# convex hull of the anchors) and test on HELD-OUT samples of it.
# --------------------------------------------------------------------------- #
def _sample_manifold(anchors, n: int, noise: float, rng) -> np.ndarray:
    """Sample the identity manifold: convex combinations of anchors + transverse noise."""
    w = rng.dirichlet(np.ones(len(anchors)), size=n)
    return w @ anchors + noise * rng.standard_normal((n, anchors.shape[1]))


def _train_pool(pos_pool, d: int, scale: float, *, hidden: int = 64, steps: int = 800,
                margin: float = 2.0, seed: int = 0) -> dict:
    """Train V_θ from a fixed pool of on-manifold positives (minibatched) vs generic negatives."""
    params = _init(d, hidden, jax.random.PRNGKey(seed))
    opt = optax.adam(3e-3)
    state = opt.init(params)
    pool = jnp.asarray(pos_pool, jnp.float32)

    def loss(params, pos, neg):
        wd = sum(jnp.sum(params[w] ** 2) for w in ("W1", "W2", "W3"))
        return jnp.mean(potential_V(params, pos)) + jnp.mean(jax.nn.relu(margin - potential_V(params, neg))) + 1e-4 * wd

    @jax.jit
    def step(params, state, pos, neg):
        val, grad = jax.value_and_grad(loss)(params, pos, neg)
        updates, state = opt.update(grad, state)
        return optax.apply_updates(params, updates), state, val

    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, len(pos_pool), 128)
        neg = jnp.asarray(rng.uniform(-scale, scale, (128, d)), jnp.float32)
        params, state, _ = step(params, state, pool[idx], neg)
    return params


def generalization_specificity(d: int = 8, *, n: int = 400, seed: int = 0, steps: int = 800,
                               noise: float = 0.05) -> dict:
    """Held-out specificity: train V_θ on samples of the REAL identity manifold (convex hull of
    the anchors) and of the SHUFFLED manifold, then test both on HELD-OUT real-manifold configs
    vs generic. Real ≫ shuffled here = *generalizing* identity specificity, not memorization."""
    _, real = load_identity_anchors(d)
    shuf = shuffled_anchors(d, seed=seed)
    rng = np.random.default_rng(seed)
    scale = float(np.abs(real).max())
    real_train = _sample_manifold(real, 512, noise, rng)
    shuf_train = _sample_manifold(shuf, 512, noise, rng)
    test_on = _sample_manifold(real, n, noise, rng)  # HELD-OUT real-manifold configs
    test_off = rng.uniform(-scale, scale, (n, d))
    p_real = _train_pool(real_train, d, scale, steps=steps, seed=seed)
    p_shuf = _train_pool(shuf_train, d, scale, steps=steps, seed=seed)
    auc_real = auc(_score(p_real, test_on), _score(p_real, test_off))
    auc_shuf = auc(_score(p_shuf, test_on), _score(p_shuf, test_off))
    return {"auc_real": auc_real, "auc_shuffled": auc_shuf, "gap": auc_real - auc_shuf}
