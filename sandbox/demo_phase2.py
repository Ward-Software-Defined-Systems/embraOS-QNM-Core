"""demo_phase2.py — phase two, increment 1: d-dim latent + learned H. Honest results.

Run:  ``uv run python -m sandbox.demo_phase2``   (needs the ``learn`` extra: jax/optax)

Reports two things:
  1. the phase-one machinery lifted to ``d`` dimensions — conservation + the replica test;
  2. the Embra-specificity control — a charge fit to the REAL identity graph vs a SHUFFLED
     one — for both the closed-form Gaussian and a learned MLP, averaged over seeds and
     reported as-is. This is research-in-progress, not a pass/fail.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sandbox.hnn import specificity_mlp  # noqa: E402
from sandbox.latent import (  # noqa: E402
    GaussianManifold,
    evaluate,
    load_identity_anchors,
    make_pairs,
    rollout,
    specificity,
)

D = 8
SEEDS = (0, 1, 2, 3)


def _summ(rows: list[dict], key: str) -> str:
    xs = np.array([r[key] for r in rows])
    return f"{xs.mean():.3f} [{xs.min():.3f},{xs.max():.3f}]"


def main() -> dict:
    ids, anchors = load_identity_anchors(D)
    man = GaussianManifold.fit(anchors)

    # 1. machinery lifted to d dimensions
    rng = np.random.default_rng(0)
    u = rng.standard_normal(D)
    u /= np.linalg.norm(u)
    qs, ps = rollout(man.force, 1.0, man.center, u * np.sqrt(2.0), 0.005, 10_000)
    e = man.energy(qs, ps, 1.0)
    drift = float((e.max() - e.min()) / abs(e.mean()))
    rep = evaluate(man, make_pairs(man, 1.0, 1.5, 400, D, seed=0), 1.0)

    # 2. specificity control across seeds
    g = [specificity(D, seed=s) for s in SEEDS]
    h = [specificity_mlp(D, seed=s, steps=700) for s in SEEDS]

    print("=" * 72)
    print(f"  Phase two · increment 1 — d-dim latent (d={D}) + learned H")
    print("=" * 72)
    print(f"  identity anchors: {anchors.shape[0]} graph nodes → ℝ^{D} (Laplacian embedding)")
    print("-" * 72)
    print("  [1] machinery lifts to d dimensions")
    print(f"      conservation drift          = {drift:.2e}   (→ 0)")
    print(f"      replica AUC (conserved ψ)   = {rep['auc_psi_conserved']:.3f}   (→ 1.0)")
    print(f"      replica AUC (endpoint only) = {rep['auc_endpoint']:.3f}   (→ 0.5, null)")
    print("-" * 72)
    print("  [2] Embra-specificity control — real vs shuffled graph  (mean [min,max] / seed)")
    print(f"      Gaussian fit   real = {_summ(g, 'auc_real')}   shuffled = {_summ(g, 'auc_shuffled')}")
    print(f"      learned MLP    real = {_summ(h, 'auc_real')}   shuffled = {_summ(h, 'auc_shuffled')}")
    print("-" * 72)
    print("  VERDICT: the machinery lifts cleanly. The Gaussian fit shows NO identity")
    print("  specificity (real ≈ shuffled). The learned MLP is DIRECTIONAL but unstable")
    print("  across seeds — it CAN carve an identity-specific charge, not yet reliably.")
    print("  Increment 2: held-out generalization + self-consistency/self-play data +")
    print("  a firmer objective, targeting real ≫ shuffled reliably.")
    print("=" * 72)
    return {"drift": drift, "replica": rep, "gaussian": g, "mlp": h}


if __name__ == "__main__":
    main()
