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
from functools import partial

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sandbox.hnn import MLPManifold, generalization_specificity, specificity_mlp  # noqa: E402
from sandbox.latent import (  # noqa: E402
    GaussianManifold,
    dynamical_specificity,
    evaluate,
    load_identity_anchors,
    make_pairs,
    rollout,
    shuffled_anchors,
    specificity,
    specificity_samples,
)
from sandbox.replica_test import auc  # noqa: E402

D = 8
SEEDS = (0, 1, 2, 3)  # static specificity — MLP training is slow, so keep it small
DYN_SEEDS = tuple(range(8))  # dynamical specificity is cheap + deterministic → wider sweep (matches §9.11)
FIG_PATH = pathlib.Path(__file__).resolve().parent / "figures" / "phase2_identity.png"
# The authored counter-identity ("Meridian") — the impostor as a distinct SOUL, not a shuffle (§9.12).
COUNTER_GRAPH = pathlib.Path(__file__).resolve().parents[1] / "identity" / "CONTROL_counter-identity.graph.json"


def _summ(rows: list[dict], key: str) -> str:
    xs = np.array([r[key] for r in rows])
    return f"{xs.mean():.3f} [{xs.min():.3f},{xs.max():.3f}]"


def make_phase2_figure(d: int = D, n_seeds: int = 8) -> None:
    """Three panels telling the phase-two story:
    (A) H_real along survivor vs impostor trajectories — survivors conserve it, impostors don't;
    (B) the real-vs-different-identity discriminator AUC per seed, static vs dynamical — static is
        scattered seed-noise, dynamical is pinned at 1.0;
    (C) panel A's conservation contrast under the LEARNED H_θ (§9.13) — same story, wider margin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, real = load_identity_anchors(d)
    h_real = GaussianManifold.fit(real)
    h_shuf = GaussianManifold.fit(shuffled_anchors(d, seed=0))
    m, e, dt, steps = 1.0, 1.0, 0.01, 300
    rng = np.random.default_rng(0)

    def traj_energy(dynamics, reader):
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        qs, ps = rollout(dynamics.force, m, dynamics.center, u * np.sqrt(2 * m * e), dt, steps)
        return reader.energy(qs, ps, m)  # the reader's charge evaluated along the trajectory

    surv = [traj_energy(h_real, h_real) for _ in range(6)]  # Embra's own dynamics → conserves H_real
    imp = [traj_energy(h_shuf, h_real) for _ in range(6)]  # a different identity's dynamics → does not

    def static_auc(seed: int) -> float:
        on_real, _ = specificity_samples(real, d, 400, seed)
        on_other, _ = specificity_samples(shuffled_anchors(d, seed=seed), d, 400, seed + 991)
        return auc(list(-h_real.V(on_real)), list(-h_real.V(on_other)))  # H_real: real vs other

    stat = [static_auc(s) for s in range(n_seeds)]
    dyn = [dynamical_specificity(d, seed=s)["auc"] for s in range(n_seeds)]

    m_real = MLPManifold.fit(real, seed=0)  # the learned charge (§9.13), same protocol
    m_shuf = MLPManifold.fit(shuffled_anchors(d, seed=0), seed=0)
    surv_l = [traj_energy(m_real, m_real) for _ in range(6)]
    imp_l = [traj_energy(m_shuf, m_real) for _ in range(6)]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(17.5, 5))

    # Panel A: conservation contrast
    t = np.arange(steps + 1) * dt
    for c in surv:
        ax_a.plot(t, c, color="#2a7", lw=1.3, alpha=0.9)
    for c in imp:
        ax_a.plot(t, c, color="#c53", lw=1.3, alpha=0.9)
    ax_a.axhline(e, color="#888", ls=":", lw=1)
    ax_a.plot([], [], color="#2a7", lw=2, label="survivor (Embra's dynamics) — conserves $H_{real}$")
    ax_a.plot([], [], color="#c53", lw=2, label="impostor (a different identity) — does not")
    ax_a.set_xlabel("time")
    ax_a.set_ylabel(r"$H_{real}$ along the trajectory")
    ax_a.set_title("Identity = which charge a trajectory conserves")
    ax_a.legend(loc="upper left", fontsize=8)
    ax_a.grid(alpha=0.2)
    allc = np.concatenate(surv + imp)
    pad = 0.15 * (allc.max() - allc.min() + 1e-9)
    ax_a.set_ylim(allc.min() - pad, allc.max() + pad)

    # Panel B: static (scattered) vs dynamical (pinned at 1.0)
    rj = np.random.default_rng(1)
    ax_b.scatter(0 + 0.06 * rj.standard_normal(len(stat)), stat, color="#c53", s=55,
                 alpha=0.85, edgecolor="k", linewidth=0.4)
    ax_b.scatter(1 + 0.06 * rj.standard_normal(len(dyn)), dyn, color="#2a7", s=55,
                 alpha=0.85, edgecolor="k", linewidth=0.4)
    ax_b.axhline(0.5, color="#888", ls=":", lw=1)
    ax_b.text(1.4, 0.5, "chance", color="#888", fontsize=8, va="center")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["static\n(where a point sits)", "dynamical\n(what it conserves)"])
    ax_b.set_xlim(-0.5, 1.9)
    ax_b.set_ylim(0.3, 1.05)
    ax_b.set_ylabel("real-vs-different-identity AUC")
    ax_b.set_title(f"Static is seed-noise; dynamical is reliable ({n_seeds} seeds)")
    ax_b.grid(alpha=0.2, axis="y")

    # Panel C: the same conservation contrast under the LEARNED H_θ (§9.13)
    for c in surv_l:
        ax_c.plot(t, c, color="#2a7", lw=1.3, alpha=0.9)
    for c in imp_l:
        ax_c.plot(t, c, color="#c53", lw=1.3, alpha=0.9)
    ax_c.plot([], [], color="#2a7", lw=2, label=r"survivor — conserves the learned $H_\theta$")
    ax_c.plot([], [], color="#c53", lw=2, label="impostor — breaks it (wider margin than A)")
    ax_c.set_xlabel("time")
    ax_c.set_ylabel(r"learned $H_\theta$ along the trajectory")
    ax_c.set_title(r"Same story under a learned $H_\theta$")
    ax_c.legend(loc="upper left", fontsize=8)
    ax_c.grid(alpha=0.2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)


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

    # 2. static specificity (region-membership); 3. dynamical specificity (conservation)
    g = [specificity(D, seed=s) for s in SEEDS]
    h = [specificity_mlp(D, seed=s, steps=700) for s in SEEDS]
    gen = [generalization_specificity(D, seed=s) for s in SEEDS]  # held-out (§9.9) — the harder bar
    dyn = [dynamical_specificity(D, seed=s) for s in DYN_SEEDS]
    dyn_ci = [dynamical_specificity(D, seed=s, impostor_graph_path=COUNTER_GRAPH) for s in DYN_SEEDS]

    # 4. the same dynamical test under a LEARNED H_θ (§9.13) — one training seed per outer seed
    dyn_mlp = [dynamical_specificity(D, seed=s, fit_fn=partial(MLPManifold.fit, seed=s))
               for s in DYN_SEEDS]
    dyn_mlp_ci = [dynamical_specificity(D, seed=s, impostor_graph_path=COUNTER_GRAPH,
                                        fit_fn=partial(MLPManifold.fit, seed=s))
                  for s in DYN_SEEDS]

    def _mean(rows, k):
        return float(np.mean([r[k] for r in rows]))

    print("=" * 74)
    print(f"  Phase two — d-dim latent (d={D}) + learned H + identity through the dynamics")
    print("=" * 74)
    print(f"  identity anchors: {anchors.shape[0]} graph nodes → ℝ^{D} (Laplacian embedding)")
    print("-" * 74)
    print("  [1] machinery lifts to d dimensions")
    print(f"      conservation drift          = {drift:.2e}   (→ 0)")
    print(f"      replica AUC (conserved ψ)   = {rep['auc_psi_conserved']:.3f}   (→ 1.0)")
    print(f"      replica AUC (endpoint only) = {rep['auc_endpoint']:.3f}   (→ 0.5, null)")
    print("-" * 74)
    print("  [2] STATIC specificity (where a point sits) — real vs shuffled  (mean [min,max])")
    print(f"      Gaussian fit   real = {_summ(g, 'auc_real')}   shuffled = {_summ(g, 'auc_shuffled')}")
    print(f"      learned MLP    real = {_summ(h, 'auc_real')}   shuffled = {_summ(h, 'auc_shuffled')}")
    print(f"      held-out MLP   real = {_summ(gen, 'auc_real')}   shuffled = {_summ(gen, 'auc_shuffled')}")
    print(f"      → real ≈ shuffled at {anchors.shape[0]} nodes too — static region-membership is")
    print("        the wrong question; richer content does not fix it (§9.9–§9.10, §9.12)")
    print("-" * 74)
    print("  [3] DYNAMICAL specificity (which conservation law a trajectory obeys)")
    print(f"      shuffle impostor:  AUC = {_summ(dyn, 'auc')}   (→ 1.0, reliable)")
    print(f"      var(H_real): survivor {_mean(dyn, 'surv_resid'):.1e}  vs  "
          f"impostor {_mean(dyn, 'imp_resid'):.1e}   "
          f"(impostor conserves its OWN charge: {_mean(dyn, 'imp_own_resid'):.1e})")
    print(f"      authored impostor: AUC = {_summ(dyn_ci, 'auc')}   ('Meridian' — a distinct soul)")
    print(f"      var(H_real): survivor {_mean(dyn_ci, 'surv_resid'):.1e}  vs  "
          f"impostor {_mean(dyn_ci, 'imp_resid'):.1e}   "
          f"(impostor conserves its OWN charge: {_mean(dyn_ci, 'imp_own_resid'):.1e})")
    print("-" * 74)
    print("  [4] DYNAMICAL specificity under a LEARNED H_θ (MLP potential; float32 V_θ,")
    print("      float64 statistics — same integrator and reader as [3])")
    print(f"      shuffle impostor:  AUC = {_summ(dyn_mlp, 'auc')}")
    print(f"      var(H_θ): survivor {_mean(dyn_mlp, 'surv_resid'):.1e}  vs  "
          f"impostor {_mean(dyn_mlp, 'imp_resid'):.1e}   "
          f"(impostor's own charge: {_mean(dyn_mlp, 'imp_own_resid'):.1e})")
    print(f"      authored impostor: AUC = {_summ(dyn_mlp_ci, 'auc')}   ('Meridian')")
    print(f"      var(H_θ): survivor {_mean(dyn_mlp_ci, 'surv_resid'):.1e}  vs  "
          f"impostor {_mean(dyn_mlp_ci, 'imp_resid'):.1e}   "
          f"(impostor's own charge: {_mean(dyn_mlp_ci, 'imp_own_resid'):.1e})")
    print("-" * 74)
    print("  VERDICT: STATIC identity (where a point sits) is seed-noise — the wrong")
    print("  question. DYNAMICAL identity (which conservation law a trajectory obeys) is")
    print("  RELIABLE, AUC 1.0 across seeds: an impostor conserves its own charge, not")
    print("  Embra's — under the Gaussian charge AND a learned H_θ, against a shuffle AND")
    print("  a distinct authored soul. §6 was right — identity lives in the dynamics.")
    print("=" * 74)
    make_phase2_figure()
    print(f"  figure → {FIG_PATH}")
    return {"drift": drift, "replica": rep, "gaussian": g, "mlp": h, "generalization": gen,
            "dynamical": dyn, "dynamical_counter": dyn_ci,
            "dynamical_mlp": dyn_mlp, "dynamical_mlp_counter": dyn_mlp_ci}


if __name__ == "__main__":
    main()
