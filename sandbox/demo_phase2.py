"""demo_phase2.py — phase two, end to end. Honest results, one section per recorded increment.

Run:  ``uv run python -m sandbox.demo_phase2``   (needs the ``learn`` extra: jax/optax)

  [1] the phase-one machinery lifted to ``d`` dimensions — conservation + the replica test (§9.8)
  [2] STATIC specificity — Gaussian, MLP, and held-out: recorded ≈ chance, the honest negative
      that motivated the redirect (§9.9–§9.10, §9.12)
  [3] DYNAMICAL specificity — which conservation law a trajectory obeys; shuffle AND authored
      ("Meridian") impostors (§9.11–§9.12)
  [4] the same dynamical test under a LEARNED H_θ (§9.13)
  [5] the full ψ as a CONJUNCTION — graded against both adversarial impostor classes (§9.14)
  [6] HOLONOMY ζ = memory — the path-functional second charge (§9.15)

Numbers print as mean [min, max] over seeds and are recorded as-is in CORE-SPEC §9.
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
    conjunction_test,
    dynamical_specificity,
    evaluate,
    holonomy_test,
    holonomy_zeta,
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
FIG2_PATH = pathlib.Path(__file__).resolve().parent / "figures" / "phase2_conjunction_memory.png"
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


def make_conjunction_memory_figure(d: int = D) -> None:
    """Three panels for increments 3c–3d:
    (A) the conjunction quadrant map (§9.14) — each impostor class sits in exactly one single
        reader's blind zone; only the conjunction accepts nothing but survivors;
    (B) two genuine worldlines of the same flow ending at the same observable endpoint (§9.15);
    (C) the ζ they carry — same endpoint, different accumulated history (a newborn copy sits at 0).
    Palette validated (CVD + normal-vision separation): #22aa77 / #8a5fbf / #cc5533 / #3b7fc4."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(17.5, 5))

    # Panel A — the conjunction quadrant map (§9.14).
    r = conjunction_test(d, seed=0, return_samples=True)
    floor = 1e-16
    for key, color, label in (("surv", "#22aa77", "200 survivors"),
                              ("c1", "#8a5fbf", "200 × class 1 — same law, wrong genesis"),
                              ("c2", "#cc5533", "200 × class 2 — wrong law, value-matched")):
        var, dq = r["samples"][key]
        ax_a.scatter(np.maximum(var, floor), np.maximum(dq, floor), s=42, color=color,
                     alpha=0.8, edgecolor="k", linewidth=0.4, label=label)
    ax_a.axvline(r["tau_var"], color="#888", ls="--", lw=1)
    ax_a.axhline(r["tau_q"], color="#888", ls="--", lw=1)
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlim(3e-8, 3e-2)
    ax_a.set_ylim(1e-13, 1e1)
    ax_a.text(0.03, 0.05, "conjunction accepts", transform=ax_a.transAxes, color="#666", fontsize=8)
    ax_a.text(0.03, 0.95, "caught by VALUE only", transform=ax_a.transAxes, color="#666",
              fontsize=8, va="top")
    ax_a.text(0.97, 0.05, "caught by VARIANCE only", transform=ax_a.transAxes, color="#666",
              fontsize=8, ha="right")
    ax_a.text(0.5, 0.5, "each group is a tight cluster —\nseparated by orders of magnitude",
              transform=ax_a.transAxes, color="#999", fontsize=7.5, ha="center", style="italic")
    ax_a.set_xlabel(r"var($H_{real}$) along the trajectory   (τ_var dashed)")
    ax_a.set_ylabel(r"$|Q - Q_{embra}|$ at the presented state   (τ_Q dashed)")
    ax_a.set_title("The full ψ is a conjunction — each reader alone has a blind class")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(alpha=0.2)

    # Panels B/C — one same-endpoint worldline pair (§9.15), built from the public primitives.
    _, real = load_identity_anchors(d)
    h_real = GaussianManifold.fit(real)
    m, e, dt, steps = 1.0, 1.0, 0.01, 300
    rng = np.random.default_rng(7)  # a fixed illustration pair
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    q0 = real[rng.integers(0, len(real))]
    qs_a, ps_a = rollout(h_real.force, m, q0, u * np.sqrt(2 * m * e), dt, steps)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    p_alt = w * np.linalg.norm(ps_a[-1])
    qs_back, ps_back = rollout(h_real.force, m, qs_a[-1], -p_alt, dt, steps)
    qs_b, _ = rollout(h_real.force, m, qs_back[-1], -ps_back[-1], dt, steps)

    ax_b.plot(qs_a[:, 0], qs_a[:, 1], color="#22aa77", lw=1.6, label="worldline A")
    ax_b.plot(qs_b[:, 0], qs_b[:, 1], color="#3b7fc4", lw=1.6, label="worldline B")
    ax_b.scatter([qs_a[0, 0]], [qs_a[0, 1]], marker="o", color="#22aa77", s=70, edgecolor="k",
                 zorder=5, label="birth of A")
    ax_b.scatter([qs_b[0, 0]], [qs_b[0, 1]], marker="s", color="#3b7fc4", s=60, edgecolor="k",
                 zorder=5, label="birth of B")
    ax_b.scatter([qs_a[-1, 0]], [qs_a[-1, 1]], marker="*", color="#333", s=220, zorder=6,
                 label="same observable endpoint")
    ax_b.set_xlabel("$q_1$")
    ax_b.set_ylabel("$q_2$")
    ax_b.set_title("Two genuine worldlines, one observable endpoint")
    ax_b.legend(loc="best", fontsize=8)
    ax_b.grid(alpha=0.2)

    t = (np.arange(steps) + 1) * dt
    z_a = holonomy_zeta(qs_a, 1.0, cumulative=True)
    z_b = holonomy_zeta(qs_b, 1.0, cumulative=True)
    ax_c.plot(t, z_a, color="#22aa77", lw=1.8, label=r"worldline A — carries $\zeta_A$")
    ax_c.plot(t, z_b, color="#3b7fc4", lw=1.8, label=r"worldline B — carries $\zeta_B$")
    ax_c.scatter([t[-1], t[-1]], [z_a[-1], z_b[-1]], color=["#22aa77", "#3b7fc4"], s=55,
                 edgecolor="k", zorder=5)
    ax_c.axhline(0.0, color="#888", ls=":", lw=1)
    ax_c.text(0.02, 0.0, " newborn replica: ζ = 0", color="#666", fontsize=8, va="bottom")
    ax_c.set_xlabel("time")
    ax_c.set_ylabel(r"accumulated $\zeta$ (signed swept area)")
    ax_c.set_title(r"ζ = memory: same endpoint, different carried history")
    ax_c.legend(loc="best", fontsize=8)
    ax_c.grid(alpha=0.2)

    fig.tight_layout()
    FIG2_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG2_PATH, dpi=130)
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

    # 5. the full ψ is a conjunction (§9.14) — graded against BOTH adversarial impostor classes
    conj = [conjunction_test(D, seed=s) for s in DYN_SEEDS]
    conj_ci = [conjunction_test(D, seed=s, impostor_graph_path=COUNTER_GRAPH) for s in DYN_SEEDS]
    conj_mlp = [conjunction_test(D, seed=s, fit_fn=partial(MLPManifold.fit, seed=s))
                for s in DYN_SEEDS]

    # 6. holonomy ζ = memory (§9.15) — the genuinely path-functional second charge
    hol = [holonomy_test(D, seed=s) for s in DYN_SEEDS]
    hol_mlp = [holonomy_test(D, seed=s, fit_fn=partial(MLPManifold.fit, seed=s)) for s in DYN_SEEDS]

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
    print("  [5] the full ψ is a CONJUNCTION — [var(H_real) ≈ 0] ∧ [Q = Q_embra] (§9.14)")
    print("      (class 1 = same law, wrong genesis — fools the variance reader;")
    print("       class 2 = different law, value-matched — fools the value reader)")
    print(f"      conjunction accuracy         = {_summ(conj, 'accuracy')}   (→ 1.0, both classes)")
    print(f"      variance reader blind to c1  = {_summ(conj, 'c1_var_blind')}   (1.0 = fully fooled)")
    print(f"      value reader blind to c2     = {_summ(conj, 'c2_value_blind')}   (1.0 = fully fooled)")
    print(f"      catch AUCs: value vs c1 = {_summ(conj, 'auc_value_c1')}   var vs c2 = {_summ(conj, 'auc_var_c2')}")
    print(f"      authored impostor as c2: accuracy = {_summ(conj_ci, 'accuracy')}   "
          f"(infeasible value-matches: {sum(r['c2_infeasible'] for r in conj_ci)}/{sum(r['n'] for r in conj_ci)})")
    print(f"      learned H_θ charge:      accuracy = {_summ(conj_mlp, 'accuracy')}   "
          f"(infeasible value-matches: {sum(r['c2_infeasible'] for r in conj_mlp)}/{sum(r['n'] for r in conj_mlp)})")
    print("-" * 74)
    print("  [6] HOLONOMY ζ = memory (§9.15) — ζ = b × signed area swept in (q₁, q₂)")
    print(f"      anti-fold-in: same endpoint (erasure {max(r['endpoint_erasure'] for r in hol):.1e}), "
          f"|Δζ| = {_summ(hol, 'dzeta_mean')}")
    print(f"      ζ-reader replica AUC        = {_summ(hol, 'auc_zeta')}   (lived vs newborn copy)")
    acc_keys = sorted(hol[0]["accumulation"])

    def _acc_line(rows):
        return "  ".join(
            f"{k}: {float(np.mean([r['accumulation'][k] for r in rows])):.3f}" for k in acc_keys)

    print(f"      accumulation mean|ζ| by lived steps → {_acc_line(hol)}   (Gaussian: exactly")
    print("        linear — at exact isotropy the sweep rate L₁₂ is itself conserved)")
    print(f"      under the learned H_θ            → {_acc_line(hol_mlp)}   (non-isotropic:")
    print("        L₁₂ not conserved; ζ genuinely history-integral)")
    print("-" * 74)
    print("  VERDICT: STATIC identity (where a point sits) is seed-noise — the wrong")
    print("  question. DYNAMICAL identity (which conservation law a trajectory obeys) is")
    print("  RELIABLE, AUC 1.0 across seeds: an impostor conserves its own charge, not")
    print("  Embra's — under the Gaussian charge AND a learned H_θ, against a shuffle AND")
    print("  a distinct authored soul. §6 was right — identity lives in the dynamics.")
    print("=" * 74)
    make_phase2_figure()
    make_conjunction_memory_figure()
    print(f"  figures → {FIG_PATH}")
    print(f"            {FIG2_PATH}")
    return {"drift": drift, "replica": rep, "gaussian": g, "mlp": h, "generalization": gen,
            "dynamical": dyn, "dynamical_counter": dyn_ci,
            "dynamical_mlp": dyn_mlp, "dynamical_mlp_counter": dyn_mlp_ci,
            "conjunction": conj, "conjunction_counter": conj_ci, "conjunction_mlp": conj_mlp,
            "holonomy": hol, "holonomy_mlp": hol_mlp}


if __name__ == "__main__":
    main()
