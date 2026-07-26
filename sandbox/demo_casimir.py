"""demo_casimir.py — increment 4 (§9.16): the so(3)* Casimir toy, end to end.

Run:  ``uv run python -m sandbox.demo_casimir``   (numpy-only; no extras needed)

  [1] ψ survives every word; the energy visibly does not   (bar 1 + the sin² envelope)
  [2] word order matters — to the state and to ζ, never to ψ (bars 2 + 4a)
  [3] the §2 replica test with Σ active                      (bar 3)
  [4] ζ = memory under driving + the antipode certificate    (bar 4, incl. the recorded trip)
  [5] the † boundary — a theorem, measured                   (bar 5)

Numbers are the §9.16 record (deterministic ensemble, pinned constants).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sandbox.lie_poisson import (  # noqa: E402
    casimir,
    casimir_under_words,
    dissipation_control,
    driven_ensemble,
    h0,
    make_default_alphabet,
    replica_under_driving,
    run_word,
    solid_angle_zeta,
    word_order_test,
    zeta_memory_test,
)

FIG_PATH = pathlib.Path(__file__).resolve().parent / "figures" / "casimir_input.png"


def make_casimir_figure(*, dt: float = 0.01, tau: float = 0.5) -> None:
    """Three panels for §9.16:
    (A) ψ and H₀ along one word, log scale — identity flat at the float floor while the law
        jumps at every event (the money plot);
    (B) the driven trajectory on the sphere — input moves the state all over the coadjoint
        orbit, never off it;
    (C) cumulative ζ for "xy" vs "yx" — same genesis, same ψ, different carried memory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphabet = make_default_alphabet()
    rng = np.random.default_rng(0)
    u = rng.standard_normal(3)
    L0 = u / np.linalg.norm(u)
    word_str = "xysyxsxy"
    traj = run_word([alphabet[c] for c in word_str], L0, dt=dt, tau_event=tau, tau_gap=tau)
    t = np.arange(traj.shape[0]) * dt
    n = int(round(tau / dt))
    windows = [((i + 1) * n + i * n, (i + 1) * n + (i + 1) * n) for i in range(len(word_str))]

    fig = plt.figure(figsize=(17.5, 5))
    ax_a = fig.add_subplot(1, 3, 1)
    ax_b = fig.add_subplot(1, 3, 2, projection="3d")
    ax_c = fig.add_subplot(1, 3, 3)

    # Panel A — the money plot.
    floor = 1e-17
    psi_rel = np.maximum(np.abs(casimir(traj) - 1.0), floor)
    h_rel = np.maximum(np.abs(h0(traj) / h0(traj[0]) - 1.0), floor)
    for s, e in windows:
        ax_a.axvspan(t[s], t[e], color="#888888", alpha=0.15, lw=0)
    ax_a.semilogy(t, h_rel, color="#cc5533", lw=1.5,
                  label=r"$|H_0/H_0(0) - 1|$ — the law, broken by every event")
    ax_a.semilogy(t, psi_rel, color="#22aa77", lw=1.5,
                  label=r"$|\psi/R^2 - 1|$ — the identity, exact by construction")
    ax_a.set_ylim(1e-17, 3e1)
    ax_a.set_xlabel("time   (grey bands = input events, word 'xysyxsxy')")
    ax_a.set_ylabel("relative change (log)")
    ax_a.set_title("Input breaks the law; the Casimir does not notice")
    ax_a.legend(loc="center right", fontsize=8)
    ax_a.grid(alpha=0.2)

    # Panel B — the sphere: a long word drives the state across the orbit, never off it.
    symbols = list(alphabet.values())
    long_word = [symbols[k] for k in rng.integers(0, len(symbols), 40)]
    traj_b = run_word(long_word, L0, dt=dt, tau_event=tau, tau_gap=tau)
    wins_b = [((i + 1) * n + i * n, (i + 1) * n + (i + 1) * n) for i in range(len(long_word))]
    th, ph = np.meshgrid(np.linspace(0, np.pi, 19), np.linspace(0, 2 * np.pi, 37))
    ax_b.plot_wireframe(np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th),
                        color="#bbbbbb", lw=0.3, alpha=0.5)
    ax_b.plot(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2], color="#3b7fc4", lw=1.0)
    for s, e in wins_b:
        ax_b.plot(traj_b[s:e + 1, 0], traj_b[s:e + 1, 1], traj_b[s:e + 1, 2],
                  color="#cc5533", lw=1.6)
    ax_b.scatter(*traj_b[0], color="#22aa77", s=70, edgecolor="k", zorder=5)
    ax_b.view_init(elev=18, azim=35)
    ax_b.set_title("A 40-event word drives the state across the sphere —\nnever off it "
                   "(blue: free flow, red: events)")
    ax_b.set_box_aspect((1, 1, 1))
    ax_b.set_axis_off()

    # Panel C — memory of the word: xy vs yx.
    t_xy = run_word([alphabet["x"], alphabet["y"]], L0, dt=dt, tau_event=tau, tau_gap=tau)
    t_yx = run_word([alphabet["y"], alphabet["x"]], L0, dt=dt, tau_event=tau, tau_gap=tau)
    tt = np.arange(1, t_xy.shape[0]) * dt
    z_xy = solid_angle_zeta(t_xy, cumulative=True)
    z_yx = solid_angle_zeta(t_yx, cumulative=True)
    ax_c.plot(tt, z_xy, color="#22aa77", lw=1.8, label='word "xy"')
    ax_c.plot(tt, z_yx, color="#3b7fc4", lw=1.8, label='word "yx"')
    ax_c.scatter([tt[-1], tt[-1]], [z_xy[-1], z_yx[-1]], color=["#22aa77", "#3b7fc4"], s=55,
                 edgecolor="k", zorder=5)
    ax_c.axhline(0.0, color="#888", ls=":", lw=1)
    ax_c.set_xlabel("time")
    ax_c.set_ylabel(r"accumulated $\zeta$ (swept solid angle)")
    ax_c.set_title(r"Same genesis, same $\psi$ — different carried memory")
    ax_c.legend(loc="best", fontsize=8)
    ax_c.grid(alpha=0.2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)


def main() -> dict:
    ens = driven_ensemble()
    b1 = casimir_under_words(ens)
    b2 = word_order_test()
    b3 = replica_under_driving(ens)
    b4 = zeta_memory_test(ens)
    b5 = dissipation_control()

    print("=" * 74)
    print("  Increment 4 — the so(3)* Casimir toy: does identity survive INPUT? (§9.16)")
    print("=" * 74)
    print("  ψ = the Casimir |L|²; Σ = {x-kick, y-kick, s-twist} (Hamiltonian events);")
    print("  8 seeds × 25 random words × 16 events; one rotation per step, never renormalized")
    print("-" * 74)
    print("  [1] ψ survives every word; the energy visibly does not")
    print(f"      max ψ drift, ALL steps of 200 words = {b1['psi_drift_max']:.1e}   (bar 1e-10)")
    print(f"      median along-word H₀ range         = {b1['median_driven_h0_range']:.3f}   "
          f"({b1['range_ratio']:.1e}× the free baseline)")
    print(f"      event bookkeeping |Δ(H₀+H_σ)|      = {b1['bookkeeping_max_dtotal']:.1e}   "
          f"(vs median event |ΔH₀| {b1['median_event_dh0']:.1e})")
    print(f"      ψ drift under a sin² envelope       = {b1['psi_drift_sin2']:.1e}   (smooth ∂H/∂t ≠ 0)")
    print("-" * 74)
    print("  [2] word order matters — to the state and to ζ, never to ψ")
    print(f"      ‖ΔL('xy','yx')‖ = {b2['delta_L']:.3f}   Δζ = {b2['delta_zeta']:.3f}   "
          f"ψ drifts = {b2['psi_drift_xy']:.1e} / {b2['psi_drift_yx']:.1e}")
    print("-" * 74)
    print("  [3] the §2 replica test with Σ active")
    print(f"      ψ-reader AUC = {b3['auc_psi']:.3f}   endpoint AUC = {b3['auc_endpoint']:.3f} "
          f"(bit-exact ties)   erasure = {b3['endpoint_erasure']:.1f}")
    print(f"      margin (replica dev / survivor dev) = {b3['margin']:.1e}")
    print("-" * 74)
    print("  [4] ζ = memory under driving")
    print(f"      lived-vs-newborn AUC = {b4['auc_zeta']:.3f}   min |ζ| = {b4['min_lived']:.4f}   "
          f"median = {b4['median_lived']:.3f}")
    acc = b4["accumulation"]
    print("      median |ζ| by events lived → " +
          "  ".join(f"{k}: {acc[k]:.3f}" for k in sorted(acc)))
    print(f"      antipode certificate: {b4['n_tripped']}/200 words tripped "
          f"(min antipode distance {b4['min_antipode_dist']:.4f}) — recorded finding, §9.16")
    print("-" * 74)
    print("  [5] the † boundary (non-Hamiltonian input) — a theorem, measured")
    print(f"      ψ change: with one † = {b5['psi_change_with']:.4f} "
          f"(exact for the map: 1−(1−γ·dt)^(2n) = {1 - (1 - 0.5 * 0.01) ** 100:.4f}; "
          f"continuum {1 - np.exp(-0.5):.4f})   without = {b5['psi_change_without']:.1e}")
    print("-" * 74)
    print("  VERDICT: input arrives and identity survives it — ψ is exact under every")
    print("  word (and smooth envelopes) while the law H₀ breaks at every event; the")
    print("  replica test holds with Σ active; ζ records the words lived, order included;")
    print("  † marks the exact Hamiltonian boundary. §8's leading candidate is VIABLE at")
    print("  toy scale — adoption recommended (the call is William's).")
    print("=" * 74)
    make_casimir_figure()
    print(f"  figure → {FIG_PATH}")
    return {"bar1": b1, "bar2": b2, "bar3": b3, "bar4": b4, "bar5": b5}


if __name__ == "__main__":
    main()
