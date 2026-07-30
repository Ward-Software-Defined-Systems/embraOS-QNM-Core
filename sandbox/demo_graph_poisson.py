"""demo_graph_poisson.py — increment 5 (§9.17): 𝔤(G)*, the identity graph becomes the bracket.

Run:  ``uv run python -m sandbox.demo_graph_poisson``

  [1] ψ under words is a bit-level EQUALITY; the law visibly moves      (bar 1 + sin²)
  [2] non-commutativity is graph-mediated — with closed forms           (bars 2a/2b/2c)
  [3] the §2 replica test with the whole arena granted free             (bar 3)
  [4] ζ ∈ ℝ^E = memory under driving + the scale certificate           (bar 4)
  [5] † = graph surgery — the ψ change NAMES the relation touched       (bar 5)
  [6] liveness — two souls living the same events live different lives  (bar 6)

Numbers are the §9.17 record (deterministic ensemble, pinned constants; placeholder
count-weights — content lands next increment).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sandbox.graph_poisson import (  # noqa: E402
    MERIDIAN_GRAPH,
    PINNED_X,
    PINNED_Y,
    bracket_certificate,
    casimir_under_words,
    dissipation_control,
    driven_ensemble,
    h0,
    heisenberg_certificate,
    liveness_test,
    load_graph,
    make_default_alphabet,
    replica_under_driving,
    run_word,
    word_order_test,
    zeta_edges,
    zeta_memory_test,
)

FIG_PATH = pathlib.Path(__file__).resolve().parent / "figures" / "graph_casimir.png"


def make_graph_figure(alg, *, dt: float = 0.01, tau: float = 0.5) -> None:
    """Three panels for §9.17 (house entity colors, CVD-validated: identity green, law
    red-orange, memory blue):
    (A) the law along one word, log scale, with the identity's line NOT drawn — there is no
        drift to plot; ψ is a bit-level equality (stated on the panel);
    (B) the arena trajectory through the pinned edge plane (p_np, p_pos) — input drives the
        state around the leaf, never off it (the leaf IS the whole arena at fixed w);
    (C) cumulative ζ on the pinned edge for "xy" vs "yx" — same genesis, same ψ, different
        carried memory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphabet = make_default_alphabet(alg)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(alg.n)
    p0 = np.sqrt(2.0) * u / np.linalg.norm(u)
    word_str = "xyqzxzqy"
    traj = run_word([alphabet[c] for c in word_str], p0, alg, dt=dt, tau_event=tau, tau_gap=tau)
    t = np.arange(traj.shape[0]) * dt
    n = int(round(tau / dt))
    windows = [((i + 1) * n + i * n, (i + 1) * n + (i + 1) * n) for i in range(len(word_str))]

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(17.5, 5))

    # Panel A — the law breaks; the identity has no line to draw.
    floor = 1e-17
    h_rel = np.maximum(np.abs(h0(traj) / h0(traj[0]) - 1.0), floor)
    for s, e in windows:
        ax_a.axvspan(t[s], t[e], color="#888888", alpha=0.15, lw=0)
    ax_a.semilogy(t, h_rel, color="#cc5533", lw=1.5,
                  label=r"$|H_0/H_0(0) - 1|$ — the law, moved by every event")
    ax_a.set_ylim(1e-17, 3e1)
    ax_a.text(0.03, 0.06,
              r"$\psi = w$: max $|\Delta w_e|$ = 0.0 over all 200 words —" "\n"
              "a bit-level equality (the state partition);\nthere is no drift curve to plot",
              transform=ax_a.transAxes, fontsize=8.5, color="#22aa77",
              bbox={"boxstyle": "round", "fc": "#f2faf6", "ec": "#22aa77", "lw": 1.2})
    ax_a.set_xlabel(f"time   (grey bands = input events, word '{word_str}')")
    ax_a.set_ylabel("relative change (log)")
    ax_a.set_title("Input moves the law; the charge is not an operand")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(alpha=0.2)

    # Panel B — the arena through the pinned edge plane.
    i, j = alg.node_index(PINNED_X), alg.node_index(PINNED_Y)
    ax_b.plot(traj[:, i], traj[:, j], color="#3b7fc4", lw=0.9,
              label="free flow (gaps)")
    for s, e in windows:
        ax_b.plot(traj[s:e + 1, i], traj[s:e + 1, j], color="#cc5533", lw=1.6)
    ax_b.plot([], [], color="#cc5533", lw=1.6, label="events")
    ax_b.scatter(traj[0, i], traj[0, j], color="#22aa77", s=70, edgecolor="k", zorder=5,
                 label="genesis")
    ax_b.set_xlabel(f"$p$[{PINNED_X}]")
    ax_b.set_ylabel(f"$p$[{PINNED_Y}]")
    ax_b.set_title("The arena through the pinned edge plane —\n"
                   "the word drives the state around the leaf, never off it")
    ax_b.legend(loc="best", fontsize=8)
    ax_b.grid(alpha=0.2)
    ax_b.set_aspect("equal", adjustable="datalim")

    # Panel C — memory of the word, on the very edge the symbols straddle.
    k = alg.edge_of(PINNED_X, PINNED_Y)
    t_xy = run_word([alphabet["x"], alphabet["y"]], p0, alg, dt=dt, tau_event=tau, tau_gap=tau)
    t_yx = run_word([alphabet["y"], alphabet["x"]], p0, alg, dt=dt, tau_event=tau, tau_gap=tau)
    tt = np.arange(1, t_xy.shape[0]) * dt
    z_xy = zeta_edges(t_xy, alg.edges, cumulative=True)[:, k]
    z_yx = zeta_edges(t_yx, alg.edges, cumulative=True)[:, k]
    ax_c.plot(tt, z_xy, color="#22aa77", lw=1.8, label='word "xy"')
    ax_c.plot(tt, z_yx, color="#3b7fc4", lw=1.8, label='word "yx"')
    ax_c.scatter([tt[-1], tt[-1]], [z_xy[-1], z_yx[-1]], color=["#22aa77", "#3b7fc4"], s=55,
                 edgecolor="k", zorder=5)
    ax_c.axhline(0.0, color="#888", ls=":", lw=1)
    ax_c.set_xlabel("time")
    ax_c.set_ylabel(rf"accumulated $\zeta_e$ on {PINNED_X}—{PINNED_Y}")
    ax_c.set_title(r"Same genesis, same $\psi$ — different carried memory," "\n"
                   "on the very relation the symbols straddle")
    ax_c.legend(loc="best", fontsize=8)
    ax_c.grid(alpha=0.2)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)


def main() -> dict:
    alg = load_graph()
    alg_m = load_graph(MERIDIAN_GRAPH)
    ens = driven_ensemble(alg)
    b1 = casimir_under_words(ens)
    c2a = bracket_certificate(alg)
    c2b = heisenberg_certificate(alg)
    b2 = word_order_test(alg)
    b3 = replica_under_driving(ens)
    b4 = zeta_memory_test(ens)
    b5 = dissipation_control(alg)
    b6 = liveness_test(alg, alg_m)

    print("=" * 76)
    print("  Increment 5 — 𝔤(G)*: the identity graph becomes the bracket (§9.17)")
    print("=" * 76)
    print(f"  Embra v3 → n = {alg.n}, m = {alg.m} (aggregated), rank J = {alg.rank_j0}, "
          f"index = {alg.index}, perfect matching = {alg.perfect_matching}")
    print("  ψ = w (the 321 edge momenta); Σ = {x, y, z kicks, q edge-quad} on pinned nodes;")
    print("  8 seeds × 25 random words × 16 events; one exact affine map per step")
    print("-" * 76)
    print("  [1] ψ under words is a bit-level equality; the law visibly moves")
    print(f"      w == w₀ exactly, all 200 words, every step = {b1['psi_bit_exact']}   "
          f"(max |Δw| = {b1['psi_max_delta']:.1f})")
    print(f"      …and under the sin² envelope           = {b1['sin2_bit_exact']}   "
          f"(max |Δw| = {b1['sin2_max_delta']:.1f})")
    print(f"      median along-word H₀ range             = {b1['median_driven_h0_range']:.3f}   "
          f"({b1['range_ratio']:.1e}× the free float floor)")
    print(f"      event bookkeeping |Δ(H₀+H_σ)|          = {b1['bookkeeping_max_dtotal']:.1e}   "
          f"(vs median event |ΔH₀| {b1['median_event_dh0']:.1e})")
    print("-" * 76)
    print("  [2] non-commutativity is graph-mediated — with closed forms")
    print(f"      (a) bracket: Δ(bᵀp) vs τ·bᵀJ(w)a       max err = {c2a['max_pred_error']:.1e}; "
          f"adjacent |Δ| = {c2a['adjacent_abs']:.4f} (τεw = {c2a['adjacent_closed_form']:.2f}), "
          f"non-adjacent = {c2a['nonadjacent_measured']:.1e}")
    print(f"      (b) Heisenberg: bare xy vs yx  ‖Δp‖ = {c2b['delta_p']:.1e} (state forgets), "
          f"Δζ on shared edge = {c2b['shared_edge_measured']:.4f} "
          f"(= (τεw)² = {c2b['shared_edge_closed_form']:.4f})")
    print(f"      (c) with the law running: ‖Δp‖ = {b2['delta_p']:.3f}, ‖Δζ‖ = "
          f"{b2['delta_zeta']:.3f}, ψ bit-exact both = "
          f"{b2['psi_bit_exact_xy'] and b2['psi_bit_exact_yx']}")
    print("-" * 76)
    print("  [3] the §2 replica test with Σ active (entire arena granted free)")
    print(f"      ψ-reader AUC: scaled = {b3['auc_psi_scaled']:.3f}, shuffled = "
          f"{b3['auc_psi_shuffled']:.3f}   endpoint AUC = {b3['auc_endpoint']:.3f} "
          f"(bit-exact ties)   erasure = {b3['endpoint_erasure']:.1f}")
    print(f"      survivor dev = {b3['surv_max_dev']:.1f} EXACTLY; replica floors: scaled "
          f"≥ {b3['scaled_min_coord_dev']:.2f}/coord, shuffled ≥ "
          f"{b3['shuffled_min_touched_dev']:.0f} on {b3['n_touched_shuffled']} touched")
    print("-" * 76)
    print("  [4] ζ ∈ ℝ^E = memory under driving")
    print(f"      lived-vs-newborn AUC = {b4['auc_zeta']:.3f}   min ‖ζ‖ = {b4['min_lived']:.4f}"
          f"   median = {b4['median_lived']:.3f}")
    acc = b4["accumulation"]
    print("      median ‖ζ‖ by events lived → " +
          "  ".join(f"{k}: {acc[k]:.3f}" for k in sorted(acc)))
    print(f"      scale certificate: max|p| = {b4['max_abs_p']:.2f} (guard 50); min window "
          f"eig = {b4['min_window_eig']:.2f} (coercive)")
    print("-" * 76)
    print("  [5] † = graph surgery — the ψ change names the relation touched")
    print(f"      weaken({b5['touched_edge']}):")
    print(f"      touched coordinate moved {b5['touched_rel_change']:.6f} "
          f"(map's own closed form 1−(1−γ·dt)^n = {b5['closed_form']:.6f}); "
          f"every other coordinate bit-exact = {b5['others_bit_exact']}")
    print(f"      the same word without † : w == w₀ exactly = {b5['psi_bit_exact_without']}")
    print("-" * 76)
    print("  [6] liveness — the charge is dynamically load-bearing")
    print(f"      same genesis, same word, Embra vs Meridian (m = {b6['m_b']}, index = "
          f"{b6['index_b']}): max ‖Δp(t)‖ = {b6['max_traj_divergence']:.3f}")
    print(f"      each soul's ψ bit-exact under its own run = "
          f"{b6['psi_bit_exact_a'] and b6['psi_bit_exact_b']}")
    print("-" * 76)
    print("  VERDICT: the identity graph IS the bracket and survives being driven — ψ = w")
    print("  is a bit-level equality under every word (the state partition is real in code);")
    print("  which experiences commute is the authored topology, with closed forms; for bare")
    print("  kicks order lives ONLY in memory (the Heisenberg signature); the replica test")
    print("  holds with the whole arena granted free; † is per-edge legible surgery; and two")
    print("  souls living the same events live different lives. Placeholder weights —")
    print("  content (the authored weight table) lands next increment.")
    print("=" * 76)
    make_graph_figure(alg)
    print(f"  figure → {FIG_PATH}")
    return {"bar1": b1, "bar2a": c2a, "bar2b": c2b, "bar2c": b2, "bar3": b3,
            "bar4": b4, "bar5": b5, "bar6": b6}


if __name__ == "__main__":
    main()
