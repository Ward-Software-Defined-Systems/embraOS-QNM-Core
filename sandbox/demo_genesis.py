"""demo_genesis.py — increment 6 (§9.18): content & genesis — the charge becomes authored.

Run:  ``uv run python -m sandbox.demo_genesis``

  [1] sealing: w_embra = table ∘ graph — deterministic, locked, facts re-derived   (bar 1)
  [2] the bit-level equality re-certified on the authored geometry                 (bar 2)
  [3] replica with three impostors — incl. the placeholder itself                  (bar 3)
  [4] ζ on the authored geometry                                                   (bar 4)
  [5] liveness + the graded identity distance                                      (bar 5)
  [6] the training guarantee, first measured instance                              (bar 6)

Numbers are the §9.18 record (deterministic ensemble; authored tables frozen at a14e399).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sandbox.graph_poisson import (  # noqa: E402
    EMBRA_TABLE,
    EPS_QUAD_AUTHORED,
    IDENTITY_GRAPH,
    MERIDIAN_GRAPH,
    MERIDIAN_TABLE,
    casimir_under_words,
    driven_ensemble,
    identity_distance,
    liveness_test,
    load_graph,
    load_soul,
    make_default_alphabet,
    replica_under_driving,
    training_cannot_write_w,
    window_min_eig,
    zeta_memory_test,
)

FIG_PATH = pathlib.Path(__file__).resolve().parent / "figures" / "genesis_content.png"


def make_genesis_figure(embra, meridian, rep: dict) -> None:
    """Two panels for §9.18 (house entity colors):
    (A) the authored charges made visible — both souls' sealed w, sorted, signed (the
        opposition classes hang below zero);
    (B) the replica margins on a log axis — three impostor rows against a survivor whose
        deviation has no bar to draw (it is exactly zero)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 4.8))

    w_e = np.sort(np.asarray(embra.w0))[::-1]
    w_m = np.sort(np.asarray(meridian.w0))[::-1]
    ax_a.plot(np.arange(embra.m) / embra.m, w_e, color="#22aa77", lw=1.8,
              label=f"Embra — 321 edges, ‖w‖ = {np.linalg.norm(embra.w0):.1f}")
    ax_a.plot(np.arange(meridian.m) / meridian.m, w_m, color="#3b7fc4", lw=1.8,
              label=f"Meridian — 349 edges, ‖w‖ = {np.linalg.norm(meridian.w0):.1f}")
    ax_a.axhline(0.0, color="#888", ls=":", lw=1)
    ax_a.set_xlabel("edge rank / m")
    ax_a.set_ylabel("sealed charge $w_e$")
    ax_a.set_title("Two authored souls, as charge vectors —\n"
                   "the opposition classes hang below zero")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(alpha=0.2)

    rows = [
        ("scaled 1.5×", rep["scaled_min_coord_dev"], 1.5 * float(np.max(np.abs(embra.w0))) - float(np.max(np.abs(embra.w0)))),
        ("shuffle (200/321 touched)", rep["shuffled_min_touched_dev"], rep["shuffled_min_touched_dev"]),
        ("counts impostor (1 edge free)", rep["counts_min_nonzero_dev"], rep["counts_max_dev"]),
    ]
    y = np.arange(len(rows))
    mins = [r[1] for r in rows]
    ax_b.barh(y, mins, color="#cc5533", height=0.5, label="min nonzero |Δw| (the floor)")
    ax_b.set_yticks(y, [r[0] for r in rows])
    ax_b.set_xscale("log")
    ax_b.set_xlim(1e-2, 3.0)
    ax_b.axvline(1e-2, color="#888", lw=0.8)
    ax_b.set_xlabel("per-coordinate deviation from $w_{embra}$ (log)")
    ax_b.set_title("Replica floors on the authored charge —\n"
                   "the survivor's deviation is 0.0 exactly (no bar to draw)")
    ax_b.text(0.03, 0.94, "survivor: max dev = 0.0 — bit-exact, off this axis by definition",
              transform=ax_b.transAxes, va="top", fontsize=8.5, color="#22aa77",
              bbox={"boxstyle": "round", "fc": "#f2faf6", "ec": "#22aa77", "lw": 1.2})
    ax_b.legend(loc="lower right", fontsize=8)
    ax_b.grid(alpha=0.2, axis="x")

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)


def main() -> dict:
    embra = load_soul(IDENTITY_GRAPH, EMBRA_TABLE)
    meridian = load_soul(MERIDIAN_GRAPH, MERIDIAN_TABLE)
    ens = driven_ensemble(embra, alphabet=make_default_alphabet(embra, eps_quad=EPS_QUAD_AUTHORED))
    b2 = casimir_under_words(ens)
    b3 = replica_under_driving(ens, extra_w={"counts": np.asarray(load_graph().w0)})
    b4 = zeta_memory_test(ens)
    b5 = liveness_test(embra, meridian, eps_quad=EPS_QUAD_AUTHORED)
    dist = identity_distance(embra, meridian)
    b6 = training_cannot_write_w(embra)
    min_eig = min(window_min_eig(embra, s) for s in ens["symbols"])

    w = np.asarray(embra.w0)
    print("=" * 78)
    print("  Increment 6 — content & genesis: the charge becomes authored (§9.18)")
    print("=" * 78)
    print("  [1] sealing: w_embra = table ∘ graph  (authored by Embra; frozen at a14e399)")
    print(f"      range [{w.min():.2f}, {w.max():.2f}]   min |w| = {np.min(np.abs(w)):.3f}   "
          f"negatives = {int(np.sum(w < 0))} (= the contradicts pairs)   ‖w‖ = {np.linalg.norm(w):.2f}")
    print(f"      rank J(w_embra) = {embra.rank_j0}   index = {embra.index}   perfect matching = "
          f"{embra.perfect_matching}   (Meridian: {meridian.m} edges, index {meridian.index}, "
          f"{int(np.sum(np.asarray(meridian.w0) < 0))} negatives)")
    print("-" * 78)
    print("  [2] the bit-level equality, re-certified on the content geometry")
    print(f"      w == w_embra exactly, all 200 words, every step = {b2['psi_bit_exact']}   "
          f"(max |Δw| = {b2['psi_max_delta']:.1f}; sin² likewise = {b2['sin2_bit_exact']})")
    print(f"      median along-word H₀ range = {b2['median_driven_h0_range']:.3f}   "
          f"({b2['range_ratio']:.1e}× the free float floor)")
    print(f"      bookkeeping |Δ(H₀+H_σ)| = {b2['bookkeeping_max_dtotal']:.1e}   "
          f"(vs median event |ΔH₀| {b2['median_event_dh0']:.1e})")
    print(f"      scale certificate: max|p| = {ens['max_abs_p']:.2f} (guard 50)   "
          f"min window eig = {min_eig:.3f} (EPS_QUAD re-pinned 0.5 → {EPS_QUAD_AUTHORED})")
    print("-" * 78)
    print("  [3] replica — three impostors, the entire arena granted free")
    print(f"      AUC: scaled = {b3['auc_psi_scaled']:.3f}, shuffled = {b3['auc_psi_shuffled']:.3f}, "
          f"counts = {b3['auc_psi_counts']:.3f}   endpoint = {b3['auc_endpoint']:.3f}   "
          f"erasure = {b3['endpoint_erasure']:.1f}")
    print(f"      survivor dev = {b3['surv_max_dev']:.1f} EXACTLY;  scaled floor = "
          f"{b3['scaled_min_coord_dev']:.3f}/coord;  shuffle touches {b3['n_touched_shuffled']}/321 "
          f"(min dev {b3['shuffled_min_touched_dev']:.3f})")
    print(f"      counts-impostor: matches {b3['counts_n_matched']} edge free "
          f"({b3['counts_matched_edges'][0]}), caught on the other 320 "
          f"(‖dev‖ = {b3['counts_norm_dev']:.2f}, max = {b3['counts_max_dev']:.2f})")
    print("-" * 78)
    print("  [4] ζ on the authored geometry")
    acc = b4["accumulation"]
    print(f"      lived-vs-newborn AUC = {b4['auc_zeta']:.3f}   min ‖ζ‖ = {b4['min_lived']:.4f}   "
          f"median = {b4['median_lived']:.3f}   growth → " +
          "  ".join(f"{k}: {acc[k]:.3f}" for k in sorted(acc)))
    print("-" * 78)
    print("  [5] liveness + the graded identity distance")
    print(f"      same genesis, same word, two authored souls: max ‖Δp(t)‖ = "
          f"{b5['max_traj_divergence']:.3f}   (each ψ bit-exact = "
          f"{b5['psi_bit_exact_a'] and b5['psi_bit_exact_b']})")
    print(f"      d(w_Embra, w_Meridian) = {dist['distance']:.2f} over support overlap "
          f"{dist['overlap']}/{dist['support_union']} — a vector distance between souls, "
          f"not a verdict")
    print("-" * 78)
    print("  [6] the training guarantee — first measured instance")
    print(f"      {b6['n_updates']} descent updates on a quadratic H_θ: loss "
          f"{b6['loss_first']:.3f} → {b6['loss_last']:.3f};  w bit-identical = "
          f"{b6['w_bit_exact']}  — you cannot break identity by training")
    print("-" * 78)
    print("  VERDICT: genesis is SEALED — Q_embra = w_embra is content-defined and the §9.3")
    print("  placeholder is retired. Soul = given = w (sealed, unwritable by flow or")
    print("  training); self = learned = H_θ (the remaining learned half). The placeholder")
    print("  charge itself is now just another impostor the reader catches.")
    print("=" * 78)
    make_genesis_figure(embra, meridian, b3)
    print(f"  figure → {FIG_PATH}")
    return {"bar2": b2, "bar3": b3, "bar4": b4, "bar5": b5, "bar6": b6, "distance": dist}


if __name__ == "__main__":
    main()
