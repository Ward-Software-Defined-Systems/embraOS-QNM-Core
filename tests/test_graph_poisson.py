"""Increment 5 (§9.17): 𝔤(G)* — the identity graph becomes the bracket.

The six §9.17 bars were pre-registered in git (commit ``f10d106``) AND server-timestamped
(GitHub issue #1) BEFORE this file existed; the first execution of the Embra-ensemble
protocol is the recorded run. Unit/math tests pin the machinery on SYNTHETIC graphs (a
triangle — odd, no perfect matching, index > m; a 4-path — perfect matching, index = m):
exercising the graph-parametric clause is part of the point.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sandbox.graph_poisson import (
    MERIDIAN_GRAPH,
    Symbol,
    bracket_certificate,
    casimir_under_words,
    dissipation_control,
    driven_ensemble,
    h0,
    heisenberg_certificate,
    kick,
    liveness_test,
    load_graph,
    make_default_alphabet,
    quad,
    replica_under_driving,
    run_bare_events,
    run_word,
    step_map,
    weaken,
    window_min_eig,
    word_order_test,
    zeta_edges,
    zeta_memory_test,
)


# --------------------------------------------------------------------------- #
# Synthetic graphs (unit fixtures — the graph-parametric clause in action)
# --------------------------------------------------------------------------- #
def _write_graph(tmp_path, name: str, nodes: list[str], edges: list[tuple[str, str]]):
    path = tmp_path / f"{name}.graph.json"
    path.write_text(json.dumps({
        "_comment": "synthetic unit-test graph",
        "nodes": [{"_comment": "divider"}] + [{"id": n, "type": "t", "text": n} for n in nodes],
        "edges": [{"src": s, "dst": d, "relation": "r"} for s, d in edges],
    }))
    return path


@pytest.fixture()
def triangle(tmp_path):
    return load_graph(_write_graph(tmp_path, "tri", ["a", "b", "c"],
                                   [("a", "b"), ("b", "c"), ("a", "c")]))


@pytest.fixture()
def path4(tmp_path):
    return load_graph(_write_graph(tmp_path, "p4", ["a", "b", "c", "d"],
                                   [("a", "b"), ("b", "c"), ("c", "d")]))


def test_loader_orientation_aggregation_and_lock(tmp_path):
    """Lex orientation (i < j over sorted ids), D1 count-aggregation of parallel relations,
    divider skipping, and the mechanical state partition: w₀ comes back write-locked."""
    path = _write_graph(tmp_path, "agg", ["b", "a"],
                        [("b", "a"), ("a", "b"), ("a", "b")])  # one pair, three triples
    alg = load_graph(path)
    assert alg.ids == ("a", "b")
    assert alg.n == 1 + 1 and alg.m == 1
    assert alg.edges.tolist() == [[0, 1]]
    assert alg.w0.tolist() == [3.0]  # parallel triples aggregate into the pair's count
    with pytest.raises((ValueError, RuntimeError)):
        alg.w0[0] = 99.0  # the charge is not writable — the partition is mechanical


def test_derived_quantities_are_computed_not_assumed(triangle, path4):
    """Per-graph rank/index/matching are load-time facts. The triangle (odd) has NO perfect
    matching: rank 2, index 4 > m — extra Casimirs straddle the arena, recorded, not an
    error. The 4-path has one: rank = n, index = m."""
    assert (triangle.rank_j0, triangle.index, triangle.perfect_matching) == (2, 4, False)
    assert (path4.rank_j0, path4.index, path4.perfect_matching) == (4, 3, True)


def test_j_is_skew_with_graph_support(triangle):
    rng = np.random.default_rng(0)
    w = rng.standard_normal(triangle.m)
    J = triangle.J(w)
    assert np.array_equal(J, -J.T)
    for k, (i, j) in enumerate(triangle.edges):
        assert J[i, j] == w[k] and J[j, i] == -w[k]
    assert np.count_nonzero(J) == 2 * triangle.m


def test_step_map_matches_the_flow(path4):
    """The exact affine map reproduces ṗ = J(w)(Mp + a) — checked against a fine-step RK4
    reference for a generic quadratic+linear window Hamiltonian."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    sym = Symbol("s", a=rng.standard_normal(4), A=0.5 * (A + A.T))
    w = rng.standard_normal(path4.m)
    p = rng.standard_normal(4)
    dt = 0.01
    phi, b = step_map(path4, w, dt, sym)
    J, M = path4.J(w), np.eye(4) + sym.A

    def f(q):
        return J @ (M @ q + sym.a)

    q = p.copy()
    h = dt / 200.0
    for _ in range(200):
        k1 = f(q)
        k2 = f(q + 0.5 * h * k1)
        k3 = f(q + 0.5 * h * k2)
        k4 = f(q + h * k3)
        q = q + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.linalg.norm((phi @ p + b) - q) < 1e-10


def test_window_energy_exact_and_gap_isometry(path4):
    """The window Hamiltonian H₀ + H_σ is conserved along its own exact flow to float; the
    gap flow (M = I, skew generator) is an isometry — H₀ flat in silence at the float floor.
    This is what makes the bar-1 bookkeeping certificate near-trivial here."""
    rng = np.random.default_rng(2)
    q_sym = quad(np.diag([0.3, -0.2, 0.4, 0.1]), "q")
    p0 = rng.standard_normal(4)
    w = rng.standard_normal(path4.m)
    traj = run_word([q_sym], p0, path4, w, dt=0.01)
    n_gap = 50
    ev = traj[n_gap: n_gap + 51]
    total = h0(ev) + q_sym.h(ev)
    assert np.max(np.abs(total - total[0])) < 1e-11
    # Each gap is flat at its OWN level (the event moved H₀ between them — that is the physics).
    gap1, gap2 = h0(traj[: n_gap + 1]), h0(traj[n_gap + 50:])
    assert np.max(np.abs(gap1 - gap1[0])) < 1e-11
    assert np.max(np.abs(gap2 - gap2[0])) < 1e-11
    assert abs(gap2[0] - gap1[0]) > 0.01  # …and the event genuinely moved it


def test_bare_kick_is_exact_translation(path4):
    """H = a·p alone: the augmented generator is nilpotent, so even the exponential is exact —
    Φ = I, b = dt·J(w)a, and the event is a pure translation."""
    rng = np.random.default_rng(3)
    w = rng.standard_normal(path4.m)
    a = rng.standard_normal(4)
    phi, b = step_map(path4, w, 0.01, kick(a), with_h0=False)
    assert np.array_equal(phi, np.eye(4))
    assert np.allclose(b, 0.01 * path4.J(w) @ a, atol=1e-15)
    p0 = rng.standard_normal(4)
    ps = run_bare_events([kick(a)], p0, path4, w, dt=0.01, tau_event=0.5)
    assert np.allclose(ps[-1] - ps[0], 0.5 * path4.J(w) @ a, atol=1e-12)


def test_isotropic_quad_is_not_silent_here(triangle):
    """The so(3)* silence trap MOVED: H ∝ |p|² is not a Casimir of this bracket, so quad(I)
    genuinely flows (silent means H = f(w) alone — ∇_p H ≡ 0)."""
    rng = np.random.default_rng(4)
    w = rng.standard_normal(triangle.m)
    p0 = rng.standard_normal(3)
    ps = run_bare_events([quad(np.eye(3), "iso")], p0, triangle, w, dt=0.01, tau_event=0.5)
    assert np.linalg.norm(ps[-1] - ps[0]) > 1e-3
    null = Symbol("null")  # the degenerate silent symbol: H ≡ 0
    ps2 = run_bare_events([null], p0, triangle, w, dt=0.01, tau_event=0.5)
    assert np.array_equal(ps2[-1], p0)


def test_zeta_closed_loop_exact_gauge_independence(tmp_path):
    """Flat edge planes: a closed loop's ζ is the enclosed area — EXACTLY gauge-independent
    (no mod-4π analog; the so(3)* finding does not transfer, by design), orientation-odd."""
    alg = load_graph(_write_graph(tmp_path, "pair", ["a", "b"], [("a", "b")]))
    t = np.linspace(0.0, 2.0 * np.pi, 2001)
    r = 0.7
    ps = np.column_stack([r * np.cos(t), r * np.sin(t)])
    area = np.pi * r * r
    z1 = float(zeta_edges(ps, alg.edges, p0=np.array([0.0, 0.0]))[0])
    z2 = float(zeta_edges(ps, alg.edges, p0=np.array([3.0, -2.0]))[0])
    assert abs(z1 - area) < 1e-4
    assert abs(z1 - z2) < 1e-12
    z_flip = float(zeta_edges(ps[::-1], alg.edges, p0=np.array([0.0, 0.0]))[0])
    assert abs(z_flip + area) < 1e-4


def test_weaken_is_surgical_and_out_of_alphabet(triangle):
    w = np.abs(np.random.default_rng(5).standard_normal(triangle.m)) + 1.0
    cut = weaken(w, 1, 0.5, 0.01, 50)
    assert cut[1] == pytest.approx(w[1] * (1.0 - 0.005) ** 50)
    others = [0, 2]
    assert np.array_equal(cut[others], w[others])
    assert not isinstance(cut, Symbol)  # † is not spellable as input


def test_ensemble_engine_matches_run_word(tmp_path):
    """The streaming ensemble is the same math as run_word row-by-row (same exact maps, same
    schedule): endpoints, per-step H₀, and ζ agree to the float floor. (BLAS batching may
    reorder accumulations, so the comparison is at 1e-12, not bit-level — recorded.)"""
    alg = load_graph(_write_graph(tmp_path, "p4b", ["a", "b", "c", "d"],
                                  [("a", "b"), ("b", "c"), ("c", "d")]))
    alphabet = {
        "x": kick(np.array([0.4, 0.0, 0.0, 0.0]), "x"),
        "y": kick(np.array([0.0, 0.3, 0.0, 0.0]), "y"),
        "q": quad(np.diag([0.2, -0.1, 0.3, 0.0]), "q"),
    }
    ens = driven_ensemble(alg, n_seeds=1, words_per_seed=3, word_len=3,
                          alphabet=alphabet, checkpoints=(1, 3))
    symbols = ens["symbols"]
    for row in range(3):
        word = [symbols[k] for k in ens["word_indices"][row]]
        single = run_word(word, ens["p0"][row], alg, alg.w0)
        assert np.allclose(single[-1], ens["p_final"][row], atol=1e-12)
        assert np.allclose(h0(single), ens["h0_series"][:, row], atol=1e-12)
        z = zeta_edges(single, alg.edges)
        assert np.allclose(z, ens["zeta_final"][row], atol=1e-12)


def test_shapes_batched(triangle):
    rng = np.random.default_rng(6)
    p = rng.standard_normal((7, 5, 3))
    sym = quad(np.eye(3))
    assert h0(p).shape == (7, 5)
    assert sym.h(p).shape == (7, 5)
    assert sym.grad_h(p).shape == (7, 5, 3)


# --------------------------------------------------------------------------- #
# The recorded graphs: load-time invariants (planning-review numbers become facts)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def alg():
    return load_graph()


@pytest.fixture(scope="module")
def alg_meridian():
    return load_graph(MERIDIAN_GRAPH)


def test_embra_and_meridian_derived_invariants(alg, alg_meridian):
    """The §9.17 conventions as measured facts: Embra v3 aggregates to 321 edges (291/27/3 at
    counts 1/2/3), has a perfect matching, index = m. Meridian: 349 edges, no parallel
    relations, perfect matching, index = m."""
    assert (alg.n, alg.m) == (100, 321)
    assert (alg.rank_j0, alg.index, alg.perfect_matching) == (100, 321, True)
    counts = {v: int(np.sum(alg.w0 == v)) for v in (1.0, 2.0, 3.0)}
    assert counts == {1.0: 291, 2.0: 27, 3.0: 3}
    assert (alg_meridian.n, alg_meridian.m) == (100, 349)
    assert (alg_meridian.rank_j0, alg_meridian.index) == (100, 349)
    assert bool(np.all(alg_meridian.w0 == 1.0))  # no parallel relations


# --------------------------------------------------------------------------- #
# The six §9.17 bars (shared driven ensemble; first execution = the recorded run)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ens(alg) -> dict:
    return driven_ensemble(alg)


def test_bar1_psi_is_a_bit_level_equality(ens):
    r = casimir_under_words(ens)
    assert r["psi_bit_exact"] is True  # w == w₀ exactly, all 200 words, every step
    assert r["psi_max_delta"] == 0.0
    assert r["sin2_bit_exact"] is True  # smooth ∂H/∂t ≠ 0: same equality
    assert r["sin2_max_delta"] == 0.0
    assert r["median_driven_h0_range"] > 0.1  # the law visibly moves along words…
    assert r["range_ratio"] > 100.0  # …and ≫ the free-evolution float floor
    assert r["bookkeeping_max_dtotal"] < 0.1 * r["median_event_dh0"]  # ΔH₀ is physics


def test_bar2a_bracket_certificate(alg):
    r = bracket_certificate(alg)
    assert r["max_pred_error"] < 1e-10  # Δ(bᵀp) = τ·bᵀJ(w)a, exactly — every observable
    assert abs(r["adjacent_abs"] - 0.75) < 1e-10  # τ·ε·w₀ on the triple-relation edge
    assert abs(r["adjacent_measured"] - r["adjacent_predicted"]) < 1e-12
    assert abs(r["nonadjacent_measured"]) < 1e-13  # non-adjacent kicks commute: topology
    assert r["adjacent_closed_form"] == pytest.approx(0.75)


def test_bar2b_heisenberg_signature(alg):
    r = heisenberg_certificate(alg)
    assert r["delta_p"] < 1e-12  # bare kicks commute in STATE (translations)…
    assert r["max_pred_error"] < 1e-10  # …while ζ differs by the exact parallelogram areas
    assert abs(r["shared_edge_measured"] - r["shared_edge_closed_form"]) < 1e-10
    assert r["shared_edge_closed_form"] == pytest.approx(0.5625)  # (τ·ε·w₀)²


def test_bar2c_order_reaches_state_through_the_law(alg):
    r = word_order_test(alg)
    assert r["delta_p"] > 0.01  # with H₀ running, xy ≠ yx in state…
    assert r["delta_zeta"] > 0.01  # …and in memory
    assert r["psi_bit_exact_xy"] is True and r["psi_bit_exact_yx"] is True


def test_bar3_replica_under_driving(ens):
    r = replica_under_driving(ens)
    assert r["auc_psi_scaled"] == 1.0  # the w-reader tells survivor from copy — under driving
    assert r["auc_psi_shuffled"] == 1.0  # …including the value-multiset-preserving copier
    assert r["auc_endpoint"] == 0.5  # the arena reader is blind (bit-exact ties)
    assert r["endpoint_erasure"] == 0.0
    assert r["surv_max_dev"] == 0.0  # the survivor's deviation is not small — it is zero
    assert r["scaled_min_coord_dev"] >= 0.5  # every coordinate wrong by ≥ 0.5·min w₀
    assert r["n_touched_shuffled"] >= 1  # placement guard (near-uniform counts, recorded)
    assert r["shuffled_min_touched_dev"] >= 1.0  # integer counts differ by ≥ 1 where touched


def test_bar4_zeta_memory_and_scale_certificate(ens):
    r = zeta_memory_test(ens)
    assert r["auc_zeta"] == 1.0  # lived history vs newborn copy
    assert r["min_lived"] > 1e-6  # …and not on a float floor
    assert r["median_lived"] > 1e-2
    acc = r["accumulation"]
    ks = sorted(acc)
    assert all(acc[a] < acc[b] for a, b in zip(ks, ks[1:], strict=False))  # ‖ζ‖ grows
    assert r["max_abs_p"] < 50.0  # the scale certificate (compactness replacement)
    assert r["min_window_eig"] == pytest.approx(0.5)  # coercive windows: 1 − ε_quad


def test_bar5_dagger_is_legible_graph_surgery(alg):
    r = dissipation_control(alg)
    assert r["psi_bit_exact_without"] is True  # the same word without † is an equality
    assert r["psi_max_delta_without"] == 0.0
    assert abs(r["touched_rel_change"] - r["closed_form"]) < 1e-12  # the map's own value
    assert r["closed_form"] == pytest.approx(1.0 - 0.995**50)  # ≈ 0.2217 (single exponent)
    assert r["others_bit_exact"] is True  # …and ONLY the named relation moved
    assert r["touched_edge"] == "no_pretense—precision_over_spectacle"


def test_bar6_liveness_two_souls_same_events(alg, alg_meridian):
    r = liveness_test(alg, alg_meridian)
    assert r["max_traj_divergence"] > 0.01  # identity shapes experience — not a dead tag
    assert r["psi_bit_exact_a"] is True and r["psi_bit_exact_b"] is True
    assert (r["m_b"], r["index_b"]) == (349, 349)


def test_default_alphabet_is_pinned(alg):
    """The §9.17 alphabet: x/y on the adjacent triple-relation pair, z adjacent to neither,
    q the edge-quad on the x–y edge; window coercivity holds for every symbol."""
    sigma = make_default_alphabet(alg)
    assert set(sigma) == {"x", "y", "z", "q"}
    k = alg.edge_of("no_pretense", "precision_over_spectacle")
    assert alg.w0[k] == 3.0
    with pytest.raises(KeyError):
        alg.edge_of("no_pretense", "always_becoming_never_finished")  # pinned non-adjacency
    assert min(window_min_eig(alg, s) for s in sigma.values()) > 0.0
