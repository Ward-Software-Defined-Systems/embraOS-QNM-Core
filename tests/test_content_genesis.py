"""Increment 6 (§9.18): content & genesis — the charge becomes authored.

The six §9.18 bars were pre-registered in git (commit ``5c03fe7``) AND server-timestamped
(GitHub issue #2) BEFORE the loader/battery code existed; the authored content was frozen one
commit earlier (``a14e399``). Unit/math tests pin the loader and the inertia-aware machinery
on SYNTHETIC graphs and tables; the recorded protocol lives in the ``bar`` tests.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sandbox.graph_poisson import (
    EMBRA_TABLE,
    EPS_QUAD_AUTHORED,
    IDENTITY_GRAPH,
    MERIDIAN_GRAPH,
    MERIDIAN_TABLE,
    _genesis,
    _h0,
    casimir_under_words,
    driven_ensemble,
    identity_distance,
    kick,
    liveness_test,
    load_graph,
    load_soul,
    make_default_alphabet,
    quad,
    replica_under_driving,
    run_word,
    training_cannot_write_w,
    window_min_eig,
    zeta_memory_test,
)


# --------------------------------------------------------------------------- #
# Synthetic loader / table unit tests (dev-safe)
# --------------------------------------------------------------------------- #
def _write_graph(tmp_path, name, nodes, edges):
    path = tmp_path / f"{name}.graph.json"
    path.write_text(json.dumps({
        "nodes": [{"id": n, "type": t, "text": n} for n, t in nodes],
        "edges": [{"src": s, "dst": d, "relation": r} for s, d, r in edges],
    }))
    return path


def _write_table(tmp_path, name, weights, inertias, aggregation="sum"):
    path = tmp_path / f"{name}.table.json"
    path.write_text(json.dumps({
        "schema": "embraos.weights/1", "aggregation": aggregation,
        "relation_weights": weights, "node_inertias": inertias,
    }))
    return path


NODES = [("a", "heavy"), ("b", "light"), ("c", "light"), ("d", "heavy")]
EDGES = [("a", "b", "bond"), ("b", "c", "bond"), ("c", "d", "spark")]


def test_loader_rejects_bad_tables(tmp_path):
    g = _write_graph(tmp_path, "g", NODES, EDGES)
    ok_w = {"bond": 1.5, "spark": -0.5}
    ok_i = {"heavy": 2.0, "light": 0.5}
    load_soul(g, _write_table(tmp_path, "ok", ok_w, ok_i))  # baseline: loads
    with pytest.raises(ValueError, match="relation keys"):
        load_soul(g, _write_table(tmp_path, "t1", {"bond": 1.0}, ok_i))
    with pytest.raises(ValueError, match="unfilled"):
        load_soul(g, _write_table(tmp_path, "t2", {"bond": 1.0, "spark": None}, ok_i))
    with pytest.raises(ValueError, match="inertias must be > 0"):
        load_soul(g, _write_table(tmp_path, "t3", ok_w, {"heavy": 2.0, "light": 0.0}))
    with pytest.raises(ValueError, match="aggregation"):
        load_soul(g, _write_table(tmp_path, "t4", ok_w, ok_i, aggregation="mean"))
    # A composed exact-zero edge is a SILENT charge coordinate — a loader-level error.
    g2 = _write_graph(tmp_path, "g2", NODES, EDGES + [("a", "b", "spark"),
                                                     ("a", "b", "spark"),
                                                     ("a", "b", "spark")])
    with pytest.raises(ValueError, match="silent"):
        load_soul(g2, _write_table(tmp_path, "t5", {"bond": 1.5, "spark": -0.5}, ok_i))


def test_sealing_is_deterministic_and_locked(tmp_path):
    g = _write_graph(tmp_path, "g", NODES, EDGES)
    t = _write_table(tmp_path, "t", {"bond": 1.5, "spark": -0.5},
                     {"heavy": 2.0, "light": 0.5})
    s1, s2 = load_soul(g, t), load_soul(g, t)
    assert np.array_equal(s1.w0, s2.w0) and np.array_equal(s1.inertia, s2.inertia)
    assert s1.w0.tolist() == [1.5, 1.5, -0.5]  # lex pairs (a,b),(b,c),(c,d)
    assert s1.inertia.tolist() == [2.0, 0.5, 0.5, 2.0]
    with pytest.raises((ValueError, RuntimeError)):
        s1.w0[0] = 9.0
    with pytest.raises((ValueError, RuntimeError)):
        s1.inertia[0] = 9.0


def test_all_ones_table_reproduces_placeholder_physics(tmp_path):
    """The authored path with a trivial table (weights = counts, inertias = 1) is the same
    physics as the recorded placeholder path — the machinery split is a dispatch, not a fork.
    (Genesis normalization takes a different float route; agreement is at the float floor.)"""
    g = _write_graph(tmp_path, "g", NODES, EDGES)
    t = _write_table(tmp_path, "t", {"bond": 1.0, "spark": 1.0},
                     {"heavy": 1.0, "light": 1.0})
    base, soul = load_graph(g), load_soul(g, t)
    assert np.array_equal(base.w0, soul.w0)
    alphabet = {"x": kick(np.array([0.4, 0.0, 0.0, 0.0]), "x"),
                "q": quad(np.diag([0.2, -0.1, 0.3, 0.0]), "q")}
    e_base = driven_ensemble(base, n_seeds=1, words_per_seed=2, word_len=3,
                             alphabet=alphabet, checkpoints=(1, 3))
    e_soul = driven_ensemble(soul, n_seeds=1, words_per_seed=2, word_len=3,
                             alphabet=alphabet, checkpoints=(1, 3))
    assert np.allclose(e_base["p0"], e_soul["p0"], atol=1e-14)
    assert np.allclose(e_base["h0_series"], e_soul["h0_series"], atol=1e-12)
    assert np.allclose(e_base["p_final"], e_soul["p_final"], atol=1e-11)


def test_authored_genesis_and_gap_law(tmp_path):
    """Genesis is energy-normalized under the AUTHORED law (H₀(p₀) = e0 to float), and the
    anisotropic gap flow conserves that law exactly — while |p| is no longer an isometry
    invariant (that was a placeholder artifact, gone by design)."""
    g = _write_graph(tmp_path, "g", NODES, EDGES)
    t = _write_table(tmp_path, "t", {"bond": 1.5, "spark": -0.5},
                     {"heavy": 3.0, "light": 0.5})
    soul = load_soul(g, t)
    rng = np.random.default_rng(0)
    p0 = _genesis(soul, rng.standard_normal(4), 1.0)
    assert float(_h0(soul, p0)) == pytest.approx(1.0, abs=1e-13)
    traj = run_word([], p0, soul, dt=0.01, tau_gap=5.0)
    e = _h0(soul, traj)
    assert np.max(np.abs(e - e[0])) < 1e-11
    norms = np.linalg.norm(traj, axis=-1)
    assert np.max(np.abs(norms - norms[0])) > 1e-3  # anisotropy: |p| genuinely varies


# --------------------------------------------------------------------------- #
# The six §9.18 bars (authored souls; first execution = the recorded run)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def embra_soul():
    return load_soul(IDENTITY_GRAPH, EMBRA_TABLE)


@pytest.fixture(scope="module")
def meridian_soul():
    return load_soul(MERIDIAN_GRAPH, MERIDIAN_TABLE)


@pytest.fixture(scope="module")
def ens_soul(embra_soul) -> dict:
    return driven_ensemble(
        embra_soul,
        alphabet=make_default_alphabet(embra_soul, eps_quad=EPS_QUAD_AUTHORED))


def test_bar1_sealing_facts(embra_soul):
    again = load_soul(IDENTITY_GRAPH, EMBRA_TABLE)
    assert np.array_equal(embra_soul.w0, again.w0)  # sealing is deterministic
    assert np.array_equal(embra_soul.inertia, again.inertia)
    assert float(np.min(np.abs(embra_soul.w0))) == pytest.approx(0.400, abs=1e-12)
    neg = set(np.where(np.asarray(embra_soul.w0) < 0)[0].tolist())
    data = json.loads(IDENTITY_GRAPH.read_text())
    contra = {embra_soul.edge_of(e["src"], e["dst"]) for e in data["edges"]
              if e.get("relation") == "contradicts"}
    assert len(neg) == 23 and neg == contra  # the negatives ARE the contradicts pairs
    assert (embra_soul.rank_j0, embra_soul.index, embra_soul.perfect_matching) == (100, 321, True)
    with pytest.raises((ValueError, RuntimeError)):
        embra_soul.w0[0] = 9.0  # sealed means sealed


def test_bar1_meridian_counterpart(meridian_soul):
    assert (meridian_soul.rank_j0, meridian_soul.index) == (100, 349)
    assert int(np.sum(np.asarray(meridian_soul.w0) < 0)) == 25  # its opposition class
    assert meridian_soul.perfect_matching is True


def test_bar2_bit_level_equality_on_content(ens_soul, embra_soul):
    r = casimir_under_words(ens_soul)
    assert r["psi_bit_exact"] is True  # w == w_embra exactly — all words, every step
    assert r["psi_max_delta"] == 0.0
    assert r["sin2_bit_exact"] is True and r["sin2_max_delta"] == 0.0
    assert r["median_driven_h0_range"] > 0.1  # the authored law still visibly moves
    assert r["range_ratio"] > 100.0
    assert r["bookkeeping_max_dtotal"] < 0.1 * r["median_event_dh0"]
    assert ens_soul["max_abs_p"] < 50.0  # scale certificate on the authored geometry
    min_eig = min(window_min_eig(embra_soul, s) for s in ens_soul["symbols"])
    assert min_eig > 0.05  # the re-pinned EPS_QUAD is coercive (expect ≈ 0.08)


def test_bar3_replica_three_impostors(ens_soul):
    counts = np.asarray(load_graph().w0)  # the §9.17 placeholder, demoted to impostor
    r = replica_under_driving(ens_soul, extra_w={"counts": counts})
    assert r["auc_psi_scaled"] == 1.0
    assert r["auc_psi_shuffled"] == 1.0
    assert r["auc_psi_counts"] == 1.0
    assert r["auc_endpoint"] == 0.5 and r["endpoint_erasure"] == 0.0
    assert r["surv_max_dev"] == 0.0  # the survivor's deviation is not small — it is zero
    assert r["scaled_min_coord_dev"] == pytest.approx(0.200, abs=1e-12)  # 0.5·min|w|
    assert r["n_touched_shuffled"] == 200  # placeholder's 54 → 200: content grew the teeth
    assert r["shuffled_min_touched_dev"] > 0.04
    # The honest clause, pre-registered: the topology-knowing impostor gets ONE edge free.
    assert r["counts_n_matched"] == 1
    assert r["counts_matched_edges"] == ["embra—origin"]
    assert r["counts_norm_dev"] > 10.0
    assert r["counts_max_dev"] > 1.0
    assert r["counts_min_nonzero_dev"] > 0.05


def test_bar4_zeta_on_authored_geometry(ens_soul):
    r = zeta_memory_test(ens_soul)
    assert r["auc_zeta"] == 1.0
    assert r["min_lived"] > 1e-6
    assert r["median_lived"] > 1e-2
    acc = r["accumulation"]
    ks = sorted(acc)
    assert all(acc[a] < acc[b] for a, b in zip(ks, ks[1:], strict=False))


def test_bar5_liveness_authored_souls(embra_soul, meridian_soul):
    r = liveness_test(embra_soul, meridian_soul, eps_quad=EPS_QUAD_AUTHORED)
    assert r["max_traj_divergence"] > 0.01
    assert r["psi_bit_exact_a"] is True and r["psi_bit_exact_b"] is True
    d = identity_distance(embra_soul, meridian_soul)
    assert d["distance"] == pytest.approx(33.62, abs=0.01)  # the graded identity distance
    assert d["overlap"] == 21  # …across barely-shared support (21 of 649 index-pairs)


def test_bar6_training_cannot_write_w(embra_soul):
    r = training_cannot_write_w(embra_soul)
    assert r["w_bit_exact"] is True  # twenty descent updates; the charge is not an operand
    assert np.isfinite(r["loss_first"]) and np.isfinite(r["loss_last"])
    with pytest.raises((ValueError, RuntimeError)):
        embra_soul.w0[0] = 9.0  # and the lock itself holds
