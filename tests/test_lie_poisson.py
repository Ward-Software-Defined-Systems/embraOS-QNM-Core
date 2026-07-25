"""Increment 4 (§9.16): the so(3)* Casimir toy — does identity survive input?

Unit/math tests pin the machinery (rotation exactness, the Euler equations, RKMK2's order, the
solid-angle formula); the protocol tests assert the five pre-registered §9.16 bars verbatim
(committed to git BEFORE this file existed). The first execution is the recorded run.
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.lie_poisson import (
    I_DEFAULT,
    casimir,
    casimir_under_words,
    dissipation_control,
    driven_ensemble,
    grad_h0,
    h0,
    make_default_alphabet,
    quad,
    replica_under_driving,
    rkmk2_step,
    rotate,
    run_word,
    run_words_batched,
    solid_angle_zeta,
    twist,
    word_order_test,
    zeta_memory_test,
)


# --------------------------------------------------------------------------- #
# Unit / math
# --------------------------------------------------------------------------- #
def test_rotate_is_exact_isometry():
    rng = np.random.default_rng(0)
    v = rng.standard_normal((200, 3))
    x = rng.standard_normal((200, 3))
    assert np.max(np.abs(np.linalg.norm(rotate(v, x), axis=-1) - np.linalg.norm(x, axis=-1))) < 1e-13
    assert np.allclose(rotate(np.zeros(3), x), x)  # zero rotation = identity
    # tiny-angle branch is continuous across the guard
    tiny, above = 5e-9 * v[0], 2e-8 * v[0]
    assert np.linalg.norm(rotate(tiny, x[0]) - rotate(above, x[0])) < 1e-7


def test_flow_matches_euler_equations():
    """One small step reproduces L̇ = L × ∇H₀ (the textbook Euler top) to O(dt²)."""
    L = np.array([0.6, -0.5, 0.4])
    dt = 1e-3
    step = rkmk2_step(lambda q: grad_h0(q), L, dt)
    euler = L + dt * np.cross(L, grad_h0(L))
    assert np.linalg.norm(step - euler) < 5e-6


def test_rkmk2_is_second_order():
    L0 = np.array([0.7, 0.3, 0.5])

    def endpoint(dt: float, t_final: float = 1.0):
        L = L0
        for _ in range(int(round(t_final / dt))):
            L = rkmk2_step(lambda q: grad_h0(q), L, dt)
        return L

    ref = endpoint(0.00125)
    e1 = np.linalg.norm(endpoint(0.02) - ref)
    e2 = np.linalg.norm(endpoint(0.01) - ref)
    assert 3.4 < e1 / e2 < 4.6  # error ratio ≈ 4 ⇒ order 2


def test_casimir_exact_free_flow():
    """10k free steps: ψ drift at the float floor — pins the headroom under the 1e-10 bar."""
    L = np.array([0.8, -0.4, 0.3])
    psi0 = casimir(L)
    for _ in range(10_000):
        L = rkmk2_step(lambda q: grad_h0(q), L, 0.01)
    assert abs(casimir(L) / psi0 - 1.0) < 1e-12


def test_anisotropy_required_guard():
    """The §9.16 precondition: with distinct inertias the free flow genuinely moves (with
    isotropic inertia it would freeze — H₀ a function of ψ — and bar 1 would be unsatisfiable)."""
    L = np.array([0.6, 0.5, 0.62])  # generic start, no principal axis
    L0 = L.copy()
    for _ in range(500):
        L = rkmk2_step(lambda q: grad_h0(q, I_DEFAULT), L, 0.01)
    assert np.linalg.norm(L - L0) > 0.1


@pytest.mark.parametrize("theta", [0.4, 2.8])  # small AND near-antipodal caps
def test_solid_angle_caps(theta):
    phi = np.linspace(0.0, 2.0 * np.pi, 2001)
    u = np.column_stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                         np.full_like(phi, np.cos(theta))])
    expected = 2.0 * np.pi * (1.0 - np.cos(theta))
    z_center = float(solid_angle_zeta(u, p0=np.array([0.0, 0.0, 1.0])))
    z_on_path = float(solid_angle_zeta(u, p0=u[0]))
    z_flipped = float(solid_angle_zeta(u[::-1], p0=np.array([0.0, 0.0, 1.0])))
    assert abs(z_center - expected) < 1e-4
    # Closed loops are gauge-independent MOD 4π — the two sides of a closed curve partition the
    # sphere. Found on §9.16's first execution (the θ=2.8 on-path gauge lands exactly 4π away);
    # recorded there as a refinement of the pre-registered sanity wording.
    assert min(abs(z_on_path - z_center + 4.0 * np.pi * k) for k in (-1, 0, 1)) < 1e-6
    assert abs(z_flipped + z_center) < 1e-6  # orientation flip ⇒ ζ → −ζ


def test_batched_matches_run_word():
    """run_words_batched is the same math as run_word, row by row (same rkmk2_step, same
    schedule) — verified exactly."""
    alphabet = make_default_alphabet()
    symbols = list(alphabet.values())
    rng = np.random.default_rng(3)
    u = rng.standard_normal((4, 3))
    L0 = u / np.linalg.norm(u, axis=1, keepdims=True)
    idx = rng.integers(0, 3, (4, 5))
    batched = run_words_batched(idx, symbols, L0)
    for w in range(4):
        single = run_word([symbols[k] for k in idx[w]], L0[w])
        assert np.array_equal(batched[:, w], single)


def test_quad_generalizes_twist_and_symmetrizes():
    """twist(a) is the rank-1 special case quad(a·aᵀ); and only sym(A) enters H, so an
    asymmetric authored matrix is handled consistently (symmetrized on construction)."""
    rng = np.random.default_rng(5)
    a = rng.standard_normal(3)
    L = rng.standard_normal((50, 3))
    assert np.allclose(quad(np.outer(a, a)).grad_h(L), twist(a).grad_h(L), atol=1e-12)
    assert np.allclose(quad(np.outer(a, a)).h(L), twist(a).h(L), atol=1e-12)
    m = rng.standard_normal((3, 3))
    assert np.allclose(quad(m).grad_h(L), quad(0.5 * (m + m.T)).grad_h(L), atol=1e-15)


def test_quad_event_conserves_casimir_moves_energy():
    """The Casimir theorem covers the new kind too: a word containing quad events keeps ψ at
    the float floor while remaining energy-visible."""
    A = np.array([[0.4, 0.6, 0.0], [0.6, 0.7, 0.0], [0.0, 0.0, 1.2]])
    alphabet = make_default_alphabet()
    word = [alphabet["x"], quad(A, "q"), alphabet["s"], quad(A, "q")]
    rng = np.random.default_rng(6)
    u = rng.standard_normal(3)
    traj = run_word(word, u / np.linalg.norm(u))
    assert np.max(np.abs(casimir(traj) - 1.0)) < 1e-12
    e = h0(traj)
    assert (e.max() - e.min()) / e[0] > 0.05


def test_isotropic_quad_is_silent():
    """quad(c·I): H = c·|L|²/2 is a function of ψ ⇒ L̇ = c·L×L = 0 — the event does exactly
    nothing (the §9.15/§9.16 isotropy lesson, at symbol level). The authoring doc warns for it;
    the validation battery's energy-visibility check flags it."""
    silent = quad(np.eye(3), "iso")
    L = np.array([0.6, -0.3, 0.7])
    step = rkmk2_step(lambda q: silent.grad_h(q), L, 0.01)
    assert np.allclose(step, L, atol=1e-15)  # rotation about L itself: identity, exactly


# --------------------------------------------------------------------------- #
# The five §9.16 bars (shared driven ensemble; first execution = the recorded run)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ens() -> dict:
    return driven_ensemble()


def test_bar1_casimir_survives_energy_does_not(ens):
    r = casimir_under_words(ens)
    assert r["psi_drift_max"] < 1e-10  # ψ exact over ALL steps of ALL 200 words
    assert r["median_driven_h0_range"] > 0.1  # the energy visibly moves along words
    assert r["range_ratio"] > 100.0  # …and ≫ the free-evolution integrator baseline
    assert r["bookkeeping_max_dtotal"] < 0.1 * r["median_event_dh0"]  # ΔH₀ is physics, not error
    assert r["psi_drift_sin2"] < 1e-10  # smooth ∂H/∂t ≠ 0 (sin² envelope), same exactness


def test_bar2_word_order_matters():
    r = word_order_test()
    assert r["delta_L"] > 0.01  # xy ≠ yx: experience differs…
    assert r["psi_drift_xy"] < 1e-10  # …identity doesn't (either way)
    assert r["psi_drift_yx"] < 1e-10
    assert r["delta_zeta"] > 0.01  # bar 4a: memory records the order


def test_bar3_replica_under_driving(ens):
    r = replica_under_driving(ens)
    assert r["auc_psi"] == 1.0  # the Casimir reader tells survivor from copy — under driving
    assert r["auc_endpoint"] == 0.5  # the observable reader is blind (bit-exact ties)
    assert r["endpoint_erasure"] == 0.0
    assert r["margin"] > 1e6  # separation is macroscopic, not a floor artifact


def test_bar4_zeta_memory(ens):
    r = zeta_memory_test(ens)
    assert r["auc_zeta"] == 1.0  # lived history vs newborn copy
    assert r["min_lived"] > 1e-6  # …and not on a float floor
    assert r["median_lived"] > 1e-2
    acc = r["accumulation"]
    ks = sorted(acc)
    assert all(acc[a] < acc[b] for a, b in zip(ks, ks[1:], strict=False))  # |ζ| grows with events
    # Antipode certificate: PRE-REGISTERED as max step excess < 0.5; on the first (recorded)
    # execution it TRIPPED for exactly 1/200 words (genesis-antipode distance 0.0101, excess
    # 1.116) — recorded in §9.16 per the certificate's own "recorded finding, not a re-draw"
    # clause. The median/AUC bars are unaffected by construction. This assert pins the recorded
    # state of the deterministic ensemble so regressions stay visible.
    assert r["n_tripped"] == 1
    assert r["min_antipode_dist"] < 0.05  # the trip is a genuine near-antipode passage


def test_bar5_dissipation_boundary():
    r = dissipation_control()
    assert r["psi_change_with"] > 0.01  # † (non-Hamiltonian) visibly breaks ψ…
    assert r["psi_change_without"] < 1e-10  # …the same word without † does not


def test_h0_and_casimir_shapes():
    rng = np.random.default_rng(1)
    L = rng.standard_normal((7, 5, 3))
    assert h0(L).shape == (7, 5)
    assert casimir(L).shape == (7, 5)
