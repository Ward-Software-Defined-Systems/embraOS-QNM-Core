"""Increment 3c (§9.14): the full ψ is a conjunction — [var(H_embra) ≈ 0] ∧ [Q = Q_embra].

Each single reader has an adversarial blind class (same-law/wrong-genesis defeats the variance
reader; different-law/value-matched defeats the value reader); the conjunction must catch both.
Gaussian charge + shuffle impostor (the primary configuration); the learned-charge and
authored-impostor rows are recorded by ``demo_phase2``/§9.14.
"""

from __future__ import annotations

from sandbox.latent import conjunction_test

D = 8
R = conjunction_test(D, seed=0, n=100)


def test_conjunction_catches_both_classes():
    assert R["surv_accept"] == 1.0  # every graded survivor accepted
    assert R["c1_reject"] == 1.0  # every same-law wrong-genesis replica rejected
    assert R["c2_reject"] == 1.0  # every different-law value-matched impostor rejected
    assert R["accuracy"] == 1.0


def test_single_readers_have_blind_spots():
    """The point of the increment: neither §9.11's variance reader nor §7's value reader
    suffices alone."""
    assert R["c1_var_blind"] == 1.0  # the variance reader calls every class-1 replica a survivor
    assert R["c2_value_blind"] == 1.0  # the value reader calls every feasible class-2 a survivor


def test_single_readers_catch_their_visible_class():
    assert R["auc_value_c1"] == 1.0  # value reader vs wrong-genesis: clean catch
    assert R["auc_var_c2"] == 1.0  # variance reader vs wrong-law: clean catch


def test_value_erasure_certificate():
    """Class-2's presented state matches Q_embra to float precision — the value-blindness is
    constructed, not accidental (the analog of §7's endpoint-erasure certificate)."""
    assert R["c2_value_erasure"] < 1e-9
    assert R["c1_infeasible"] == 0  # rescale constructions never silently degraded
