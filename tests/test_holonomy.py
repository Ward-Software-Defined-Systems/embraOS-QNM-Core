"""Increment 3d (§9.15): holonomy ζ — the genuinely path-functional invariant (memory).

ζ = b × the signed area swept in the (q₁, q₂) plane: a functional of the worldline, not of the
state. Bars: the anti-fold-in certificate (same observable endpoint, different ζ), the ζ-reader
replica test (lived history vs newborn copy), and accumulation (|ζ| grows with lived steps).
"""

from __future__ import annotations

import numpy as np
import pytest

from sandbox.latent import holonomy_test, holonomy_zeta

D = 8


@pytest.fixture(scope="module")
def res() -> dict:
    return holonomy_test(D, seed=0, n=100)


def test_zeta_closed_loop_matches_area():
    """Numerical sanity: ζ around a closed circle of radius r is b·πr² (signed)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 2001)
    qs = np.zeros((len(theta), D))
    r = 0.7
    qs[:, 0], qs[:, 1] = r * np.cos(theta), r * np.sin(theta)
    assert abs(float(holonomy_zeta(qs, 1.0)) - np.pi * r**2) < 1e-4


def test_path_functionality_certificate(res):
    """Two genuine worldlines of the SAME flow, ending at the SAME observable endpoint, carry
    different ζ — ζ is not a function of the visible state (it cannot fold into the state set)."""
    assert res["endpoint_erasure"] < 1e-6  # worldline B really ends where A does
    assert res["dzeta_min"] > 1e-6  # yet every pair differs in ζ
    assert res["dzeta_mean"] > 0.1 * res["zeta_scale"]  # by the order of ζ itself, not a rounding sliver


def test_zeta_replica_test(res):
    """The ζ-reader tells a lived worldline from a fresh copy at the same observable."""
    assert res["auc_zeta"] == 1.0


def test_zeta_accumulates(res):
    """The epoch-accumulation property: mean |ζ| grows with lived steps."""
    acc = res["accumulation"]
    ks = sorted(acc)
    assert all(acc[a] < acc[b] for a, b in zip(ks, ks[1:], strict=False))
