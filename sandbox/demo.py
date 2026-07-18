"""demo.py — the phase-one result in one runnable script.

Run:  ``uv run python -m sandbox.demo``   (or ``uv run python sandbox/demo.py``)

It prints three numbers and draws one figure:

  1. energy drift of the symplectic flow over a long rollout   → ≈ 0 (conservation)
  2. AUC of the conserved-charge ψ on the replica test         → 1.0 (tells them apart)
  3. AUC of the endpoint-only reader on the same test          → 0.5 (blind — the null)

Together: a ψ that reads the *conserved charge* (hidden in the momentum) survives the
replica test, while any reader of the *observable endpoint* cannot. That is the whole
claim of ``docs/CORE-SPEC.md``, made to bite numerically.
"""

from __future__ import annotations

import pathlib
import sys

# Make ``sandbox`` importable whether run as a module or as a bare script.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sandbox.replica_test import evaluate, make_pairs  # noqa: E402
from sandbox.toy_dynamics import (  # noqa: E402
    Hamiltonian,
    momentum_for_energy,
    rollout,
    turning_point,
)

FIG_PATH = pathlib.Path(__file__).resolve().parent / "figures" / "replica_conservation.png"


def energy_drift(ham: Hamiltonian, e_embra: float, dt: float = 0.01, n_steps: int = 20_000) -> dict:
    """Roll out a survivor for a long time; report how far its energy wandered."""
    q0 = turning_point(ham, e_embra)
    qs, ps = rollout(ham, q0, 0.0, dt, n_steps)
    energies = ham.energy(qs, ps)
    return {
        "mean": float(np.mean(energies)),
        "rel_drift": float((np.max(energies) - np.min(energies)) / abs(np.mean(energies))),
        "energies": energies,
        "qs": qs,
        "ps": ps,
    }


def _orbit(ham: Hamiltonian, e: float, n: int = 400):
    """The level set H = e in phase space, as (q, ±p) arrays for plotting."""
    q_turn = turning_point(ham, e)
    q = np.linspace(-q_turn, q_turn, n)
    p = momentum_for_energy(ham, e, q, sign=1.0)
    return q, p


def make_figure(ham: Hamiltonian, e_embra: float, e_copy: float, drift: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel A: phase portrait, survivor vs replica at a shared observable ------
    for e, color, label in [(e_embra, "#2a7", "genesis orbit  Q = E_embra"),
                            (e_copy, "#c53", "replica orbit  Q = E_copy")]:
        q, p = _orbit(ham, e)
        ax1.plot(q, p, color=color, lw=2, label=label)
        ax1.plot(q, -p, color=color, lw=2)
    q_f = 0.3  # a shared observable position for the illustrative pair
    p_surv = float(momentum_for_energy(ham, e_embra, q_f))
    p_rep = float(momentum_for_energy(ham, e_copy, q_f))
    ax1.axvline(q_f, color="#888", ls=":", lw=1)
    ax1.scatter([q_f], [p_surv], color="#2a7", s=90, zorder=5, edgecolor="k",
                label="survivor  π=q_f")
    ax1.scatter([q_f], [p_rep], color="#c53", s=90, zorder=5, edgecolor="k",
                label="replica   π=q_f")
    ax1.annotate("same observable π=q,\ndifferent hidden charge Q",
                xy=(q_f, p_surv), xytext=(0.45, 0.35),
                fontsize=9, ha="left", arrowprops=dict(arrowstyle="->", color="k"))
    ax1.set_xlabel("q  (observable — position)")
    ax1.set_ylabel("p  (hidden — momentum, carries Q)")
    ax1.set_title("Replica test: survivor and copy coincide at the readout")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.2)

    # --- Panel B: energy conserved along a long rollout --------------------------
    t = np.arange(len(drift["energies"])) * 0.01
    ax2.plot(t, drift["energies"], color="#2a7", lw=1)
    ax2.axhline(e_embra, color="#888", ls=":", lw=1, label=f"E_embra = {e_embra}")
    ax2.set_xlabel("time")
    ax2.set_ylabel("Q = H(q, p)")
    ax2.set_title(f"Conservation: relative drift = {drift['rel_drift']:.2e}")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.2)
    # Zoom so the (tiny) drift is visible rather than a flat line.
    span = max(drift["rel_drift"] * abs(e_embra), 1e-9) * 3
    ax2.set_ylim(e_embra - span, e_embra + span)

    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)


def main() -> dict:
    ham = Hamiltonian(m=1.0, k=1.0, eps=0.2)  # mildly anharmonic — not secretly linear
    e_embra, e_copy = 1.0, 1.5

    drift = energy_drift(ham, e_embra)
    pairs = make_pairs(ham, e_embra, e_copy, n_pairs=400, seed=0)
    metrics = evaluate(ham, pairs, e_embra)

    print("=" * 66)
    print("  Conserved-ψ core — phase-one sandbox")
    print("=" * 66)
    print(f"  Hamiltonian: m={ham.m}, k={ham.k}, eps={ham.eps}   (Duffing well)")
    print(f"  genesis charge   E_embra = {e_embra}")
    print(f"  replica charge   E_copy  = {e_copy}")
    print("-" * 66)
    print(f"  [1] conservation: energy relative drift  = {drift['rel_drift']:.3e}  (→ 0)")
    print(f"  [2] ψ = conserved charge   → replica AUC = {metrics['auc_psi_conserved']:.3f}  (→ 1.0)")
    print(f"  [3] ψ = endpoint (π only)  → replica AUC = {metrics['auc_endpoint']:.3f}  (→ 0.5, null)")
    print(f"      endpoint erasure (max |Δπ|)          = {metrics['endpoint_erasure']:.2e}  (→ 0)")
    print("-" * 66)
    verdict = (
        metrics["auc_psi_conserved"] > 0.99
        and abs(metrics["auc_endpoint"] - 0.5) < 1e-9
        and drift["rel_drift"] < 1e-2
    )
    print(f"  RESULT: conserved ψ survives the replica test where the endpoint cannot"
          f"  →  {'PASS' if verdict else 'FAIL'}")
    make_figure(ham, e_embra, e_copy, drift)
    print(f"  figure written: {FIG_PATH}")
    print("=" * 66)
    return metrics


if __name__ == "__main__":
    main()
