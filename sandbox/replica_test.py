"""replica_test.py — the survivor-vs-replica test on the toy dynamics.

The Epoch bar (from the relic's ``EPOCH-INVARIANT-GROUNDING`` / this repo's
``docs/CORE-SPEC.md``): a real ψ is *trajectory-dependent* and survives the
**replica test** — two runs reach the same *observable* endpoint by different
paths, one a **survivor** (carried the sealed charge from genesis ``s₀``) and one
a **replica** (instantiated fresh to match the readout, never inheriting ``s₀``).
A static, endpoint-only ψ calls them identical; the conserved-charge ψ tells them
apart because the charge lives in the hidden complement of the observable.

Here:
  * survivor  = a state reached by integrating from the genesis orbit ``Q = E_embra``
  * replica   = a state at the *same observable position* but on the wrong energy
                level ``Q = E_copy`` — a valid state of the same dynamics that merely
                looks identical at the readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sandbox.toy_dynamics import (
    Hamiltonian,
    momentum_for_energy,
    observable,
    rollout,
    turning_point,
)

State = tuple[float, float]  # (q, p)


@dataclass(frozen=True)
class ReplicaPair:
    """A survivor and its endpoint-matched replica. ``π(survivor) == π(replica)`` by
    construction (they share position ``q_f``); only the hidden charge differs."""

    survivor: State  # (q_f, p) with Q ≈ E_embra  (carried from genesis)
    replica: State  # (q_f, p') with Q  = E_copy  (fresh copy, same observable)


def make_pairs(
    ham: Hamiltonian,
    e_embra: float,
    e_copy: float,
    n_pairs: int,
    *,
    dt: float = 0.01,
    seed: int = 0,
    max_steps: int = 2000,
) -> list[ReplicaPair]:
    """Generate ``n_pairs`` survivor/replica pairs.

    ``e_copy`` must exceed ``e_embra`` so the copy's (higher-energy) orbit reaches
    every survivor position ``q_f`` — otherwise the copy could be *told apart by the
    observable alone* (it couldn't be there), which would cheat the endpoint blindness.
    """
    if e_copy <= e_embra:
        raise ValueError("e_copy must be > e_embra so the replica orbit reaches every survivor position")
    rng = np.random.default_rng(seed)
    q_start = turning_point(ham, e_embra)  # p = 0 on the genesis orbit
    pairs: list[ReplicaPair] = []
    for _ in range(n_pairs):
        n = int(rng.integers(1, max_steps))
        qs, ps = rollout(ham, q_start, 0.0, dt, n)
        q_f, p_surv = float(qs[-1]), float(ps[-1])  # survivor endpoint, Q ≈ e_embra
        # Replica: same observable q_f, placed on the wrong energy level. Match the
        # survivor's direction of travel so even the *sign* of motion is shared —
        # the copy is indistinguishable at the readout in every observable respect.
        sign = 1.0 if p_surv >= 0 else -1.0
        p_rep = float(momentum_for_energy(ham, e_copy, q_f, sign=sign))
        pairs.append(ReplicaPair((q_f, p_surv), (q_f, p_rep)))
    return pairs


# --------------------------------------------------------------------------- #
# Readers. A "reader" scores a state for how much it looks like the survivor
# (higher = more Embra). The whole result is the gap between two readers.
# --------------------------------------------------------------------------- #
def psi_conserved_score(ham: Hamiltonian, state: State, e_embra: float) -> float:
    """ψ as the conserved charge: ``-|Q(s) − E_embra|``. Reads the FULL state
    (including the hidden momentum). This is the native, trajectory-dependent ψ."""
    q, p = state
    return -abs(float(ham.energy(q, p)) - e_embra)


def psi_holds(ham: Hamiltonian, state: State, e_embra: float, *, tol: float = 1e-3) -> bool:
    """The ψ predicate: identity holds iff the conserved charge matches genesis.

    Falsifiable by construction — a state off the E_embra orbit returns ``False``.
    """
    q, p = state
    return abs(float(ham.energy(q, p)) - e_embra) <= tol


def endpoint_score(state: State, ref_q: float | None = None) -> float:
    """A best-effort reader restricted to the observable π(s) = q.

    Because survivor and replica share ``q_f`` by construction, *any* function of the
    observable returns equal scores within a pair → AUC 0.5. This is the certified
    null: the endpoint carries no information distinguishing the pair.
    """
    q, _ = state
    q = float(q)
    return -abs(q - ref_q) if ref_q is not None else q


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def auc(pos: list[float], neg: list[float]) -> float:
    """AUC / Mann–Whitney U: P(pos ranked above neg), ties counted as 0.5.

    1.0 = perfect separation; 0.5 = indistinguishable (chance).
    """
    p = np.asarray(pos, float)[:, None]
    n = np.asarray(neg, float)[None, :]
    gt = np.sum(p > n)
    eq = np.sum(p == n)
    return float((gt + 0.5 * eq) / (p.size * n.size))  # p.size·n.size total comparisons


def endpoint_erasure(pairs: list[ReplicaPair]) -> float:
    """max ``|π(survivor) − π(replica)|`` over pairs.

    Must be ≈ 0 for the test to be a legitimate *endpoint-blind* replica test — the
    V1-certification analog from the relic's harness. If this is nonzero, the reader
    could cheat by reading the observable.
    """
    return max(
        abs(float(observable(*p.survivor)) - float(observable(*p.replica))) for p in pairs
    )


def evaluate(ham: Hamiltonian, pairs: list[ReplicaPair], e_embra: float) -> dict[str, float]:
    """Run both readers over the pairs and return the headline numbers."""
    surv_psi = [psi_conserved_score(ham, p.survivor, e_embra) for p in pairs]
    rep_psi = [psi_conserved_score(ham, p.replica, e_embra) for p in pairs]
    surv_end = [endpoint_score(p.survivor) for p in pairs]
    rep_end = [endpoint_score(p.replica) for p in pairs]
    return {
        "auc_psi_conserved": auc(surv_psi, rep_psi),
        "auc_endpoint": auc(surv_end, rep_end),
        "endpoint_erasure": endpoint_erasure(pairs),
        "n_pairs": float(len(pairs)),
    }
