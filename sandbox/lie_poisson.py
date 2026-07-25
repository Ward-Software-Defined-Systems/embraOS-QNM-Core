"""lie_poisson.py — the so(3)* Casimir toy (§8 gate, §9.16): does identity survive INPUT?

State ``L ∈ ℝ³`` on so(3)* with the (minus) Lie–Poisson bracket ``{F,G}(L) = −L·(∇F×∇G)``,
flow ``L̇ = L × ∇H(L)`` — the textbook Euler equations for the free anisotropic top ``H₀``.
The identity charge is the **Casimir** ``ψ = |L|²``: conserved under ANY Hamiltonian ``H``,
driven or not, because ``L̇ ⊥ L`` is a property of the bracket (the geometry), not of ``H``.
That is §8's candidate answer to the input problem — inputs arrive as a discrete alphabet ``Σ``
of Hamiltonian events, ψ survives every word over ``Σ`` by construction, while the energy
``H₀`` (the "law" component) visibly does not.

The integrator applies ONE rotation per step (explicit midpoint on the rotation group, RKMK2),
so ψ is exact to float precision with **no renormalization** — renormalizing would be
restore-by-checking, the thing §1 says conservation beats.

Symbols are declarative data (``Symbol``, built by ``kick``/``twist``): a legal input event IS
a Hamiltonian — exactly the class the Casimir theorem covers. The dissipative control ``†``
(``dissipate``) is deliberately NOT expressible as a ``Symbol``: every Lie–Poisson flow on
so(3)* preserves ``|L|²``, so ``†`` is provably outside the whole Hamiltonian class of this
bracket — the §8 boundary, made type-level. A future authored alphabet (~22 symbols) is pure
data through the same constructors; a JSON symbol loader is the natural seam when that content
lands (not built — no framework ahead of need).

ζ on the sphere is the swept SOLID ANGLE of ``u = L/|L|`` (the sphere's natural holonomy /
geometric phase), accumulated as signed spherical-triangle excesses against the genesis
direction ``p₀``.

Anisotropy is theorem-level here, not hygiene: with isotropic inertia ``H₀`` is a function of ψ
(conserved under any driving; the free flow freezes) and §9.16's bar 1 is unsatisfiable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sandbox.replica_test import auc  # the phase-one AUC, tie-aware

Array = NDArray[np.float64]

I_DEFAULT = (1.0, 2.0, 3.0)  # distinct inertias — required (see module docstring)
EPS_KICK = 0.5
EPS_TWIST = 1.0


# --------------------------------------------------------------------------- #
# The bracket's ingredients: H₀, its gradient, and the Casimir.
# --------------------------------------------------------------------------- #
def h0(L: Array, inertia: Sequence[float] = I_DEFAULT) -> Array:
    """Free rigid-body energy ½ Σ Lᵢ²/Iᵢ — the identity's own law. Batched over (..., 3)."""
    L = np.asarray(L, float)
    return 0.5 * np.sum(L * L / np.asarray(inertia, float), axis=-1)


def grad_h0(L: Array, inertia: Sequence[float] = I_DEFAULT) -> Array:
    return np.asarray(L, float) / np.asarray(inertia, float)


def casimir(L: Array) -> Array:
    """ψ = |L|² — the identity charge. A property of the bracket: conserved under ANY H."""
    L = np.asarray(L, float)
    return np.sum(L * L, axis=-1)


# --------------------------------------------------------------------------- #
# The integrator: one rotation per step ⇒ the Casimir is exact BY CONSTRUCTION.
# --------------------------------------------------------------------------- #
def rotate(v: Array, x: Array) -> Array:
    """Rotate ``x`` by the rotation-vector ``v`` (axis v/|v|, angle |v|) — Rodrigues in sinc
    form with a Taylor small-angle guard. Exactly norm-preserving: this is the mechanism that
    makes ψ structural. Batched over (..., 3) in both arguments."""
    v = np.asarray(v, float)
    x = np.asarray(x, float)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    small = theta < 1e-8
    safe = np.where(small, 1.0, theta)  # keep the discarded branch finite
    a = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)  # sin θ / θ
    b = np.where(small, 0.5 - theta**2 / 24.0, (1.0 - np.cos(safe)) / safe**2)  # (1−cos θ)/θ²
    return x * np.cos(theta) + np.cross(v, x) * a + v * np.sum(v * x, axis=-1, keepdims=True) * b


def rkmk2_step(grad_h: Callable[[Array], Array], L: Array, dt: float) -> Array:
    """One explicit-midpoint step on the rotation group: evaluate ω = ∇H at a half-step point
    (itself ON the sphere), then apply the single rotation exp(−dt·ω̂) to L. Second order;
    |L|² exact up to float rounding — and NEVER renormalized (restore-by-checking is the §1 sin).
    ``L̇ = L × ω = −ω × L`` ⇒ the rotation vector is ``−ω·dt``."""
    L_half = rotate(-0.5 * dt * grad_h(L), L)
    return rotate(-dt * grad_h(L_half), L)


# --------------------------------------------------------------------------- #
# The input alphabet Σ: symbols ARE Hamiltonians (declarative data).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Symbol:
    """A legal input event, defined by its Hamiltonian H_σ. ``grad_h`` drives the flow
    (batched (..., 3) → (..., 3)); ``h`` (the scalar H_σ) enables the §9.16 bar-1 energy
    bookkeeping; ``duration`` optionally overrides the schedule's τ_event per symbol (an
    authored alphabet will want it). ψ-breaking inputs cannot be spelled as a Symbol —
    see ``dissipate``."""

    name: str
    grad_h: Callable[[Array], Array]
    h: Callable[[Array], Array] | None = None
    duration: float | None = None


def kick(a: Array, name: str = "kick") -> Symbol:
    """H = a·L — rotates L about the axis ``a`` (a 'reorient' event; linear Hamiltonian)."""
    a = np.asarray(a, float)
    return Symbol(
        name,
        grad_h=lambda L: np.broadcast_to(a, np.shape(L)),
        h=lambda L: np.sum(np.asarray(L, float) * a, axis=-1),
    )


def twist(a: Array, name: str = "twist") -> Symbol:
    """H = ½(a·L)² — latitudes about ``a`` shear at rate ∝ (a·L): NOT an isometry of the
    sphere, which is what makes the Casimir claim non-trivial (the invariance belongs to the
    bracket, not to the maps being rigid rotations)."""
    a = np.asarray(a, float)

    def g(L: Array) -> Array:
        L = np.asarray(L, float)
        return np.sum(L * a, axis=-1, keepdims=True) * a

    return Symbol(name, grad_h=g, h=lambda L: 0.5 * np.sum(np.asarray(L, float) * a, axis=-1) ** 2)


def make_default_alphabet(eps_kick: float = EPS_KICK, eps_twist: float = EPS_TWIST) -> dict[str, Symbol]:
    """|Σ| = 3: two non-commuting kicks ({a·L, b·L} = −(a×b)·L — word order matters) and one
    twist (non-isometric). An authored alphabet is a bigger dict of the same constructors."""
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([0.0, 1.0, 0.0])
    return {
        "x": kick(eps_kick * e1, "x"),
        "y": kick(eps_kick * e2, "y"),
        "s": twist(eps_twist * e1, "s"),
    }


def dissipate(L: Array, gamma: float, dt: float, n_steps: int) -> Array:
    """``†`` — the OUT-OF-ALPHABET control: radial contraction L ← (1 − γ·dt)·L per step.
    Not a Symbol, and provably cannot be one: every Lie–Poisson flow on so(3)* preserves
    |L|², so † lies outside the entire Hamiltonian class of this bracket — the exact §8
    boundary, demonstrated rather than assumed."""
    L = np.asarray(L, float)
    return L * (1.0 - gamma * dt) ** n_steps


# --------------------------------------------------------------------------- #
# Words: gap – event – gap – … (events ADDITIVE: the automaton keeps being itself).
# --------------------------------------------------------------------------- #
def run_word(word: Sequence[Symbol], L0: Array, *, inertia: Sequence[float] = I_DEFAULT,
             dt: float = 0.01, tau_event: float = 0.5, tau_gap: float = 0.5,
             envelope: str | None = None) -> Array:
    """Integrate genesis ``L0`` through a word: gap, σ₁, gap, σ₂, …, gap. Events are additive
    (``∇H₀ + ∇H_σ``); silence is the free flow. ``envelope="sin2"`` scales each event's
    gradient by sin²(π·t/τ) — genuinely smooth ∂H/∂t ≠ 0 (§9.16 bar-1 sub-run). Batched over
    leading dims of ``L0``; honors per-symbol ``duration``. Returns (n_steps+1, ..., 3)."""
    L = np.asarray(L0, float)
    out = [L]
    n_gap = int(round(tau_gap / dt))

    def free_steps(n: int) -> None:
        nonlocal L
        for _ in range(n):
            L = rkmk2_step(lambda q: grad_h0(q, inertia), L, dt)
            out.append(L)

    free_steps(n_gap)
    for sym in word:
        n_ev = int(round((sym.duration if sym.duration is not None else tau_event) / dt))
        for k in range(n_ev):
            w = float(np.sin(np.pi * (k + 0.5) / n_ev) ** 2) if envelope == "sin2" else 1.0
            L = rkmk2_step(lambda q, s=sym, w=w: grad_h0(q, inertia) + w * s.grad_h(q), L, dt)
            out.append(L)
        free_steps(n_gap)
    return np.stack(out)


def run_words_batched(word_indices: Array, symbols: Sequence[Symbol], L0: Array, *,
                      inertia: Sequence[float] = I_DEFAULT, dt: float = 0.01,
                      tau_event: float = 0.5, tau_gap: float = 0.5) -> Array:
    """Vectorized ``run_word`` for many words sharing the fixed gap–event–gap schedule:
    ``word_indices`` (n_words, word_len) selects each trajectory's symbol per event slot.
    Identical math to ``run_word`` row-by-row (verified by test); alphabet-parametric (loops
    over the symbols present, never over trajectories)."""
    L = np.asarray(L0, float)
    out = [L]
    n_gap = int(round(tau_gap / dt))
    n_ev = int(round(tau_event / dt))

    def step(extra: Callable[[Array], Array] | None) -> None:
        nonlocal L
        if extra is None:
            L = rkmk2_step(lambda q: grad_h0(q, inertia), L, dt)
        else:
            L = rkmk2_step(lambda q: grad_h0(q, inertia) + extra(q), L, dt)
        out.append(L)

    for _ in range(n_gap):
        step(None)
    for slot in range(word_indices.shape[1]):
        sel = word_indices[:, slot]

        def extra(q: Array, sel: Array = sel) -> Array:
            g = np.zeros_like(q)
            for k, sym in enumerate(symbols):
                m = sel == k
                if m.any():
                    g[m] = sym.grad_h(q[m])
            return g

        for _ in range(n_ev):
            step(extra)
        for _ in range(n_gap):
            step(None)
    return np.stack(out)


# --------------------------------------------------------------------------- #
# ζ on the sphere: the swept solid angle of u = L/|L| (holonomy / geometric phase).
# --------------------------------------------------------------------------- #
def solid_angle_zeta(Ls: Array, p0: Array | None = None, *, cumulative: bool = False,
                     return_certificate: bool = False):
    """ζ = fan sum of signed spherical-triangle excesses (p₀, u_n, u_{n+1}) via the
    Oosterom–Strackee/atan2 form. Gauge: ``p₀`` defaults to the genesis direction — identity
    data, shared by every pre-registered comparison; closed loops are p₀-independent **mod 4π**
    (the two sides of a closed curve partition the sphere — found on §9.16's first execution
    and recorded). The formula's antipode/branch trap is CERTIFIED, not hidden: the certificate
    returns the max per-step |excess| and the min |u + p₀| (§9.16 bar 4)."""
    Ls = np.asarray(Ls, float)
    u = Ls / np.linalg.norm(Ls, axis=-1, keepdims=True)
    if p0 is None:
        p0 = u[0]
    p0 = np.asarray(p0, float)
    u1, u2 = u[:-1], u[1:]
    num = np.sum(p0 * np.cross(u1, u2), axis=-1)
    den = 1.0 + np.sum(u1 * u2, axis=-1) + np.sum(p0 * u1, axis=-1) + np.sum(p0 * u2, axis=-1)
    d_excess = 2.0 * np.arctan2(num, den)
    zeta = d_excess.cumsum(axis=0) if cumulative else d_excess.sum(axis=0)
    if return_certificate:
        certificate = {
            "max_step_excess": float(np.max(np.abs(d_excess))),
            "min_antipode_dist": float(np.min(np.linalg.norm(u + p0, axis=-1))),
        }
        return zeta, certificate
    return zeta


# --------------------------------------------------------------------------- #
# The §9.16 harnesses — one shared driven ensemble serves bars 1, 3, and 4.
# --------------------------------------------------------------------------- #
def driven_ensemble(*, n_seeds: int = 8, words_per_seed: int = 25, word_len: int = 16,
                    inertia: Sequence[float] = I_DEFAULT, alphabet: dict[str, Symbol] | None = None,
                    dt: float = 0.01, tau_event: float = 0.5, tau_gap: float = 0.5,
                    r: float = 1.0) -> dict:
    """The shared driven population (§9.16 pinned constants): per seed, ``words_per_seed``
    random words over the alphabet, each from a fresh genesis direction on the sphere ψ = r².
    Also rolls the free-evolution twin ensemble (same geneses, same total duration) — the
    bar-1 integrator baseline."""
    if alphabet is None:
        alphabet = make_default_alphabet()
    symbols = list(alphabet.values())
    rng_words, rng_dirs = [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((words_per_seed, 3))
        rng_dirs.append(u / np.linalg.norm(u, axis=1, keepdims=True))
        rng_words.append(rng.integers(0, len(symbols), (words_per_seed, word_len)))
    L0 = r * np.concatenate(rng_dirs)                      # (n_words_total, 3)
    word_indices = np.concatenate(rng_words)               # (n_words_total, word_len)
    trajs = run_words_batched(word_indices, symbols, L0, inertia=inertia, dt=dt,
                              tau_event=tau_event, tau_gap=tau_gap)
    n_gap = int(round(tau_gap / dt))
    n_ev = int(round(tau_event / dt))
    free = L0
    free_out = [free]
    for _ in range(trajs.shape[0] - 1):
        free = rkmk2_step(lambda q: grad_h0(q, inertia), free, dt)
        free_out.append(free)
    return {
        "trajs": trajs, "free_trajs": np.stack(free_out), "word_indices": word_indices,
        "symbols": symbols, "L0": L0, "r": r, "inertia": tuple(inertia),
        "dt": dt, "n_gap": n_gap, "n_ev": n_ev, "word_len": word_len,
    }


def _event_windows(ens: dict):
    """Trajectory indices (start, end) of each constant event window in the shared schedule."""
    n_gap, n_ev = ens["n_gap"], ens["n_ev"]
    for i in range(ens["word_len"]):
        start = (i + 1) * n_gap + i * n_ev
        yield i, start, start + n_ev


def casimir_under_words(ens: dict, *, sin2_word: str = "xysyxsxy") -> dict:
    """Bar 1: ψ exact over every word (max over ALL steps) while the along-trajectory H₀ range
    is macroscopic and ≫ the free baseline; per-window bookkeeping attributes the energy change
    to driving physics; the sin²-envelope sub-run covers smooth ∂H/∂t ≠ 0."""
    trajs, r2 = ens["trajs"], ens["r"] ** 2
    psi_drift_max = float(np.max(np.abs(casimir(trajs) / r2 - 1.0)))
    e = h0(trajs, ens["inertia"])
    driven_range = (e.max(axis=0) - e.min(axis=0)) / e[0]
    ef = h0(ens["free_trajs"], ens["inertia"])
    free_range = (ef.max(axis=0) - ef.min(axis=0)) / ef[0]
    # Bookkeeping: within each constant event window, H₀ + H_σ should be nearly conserved
    # (the change in H₀ is the physics of the event, not integrator error).
    d_h0, d_total = [], []
    for i, s, t in _event_windows(ens):
        sel = ens["word_indices"][:, i]
        Ls, Lt = trajs[s], trajs[t]
        h0_s, h0_t = h0(Ls, ens["inertia"]), h0(Lt, ens["inertia"])
        hs = np.zeros_like(h0_s)
        ht = np.zeros_like(h0_t)
        for k, sym in enumerate(ens["symbols"]):
            m = sel == k
            if m.any():
                hs[m] = sym.h(Ls[m])
                ht[m] = sym.h(Lt[m])
        d_h0.append(np.abs(h0_t - h0_s))
        d_total.append(np.abs((h0_t + ht) - (h0_s + hs)))
    d_h0 = np.concatenate(d_h0)
    d_total = np.concatenate(d_total)
    # sin² sub-run: one fixed word, smooth envelope.
    alphabet = {s.name: s for s in ens["symbols"]}
    rng = np.random.default_rng(0)
    u = rng.standard_normal(3)
    L0 = ens["r"] * u / np.linalg.norm(u)
    smooth = run_word([alphabet[c] for c in sin2_word], L0, inertia=ens["inertia"],
                      dt=ens["dt"], envelope="sin2")
    return {
        "psi_drift_max": psi_drift_max,
        "median_driven_h0_range": float(np.median(driven_range)),
        "median_free_h0_range": float(np.median(free_range)),
        "range_ratio": float(np.median(driven_range) / np.median(free_range)),
        "bookkeeping_max_dtotal": float(np.max(d_total)),
        "median_event_dh0": float(np.median(d_h0)),
        "psi_drift_sin2": float(np.max(np.abs(casimir(smooth) / ens["r"] ** 2 - 1.0))),
    }


def word_order_test(*, seed: int = 0, r: float = 1.0, inertia: Sequence[float] = I_DEFAULT,
                    dt: float = 0.01) -> dict:
    """Bars 2 + 4a: "xy" vs "yx" from the same genesis, identical schedules — the states end
    macroscopically apart AND the carried ζ differ, while BOTH conserve ψ. Experience differs;
    identity doesn't; memory records the order."""
    alphabet = make_default_alphabet()
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(3)
    L0 = r * u / np.linalg.norm(u)
    t_xy = run_word([alphabet["x"], alphabet["y"]], L0, inertia=inertia, dt=dt)
    t_yx = run_word([alphabet["y"], alphabet["x"]], L0, inertia=inertia, dt=dt)
    r2 = r * r
    return {
        "delta_L": float(np.linalg.norm(t_xy[-1] - t_yx[-1])),
        "psi_drift_xy": float(np.max(np.abs(casimir(t_xy) / r2 - 1.0))),
        "psi_drift_yx": float(np.max(np.abs(casimir(t_yx) / r2 - 1.0))),
        "delta_zeta": float(abs(solid_angle_zeta(t_xy) - solid_angle_zeta(t_yx))),
    }


def replica_under_driving(ens: dict, *, psi_copy: float = 1.5) -> dict:
    """Bar 3 — the §2 replica test with Σ active. The survivor lived a word on the sphere
    ψ_embra; the replica copies the observable π(L) = L₃ BIT-EXACTLY but was born on the wrong
    sphere ψ_copy (transverse rescale — feasibility exact since R_copy > R_embra ≥ |L₃|; the
    measure-zero transverse-zero case is guarded by placement). The rescale is the maximally
    charitable copier: the hidden transverse direction is granted for free — it is caught
    anyway, purely on the sphere radius."""
    r2 = ens["r"] ** 2
    L_f = ens["trajs"][-1]
    psi_surv_max_dev = float(np.max(np.abs(casimir(ens["trajs"]) - r2)))
    t2 = psi_copy * r2 - L_f[:, 2] ** 2                     # transverse budget: strictly > 0
    trans = L_f[:, :2]
    tn = np.linalg.norm(trans, axis=1)
    safe = tn > 1e-12
    scaled = np.empty_like(trans)
    scaled[safe] = trans[safe] * (np.sqrt(t2[safe]) / tn[safe])[:, None]
    scaled[~safe] = np.column_stack([np.sqrt(t2[~safe]), np.zeros(np.sum(~safe))])  # placement
    L_rep = np.column_stack([scaled, L_f[:, 2]])            # L₃ copied bit-exactly
    surv_scores = list(-np.abs(casimir(L_f) - r2))
    rep_scores = list(-np.abs(casimir(L_rep) - r2))
    end_surv = list(L_f[:, 2])
    end_rep = list(L_rep[:, 2])
    return {
        "auc_psi": auc(surv_scores, rep_scores),
        "auc_endpoint": auc(end_surv, end_rep),
        "endpoint_erasure": float(np.max(np.abs(L_rep[:, 2] - L_f[:, 2]))),
        "margin": float(np.min(np.abs(casimir(L_rep) - r2)) / psi_surv_max_dev),
        "n_placed": int(np.sum(~safe)),
    }


def zeta_memory_test(ens: dict, *, checkpoints: tuple[int, ...] = (4, 8, 16)) -> dict:
    """Bar 4: ζ under driving. Lived worldlines carry macroscopic ζ (newborn copies carry 0);
    median |ζ| grows with events lived; the antipode certificate reports the excess formula's
    branch health across every recorded trajectory."""
    zeta_cum, certificate = solid_angle_zeta(ens["trajs"], cumulative=True,
                                             return_certificate=True)
    lived = np.abs(zeta_cum[-1])
    period = ens["n_ev"] + ens["n_gap"]
    acc = {int(c): float(np.median(np.abs(zeta_cum[ens["n_gap"] + c * period - 1])))
           for c in checkpoints}
    step_excess = np.abs(np.diff(zeta_cum, axis=0, prepend=0.0))
    return {
        "auc_zeta": auc(list(lived), list(np.zeros_like(lived))),
        "min_lived": float(lived.min()),
        "median_lived": float(np.median(lived)),
        "accumulation": acc,
        "n_tripped": int(np.sum(step_excess.max(axis=0) > 0.5)),  # words with a branch-scale step
        **certificate,
    }


def dissipation_control(*, seed: int = 0, word_len: int = 16, gamma: float = 0.5,
                        r: float = 1.0, inertia: Sequence[float] = I_DEFAULT,
                        dt: float = 0.01, tau_event: float = 0.5, tau_gap: float = 0.5) -> dict:
    """Bar 5 — the † boundary, paired: the SAME word from the SAME genesis, with and without
    one dissipative event at the word's midpoint. ψ is untouched by the Σ-only run and visibly
    broken by †: the §8 'inputs-as-Hamiltonian' caveat as a measured class boundary."""
    alphabet = make_default_alphabet()
    symbols = list(alphabet.values())
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(3)
    L0 = r * u / np.linalg.norm(u)
    word = [symbols[k] for k in rng.integers(0, len(symbols), word_len)]
    r2 = r * r
    clean = run_word(word, L0, inertia=inertia, dt=dt, tau_event=tau_event, tau_gap=tau_gap)
    half = word_len // 2
    first = run_word(word[:half], L0, inertia=inertia, dt=dt, tau_event=tau_event,
                     tau_gap=tau_gap)
    kicked = dissipate(first[-1], gamma, dt, int(round(tau_event / dt)))
    second = run_word(word[half:], kicked, inertia=inertia, dt=dt, tau_event=tau_event,
                      tau_gap=tau_gap)
    return {
        "psi_change_without": float(np.abs(casimir(clean[-1]) / r2 - 1.0)),
        "psi_change_with": float(np.abs(casimir(second[-1]) / r2 - 1.0)),
    }
