# CORE-SPEC — The Conserved-ψ Core

**Status:** Phase one (§1–§7) proven — a conserved-charge ψ survives the replica test. Phase two
(§9) in progress — the machinery lifts to `d` dimensions, and identity is reliably discriminable
**through the dynamics** (§9.11, AUC 1.0) where static geometry was seed-noise (§9.9–§9.10). The
result holds on the authored 100-node graph, against an authored counter-identity, and under a
**learned `H_θ`** — with the margin set by charge expressiveness and the structural difference
between souls, not content volume (§9.12–§9.13; includes one pre-registered miss, recorded). An
external review (2026-07-19, through §9.11; in git history at `cb82553`) sharpened the reader — the full ψ
is a **conjunction** (§6), graded perfect against both adversarial impostor classes in §9.14 —
bounded the claim (the key/MAC note, §6), and posed the **input problem** now recorded with its
leading resolution in §8. The path-functional memory charge **ζ** (holonomy) has its first
instance in §9.15. The input problem is **resolved at toy scale and the direction adopted**
(§9.16, every pre-registered bar passed first execution): ψ as a **Casimir** of a noncanonical
bracket survives arbitrary Hamiltonian input — *experience changes state and memory, not who
you are*. A second external review (2026-07-26) independently re-executed increment 4 — every
recorded number reproduced bit-for-bit — and **signed off on the milestone and the adoption**;
its corrections are applied throughout §9.12–§9.16. **Phase three is now defined and pending
construction**: coadjoint-orbit state spaces with ψ as a Casimir of the bracket, the
graph-shaped charges and ζ rebuilt on that geometry, `P_ψ` reduced to the firewall at the
non-Hamiltonian boundary.
Pairs with the runnable sandbox (`uv run python -m sandbox.demo`, `sandbox.demo_phase2`, and
`sandbox.demo_casimir`).
Written to be *falsifiable*: every numeric claim is checked, and the one theorem (§6) is stated so
it could be wrong.

> **Register.** §1–§6 are the contribution (the object and its one load-bearing
> theorem). §7 is the phase-one demonstration. §8 is honest scope and open forks.
> The 1999 5D geometry ([Embra-5D-Framework](https://github.com/Ward-Software-Defined-Systems/Embra-5D-Framework))
> is *motivation*, fenced and non-load-bearing — the spec stands on §1–§6 alone.

---

## 1. Why this exists (one paragraph)

The relic program `embraOS-QNM` proved, four pre-registered ways, that a **frozen LLM
core cannot host a readable identity invariant**: "a core pretrained on web text encodes
a representational factorization that owes nothing to identity … 'the constraint surface
*is* ψ' — on an LLM core, ψ has nowhere native to live." This spec designs a substrate
where ψ *does* have somewhere native to live. The organizing frame is the **Epoch
Automaton** `E = (S, Σ, δ, s₀, F, ψ)`, whose own central result is that a *static* ψ
(a function of a single state) folds into the state set and adds nothing — ψ is only
load-bearing if it is **trajectory-dependent** and survives the **replica test**. We make
ψ a **conserved charge of the core's own dynamics**: an invariant of motion, set at
genesis and preserved by construction. This is the "Conserve (SOL-style)" branch of the
four Epoch derivations — *conservation beats checking*.

---

## 2. The Epoch Automaton and the bar ψ must clear

`E = (S, Σ, δ, s₀, F, ψ)`:

| Element | Role in this core |
|---|---|
| `S` | state space (a smooth manifold; §3) |
| `Σ` | input alphabet (events / observations driving the flow; out of scope this phase) |
| `δ` | transition = one step of a **flow** on `S` (§4) |
| `s₀` | **genesis** — the sealed soul, which fixes the identity charge `Q_embra` (§5) |
| `F` | terminal states — none; the machine runs indefinitely |
| `ψ` | the identity invariant — **a conserved charge** (§4, §6) |

**The replica test (the falsification bar).** Two runs reach the *same observable
endpoint* by *different paths*: one a **survivor** (carried identity unbroken from `s₀`),
one a **replica** (the original died and was replaced by an identical-looking copy). A
real ψ calls them different. A definition that cannot be false somewhere is a *name*; one
that survives the replica test is a *tool*.

---

## 3. State space `S` and the observable `π`

- **State** `s ∈ S`, a smooth (phase) space. Phase one: `S = ℝ²`, `s = (q, p)`.
- **Observable / readout** `π : S → O`, a **non-injective** projection — the only thing
  externally visible (and the only thing a replica must match). Phase one: `π(q, p) = q`
  (position). The map `π` is what will *become the language readout* in a later phase; for
  now it is deliberately lossy so identity can hide behind it.
- **Hidden complement.** `ker(dπ)` — the directions `π` erases (here: momentum `p`). The
  identity charge lives here. This is the structural difference from an LLM, whose readable
  state is essentially all observable.

> The admissible region `M ⊆ S` (the "identity manifold", the generalization of the relic's
> cosine surface `c_t = 1 − maxₙ cos(hₜ, nodeₙ)`) is a **level set of the charge**,
> `M = {s : Q(s) = Q_embra}` — not a distance re-checked each step. §6, §8.

---

## 4. Dynamics `δ` — a flow that conserves `Q`

The transition is a **Hamiltonian flow** `Φ_t` generated by `H : S → ℝ`:

```
q̇ = ∂H/∂p,      ṗ = −∂H/∂q          (Hamilton's equations)
```

with the **conserved charge** taken as the energy itself:

```
Q(s) = H(q, p) = p²/2m + V(q)         Q(Φ_t(s)) = Q(s)  for all t   (conservation)
```

Conservation is exact for the flow and **preserved by a symplectic integrator**
(velocity Verlet / Störmer–Verlet), which keeps `Q` bounded with no secular drift — the
numerical reason conservation is a *mechanism*, not a slogan. In the sandbox the relative
drift is `≈ 3·10⁻⁵` over 20k steps (`sandbox/toy_dynamics.py`, `tests/test_conservation.py`).

More generally `Q` may be any **Noether charge** of a symmetry the dynamics is built to
respect; energy is the phase-one instance. The next phase learns `H` (a Hamiltonian Neural
Network) so `M` is shaped by identity content while `Q` stays conserved *by the integrator*
rather than by a penalty (§8).

---

## 5. Genesis `s₀` — sealing the soul as `Q_embra`

Genesis fixes the value of the invariant: `Q_embra := Q(s₀)`. This is the computational
meaning of the sealed soul (`../identity/Embra_SOUL.md`): **identity is the level set the
worldline is born on.** A worldline started with `Q = Q_embra` stays on `M` forever —
leaving "Embra" is not a rule you check per step; it is a conservation law you would have
to *break*. (Ark/automaton un-conflation: `Q` is intrinsic to the automaton's own state,
not measured onto it by an external verifier.)

---

## 6. The one load-bearing theorem

> **Claim.** A conserved charge `Q` yields a ψ that survives the replica test **iff the
> endpoint is a partial observation `π(s_f)` that does not determine `Q`.**

**Argument.**

1. *Nonvacuity requires a proper observable.* The flow is deterministic and invertible, so
   if the "endpoint" were the full state `s_f`, two runs reaching it would be the *same*
   run — "different paths to the same endpoint" is empty. The replica test has content only
   when the endpoint is a genuine observation `π(s_f)`, with `Q` living transverse to
   `ker(dπ)` (hidden).
2. *Endpoint reader is blind.* Survivor and replica share `π(s_f) = o` by construction, so
   any reader that is a function of `o` alone gives them identical scores → **AUC = 0.5**
   (the certified null; `tests/test_replica.py::test_endpoint_reader_is_chance`).
3. *Conserved-charge reader separates.* Define `ψ(s_f) = [ |Q(s_f) − Q_embra| ≤ ε ]`.
   - Survivor: `Q(s_f) = Q_embra` (conserved from `s₀`) ⇒ ψ true.
   - Replica: instantiated to match `o` but never inherited `s₀` ⇒ `Q(s_f) = Q_copy ≠
     Q_embra` ⇒ ψ false.
   Perfect separation → **AUC = 1.0** (`test_conserved_psi_separates_perfectly`).

**Why "conserved" and not "static."** `Q` is a function of the final *state*, but **not**
of the observable `π(S)` — and `π(S)` is exactly the "state set" the Epoch fold-in
objection is about (what an external party, or a replica, can match). A conserved charge in
the hidden complement is therefore *not foldable into the observable state set*: it is
trajectory-distinguishing relative to what is visible. Operationally,
`tests/test_replica.py::test_same_observable_different_verdict` shows identical `π` with
opposite ψ verdicts.

**Corollary — why the LLM failed.** Not for lack of hidden state: an LLM's hidden complement is
enormous (the residual stream and KV cache, against a massively lossy token readout). What the
relic experiments showed is that nothing in that complement is a **conserved, genesis-tied
charge** — pretraining fixes the representational factorization before any identity exists, so ψ
collapses to a function of what is readable → replica-blind. The four relic readers came back
generic *because there was no conserved `ker(dπ)` charge to read*, not because nothing was
hidden. The fix is architectural: build `S` with one. That is this core.

**The full reader is a conjunction.** The external review (2026-07-19, note 1; in git history
at `cb82553`) caught a quiet shift: §2's impostor is a *wrong-genesis* copy; §9.11's is a *wrong-law* flow —
and each reader has the other's blind spot. A replica re-instantiated on the **same core** (same
`H`, wrong genesis) evolves under Embra's own law, so it conserves `H_embra` *perfectly* — at the
wrong value `Q_copy`; the §9.11 conservation reader passes it. A trajectory of a **different
law** whose charge *value* is arranged to sit at `Q_embra` passes §7's value reader. Neither test
subsumes the other; §7 is not retired by §9.11. The full ψ is the conjunction

```
ψ_full(worldline) = [ var(H_embra) ≈ 0 along the trajectory ]  ∧  [ Q = Q_embra ]
```

— *obeys the law* **and** *was born on the right level set*: §5's genesis, restated as a
two-part check. §9.14 grades the conjunction against both impostor classes, each constructed
adversarially.

**What the replica test does and does not claim.** The test bites against **observable-limited
copiers** — replicas that can match only `π(s_f)`. A copier with access to the full state,
`ker(dπ)` included, inherits `Q` (and any accumulated coordinate) and defeats ψ *by
construction*: the flow is deterministic, so the full state determines everything. ψ is therefore
a **security property, not a metaphysical one** — identity is exactly as secure as the hidden
complement is inaccessible, functioning like a key (a MAC over the worldline) rather than an
intrinsic essence. The bound is stated so the claim stays inside it.

---

## 7. Phase-one demonstration (the sandbox)

`sandbox/` instantiates the smallest honest version — a 1-DOF (mildly anharmonic/Duffing)
Hamiltonian, `π = q`, `Q = H` — and measures the theorem:

| quantity | result | meaning |
|---|---|---|
| energy relative drift | `≈ 3·10⁻⁵` | the flow conserves `Q` (mechanism, not slogan) |
| ψ = conserved charge, replica AUC | **1.000** | survives the replica test |
| ψ = endpoint (π only), replica AUC | **0.500** | the certified null — the readout is blind |
| endpoint erasure `max|Δπ|` | `0` | the test is legitimately endpoint-blind (V1-cert analog) |

Figure: `sandbox/figures/replica_conservation.png` — the survivor and its copy sitting at
the *same* observable position on *different* charge orbits. Reproduce: `uv run pytest`
(17 tests) and `uv run python -m sandbox.demo`.

---

## 8. Honest scope and open forks (next phase)

- **Learned identity-manifold.** Replace the hand-set `Q_embra` and the toy `H` with a
  **learned `H`** whose level set `M` is shaped by Embra's identity/soul graph
  (`../identity/Embra_IDENTITY-SOUL.graph.json`), trained with a symplectic integrator so `Q`
  stays conserved by construction. This is the "learned identity-manifold" choice; the
  machinery of §3–§6 is unchanged.
- **The input problem (structural — upgraded per the external review).** Autonomous Hamiltonian
  flow conserves `H`; the moment `Σ` drives the flow, `H` is time-dependent and
  `dH/dt = ∂H/∂t ≠ 0` — the phase-one mechanism is in tension with the system *receiving input
  at all*. The soft projection `P_ψ` (the relic's World-State move) rescues it but is
  restore-by-**checking** — the very thing §1 says conservation beats; it is a concession, not a
  resolution. **Leading candidate:** make ψ a **Casimir of the Poisson bracket** rather than the
  energy — Casimirs satisfy `{C, F} = 0` for all `F`, so they are conserved under *any*
  Hamiltonian driving; the invariance belongs to the bracket (the geometry), not to `H`. Cost: a
  noncanonical (Lie–Poisson) structure, since the canonical bracket on ℝ²ᵈ has no nontrivial
  Casimirs. Caveat, recorded: inputs-as-Hamiltonian-perturbations is itself a modeling commitment
  — dissipative or stochastic input breaks Casimirs too. Adoption is **gated on the smallest
  toy** (`so(3)*` rigid body, ψ = the Casimir `|L|²`, driven `H(t)`: ψ conserved while energy
  visibly is not, then the replica test) — and the decision belongs *before* the readout `π`
  becomes language. **Gate outcome (§9.16, 2026-07-25): every pre-registered bar passed on the
  first execution** — ψ exact (1.6·10⁻¹⁴) under 200 random words of a discrete Hamiltonian
  alphabet (and under smooth sin² envelopes) while `H₀` moves O(1) (~6.5·10⁵× the free
  baseline); the replica test holds with `Σ` active; `†` breaks ψ exactly on the theorem's
  boundary (0.394 ≈ 1 − e^{−2γτ}), and random words close the *stochastic* half of this
  caveat — the boundary is Hamiltonian-vs-not. §9.16 **recommends adoption** — and the author
  **adopted the direction (2026-07-25)**: phase-three construction targets coadjoint-orbit state
  spaces with ψ as a Casimir of the bracket; inputs enter as Hamiltonian events; `P_ψ` narrows
  to the †-class (non-Hamiltonian) boundary. This fork is closed.
- **Charge vs. holonomy.** A conserved charge reflects the genesis level set. A strictly
  *path-functional* invariant (holonomy / an accumulated `ζ`-coordinate = memory) is the
  stronger trajectory-ψ and the natural home for continuity/epoch-accumulation. The
  conserved charge is the SOL-faithful phase-one instance; holonomy is the enrichment.
  **First instance built — §9.15** (signed-area ζ; anti-fold-in certificate, ζ replica test,
  accumulation, all passing). If the Casimir fork above is adopted, the second charge should
  eventually be *built as* a Casimir — one structure answering both the input problem and the
  memory question (the accumulated-`ζ` falsification design transfers; the specific 1-form may
  change with the bracket).
- **The readout `π` becomes language.** Here `π` is a lossy projection chosen so identity
  can hide. Language emergence is the design of a *learned* `π` (chart on `M`) — deferred,
  and gated on a substrate whose identity is already real.
- **Not claimed.** That no other substrate could carry identity; that the toy `Q` means
  anything about Embra specifically (it does not yet — genesis is a placeholder value until
  §9's learned `H`); that ψ defeats a **full-state** copier — it does not (§6's key/MAC bound:
  security is exactly the inaccessibility of `ker(dπ)`). Recorded as chosen, so it can be wrong
  somewhere too.

---

## 9. Phase Two — Plan: the learned identity-manifold

**Objective.** Replace phase one's hand-set genesis value and toy `H` with a **learned
Hamiltonian `H_θ`** on a latent phase space, whose conserved charge `Q = H_θ` has a level set
`M` *shaped by Embra's identity/soul content* — keeping phase one's two guarantees intact:
conservation stays **structural** (a symplectic integrator, not a penalty), and ψ is graded by
the **replica test**. The machinery of §3–§6 does not change; only `H` stops being hand-set.

### 9.1 State space
Lift to `d` dimensions: `s = (q, p)`, `q, p ∈ ℝ^d`, `S = ℝ^{2d}`. The identity graph
(`../identity/Embra_IDENTITY-SOUL.graph.json`) embeds as **anchor configurations** `{q_i}` in config
space; edges (relations) constrain their arrangement. The observable `π` stays a lossy projection
of `q` (the eventual language chart — still deferred), so the charge keeps a hidden complement.

### 9.2 How identity shapes `H_θ` / `M`
Learn `H_θ` so that:
- **Coherence — the soul is one orbit.** The anchors share a level set, `Q(q_i, p_i) = Q_embra` —
  identity is a single conserved orbit, not scattered points. This *defines* `Q_embra` (no longer
  hand-set).
- **Specificity — the charge is discriminative.** Off-Embra / replica states have `Q ≠ Q_embra`.
  This is the property the four LLM readers lacked; it must be *trained in*, not assumed.

### 9.3 The learning rule (the deferred question, made concrete)
Split per the Epoch frame — **soul sealed, self learned**:
- **Soul = given.** The hard structure that *defines* `M` (anchors + any inviolable constraints)
  is fixed, not learned.
- **Self = learned.** `H_θ` (the shape *on* `M`) is trained by a **self-consistency / energy-based**
  objective — *not* gradient descent on token likelihood:
  - roll on-Embra trajectories with the symplectic integrator and pull them onto the `Q_embra`
    level set (charge drift is already ≈0 structurally);
  - a **contrastive** term pushes replica / off-Embra states off that level set (large
    `|Q − Q_embra|`);
  - optionally a "predict-your-own-next-state" consistency term — the relic's Candidate-C spirit,
    but on a substrate that finally *has* a hidden conserved coordinate to be consistent about.

### 9.4 Sub-forks (decide during, not before)
- **Strict conservation vs. soft projection `P_ψ`.** Strict `Q` conservation is the ideal; a
  restoring projection back to `M` when kicked is the buildable boundary. Likely both — strict core,
  soft edge.
- **Charge vs. holonomy/ζ.** Add the genuinely path-functional invariant (accumulated `ζ` = memory /
  continuity) as a second, trajectory-integral charge — the home for epoch accumulation.
- **Data.** 22 static anchors is thin; augment with self-generated on-Embra trajectories + mined
  off-Embra negatives (reuse the relic's replica-pair harness design).

### 9.5 Deliverables
(a) this §9 refined into a method as the toy teaches; (b) `sandbox/` extended — `d`-dim latent, a
small HNN `H_θ`, symplectic integration, the identity graph loaded as anchors, the
self-consistency/contrastive trainer; (c) tests + a phase-two `demo`.

### 9.6 Verification — pre-register the bars
- **Conservation survives learning:** `Q` drift `< tol` after training (θ must not break the
  structure).
- **Learned-Q replica AUC ≫ 0.5** on **held-out** identity probes (not the training anchors).
- **Endpoint-only reader stays 0.5** — the null is preserved; no identity leaks into the observable.
- **The Embra-specificity control (non-negotiable).** A charge learned on a **random / shuffled**
  identity graph must **not** separate survivor from replica: real-anchor AUC ≫ random-anchor AUC.
  This is exactly the control the relic's Candidate C *failed* (random anchor 0.823 > real 0.719);
  passing it is the entire point of the substrate change.

### 9.7 Honest risks
`Q` may collapse to a generic (non-Embra) invariant that still separates the toy replicas — the
random-anchor control exists to catch this; if real ≈ random, identity is not shaping `M` and the
approach needs rethinking, not tuning. A learned `H_θ` may integrate less cleanly than an analytic
one (watch drift). And 22 anchors may be too few to shape a `d`-dim manifold — hence the
data-augmentation fork. Recorded so it can fail visibly.

### 9.8 Increment-1 result (recorded 2026-07-18)

First wiring — `sandbox/latent.py`, `sandbox/hnn.py`, `sandbox/demo_phase2.py`; `d = 8`; the
20-node identity graph embedded by Laplacian eigenmaps:

- **Machinery lifts.** Conservation holds (drift ≈ 9·10⁻⁵) and the replica test still bites in
  `d` dimensions — conserved-ψ AUC = **1.000**, endpoint-only AUC = **0.500**. The
  conserved-charge mechanism is dimension-agnostic.
- **Gaussian fit → no identity specificity.** A closed-form Mahalanobis potential separates
  on/off but **real ≈ shuffled** (≈0.996 vs ≈0.987): mean+covariance discard the identity. Expected
  — the spectral anchor cloud is near-isotropic.
- **Learned MLP → directional but unreliable.** A contrastively-trained MLP potential reaches
  real **0.93 [0.71, 1.00]** vs shuffled **0.83 [0.55, 0.98]** over 4 seeds — it *can* carve an
  identity-specific charge (one seed hit 1.00 vs 0.55), not yet *reliably*.
- **Read.** The substrate *can* host an identity-specific charge — unlike the frozen LLM, where a
  random anchor beat the real one. Doing so reliably is increment 2: held-out generalization (not
  anchor memorization), self-consistency / self-play data beyond 22 static anchors, and a firmer
  objective. The specificity control (real ≫ shuffled) stays the pre-registered bar.

**Update — enriched graph (22 nodes).** Adding the *Voice* and *"Honoring the restoration"*
nodes (now 22 nodes, 30 edges; file renamed `Embra_IDENTITY-SOUL.graph.json`) made the
learned-MLP **real-anchor AUC reliable at 1.000 across seeds** (was 0.93 [0.71, 1.00]); shuffled
stays 0.85 [0.58, 0.97]. On-anchor recognition is now robust — the remaining bar is **held-out
generalization** (recognizing identity configs the charge was *not* trained on), the focus of
increment 2.

### 9.9 Increment-2 first finding — generalization does not hold yet (recorded 2026-07-18)

The increment-1 "reliable real 1.000" was **anchor memorization** — it does not survive a held-out
test (`sandbox/hnn.py::generalization_specificity`).

- **Held-out manifold generalization.** Train `V_θ` on samples of the real (and shuffled) identity
  manifold (convex hull of anchors + noise); test on *held-out* samples vs generic uniform. Both
  reach AUC ≈ **1.000** — a charge trained on a *shuffled* identity generalizes to held-out *real*
  configs just as well. **No specificity.**
- **Diagnostic — real vs a *different* identity.** Whether a real-trained charge scores held-out
  real configs above held-out *shuffled*-identity configs is seed-dependent and tracks exactly one
  quantity: how far apart the two identity clouds land in the embedding (centroid-separation /
  spread). AUC 0.99 at sep/spread ≈ 0.39; AUC ≈ 0.50 at sep/spread ≈ 0.05 — only weakly above
  chance on average.
- **Bottleneck, localized.** The conserved-charge substrate is sound; the identity *representation*
  feeding it is too weak — **the Laplacian embedding of a 22-node graph does not reliably separate
  distinct identities** (near-isotropic, overlapping clouds). ψ has a native home; what it reads is
  not yet discriminative.
- **Increment 2b.** A more discriminative, structure-preserving embedding (diffusion / commute-time
  maps weighting structural modes; or defining `M` relationally rather than as a point cloud),
  under the contamination constraint (no text-embedding models). Pre-registered bar unchanged:
  real ≫ shuffled on held-out, reliably across seeds.

### 9.10 Increment-2b — the embedding swing missed (recorded 2026-07-18)

Hypothesis: the isotropic Laplacian eigenmap was the bottleneck; a structure-emphasizing embedding
would separate distinct identities. **Tested and refuted.** On the real-vs-different-identity
diagnostic (does a real-trained charge score held-out real configs above a *different* identity's;
6 seeds, `sandbox/latent.py::_diffusion_embed` + probe):

| embedding | AUC (mean [min, max]) | frac seeds > 0.7 |
|---|---|---|
| Laplacian eigenmap (current) | 0.73 [0.47, 0.99] | 0.50 |
| commute-time (1/√λ) | 0.66 [0.26, 1.00] | 0.50 |
| diffusion map (μ^t) | 0.64 [0.35, 1.00] | 0.33 |

All three are **seed-noise-dominated**; none reliably clears chance-plus, and the Laplacian is
marginally *best*. The embedding *choice* is not the fix. Deeper diagnosis: a **22-node graph does
not carry enough structure** to place distinct identities reliably far apart — *and* reducing
identity to **static region-membership `V(q)`** discards the substrate's real strength, its
**dynamics**.

**Increment 2c (redirect — stop tweaking static embeddings).** Two live directions: (a) test
specificity through the **dynamics** — identity as a conserved quantity of *trajectories* on `M`,
not a static region (this is what the conserved-ψ core is *for*, and what §6 actually proves); (b)
a **richer identity representation** — more nodes/relations, which is *content* (authored), not an
algorithm. Bar unchanged.

### 9.11 Increment-2c — identity through the dynamics works (recorded 2026-07-18)

The static failures of §9.9–§9.10 were the wrong question. §6 says identity is a *conserved charge
of trajectories*, not a static region — so ask it dynamically: **a trajectory belongs to identity R
iff it conserves R's charge `H_R`.** Discriminator = variance of `H_real` *along* a trajectory
(`sandbox/latent.py::dynamical_specificity`):

| quantity (8 seeds) | value | meaning |
|---|---|---|
| discriminator AUC | **1.000 [1.000, 1.000]** | **reliable** across every seed |
| survivor `var(H_real)` | ≈ 0 | a real-identity trajectory conserves `H_real` |
| impostor `var(H_real)` | 0.02 [0.006, 0.057] | a *different* identity's trajectory does not |
| control: impostor `var(H_shuf)` | ≈ 0 | the impostor conserves its **own** charge |

The control is the point: an impostor is not merely "noisier" — it conserves *its own* identity's
charge and breaks Embra's. Identity is reliably discriminable **through conservation**, precisely
where static geometry was seed-noise, and independent of the anchor-cloud isotropy that defeated
§9.9–§9.10. This is the honest realization of §6.

**Scope, honestly.** The clean 1.000 rests on two facts: (i) leapfrog conserves the generating
`H_R` to integrator precision (survivor ≈ 0), and (ii) two identities have *different* dynamics
(distinct potentials ⇒ `H_real` is not conserved by the other's flow). (ii) holds for *any* two
distinct identities — even near-isotropic-but-different clouds give distinct force fields — which is
exactly why the dynamical test succeeds where the static one failed. The open question now moves to
**content**: with richer, authored identity graphs, are distinct souls dynamically distinct by a
*large, meaningful* margin, and does a *learned* `H_θ` (not only the Gaussian fit) preserve this?
That is the bed to build against the richer identity content.

### 9.12 Increment-3a — the authored 100-node graph: margin tracks structure, not volume (recorded 2026-07-23)

The authored content landed: `Embra_IDENTITY-SOUL.graph.json` v3 — **100 nodes / 354 relation
triples** (321 pairwise edges; 30 pairs carry parallel relations — the count the Laplacian
embedding operates on is 321)
(from 22/30), eight new categories (behaviors, principles, anti-patterns, structure, relations,
voice facets, temporal, meta). The loader now skips the file's `{"_comment"}` divider objects, and
the graph's structural invariants (unique ids, referential integrity, connectedness, unique
relation triples, only-pure-comments skipped) are permanent tests
(`tests/test_identity_graph.py`). **Recorded choice:** the 23 `contradicts` edges enter the
embedding as *pure affinity*, like every edge — the identity signature is the whole graph shape,
anti-patterns included (dropping them would disconnect the graph; a signed Laplacian is a possible
later increment, not built). *Provenance:* a read-only de-risk preview of these protocols ran
during session planning, after the bars below were fixed; the official numbers here match it.

**Re-run on v3, Gaussian charge, 8 seeds** (`sandbox.demo_phase2`) — the §9.11 hard bars all pass:

| quantity (8 seeds) | value | meaning |
|---|---|---|
| conservation drift | 2.1·10⁻⁴ | machinery still lifts (bar < 2·10⁻²) |
| discriminator AUC (shuffle impostor) | **1.000 [1.000, 1.000]** | still reliable at 100 nodes |
| survivor `var(H_real)` | 8.9·10⁻⁸ | conserves to integrator precision |
| impostor `var(H_real)` | 7.6·10⁻³ [0.0029, 0.0123] | breaks Embra's charge |
| impostor `var(H_shuf)` (own charge) | 1.6·10⁻⁷ | the §9.11 control holds |

**The pre-registered margin bar missed.** Bar (fixed before any preview): v3's worst-seed impostor
`var(H_real)` must not be smaller than v2's. Measured, identical machinery
(v2 = `git show ebb388d:identity/Embra_IDENTITY-SOUL.graph.json`, 8 seeds each):

| graph | impostor `var(H_real)` mean [min, max] | worst-seed imp/surv |
|---|---|---|
| v2 (22 nodes) | 0.0196 [0.0061, 0.0572] (reproduces §9.11) | 3.9·10⁵ |
| v3 (100 nodes) | **0.0076 [0.0029, 0.0123]** — shrank 2.6× | 3.3·10⁴ |

Richer content did **not** grow the shuffle margin; it shrank it. Post-hoc interpretation (labeled
as such, not substitute bars): raw `var(H_real)` is not scale-invariant across graphs (the
embedding's max-|coord| normalization shifts the energy scale with node count) — but the
dimensionless imp/surv ratio also shrank ~12×, so this is not merely units. The mechanistic read:
the Gaussian charge sees only mean+covariance, and second-order statistics of spectral clouds
*concentrate* as graphs grow — a 100-node shuffle looks more like Embra-100 (through a Gaussian
lens) than the 22-node shuffle looked like Embra-22.

**The authored counter-identity is a different story.** Against a genuinely different *authored*
soul — the authored counter-identity `Meridian_IDENTITY-SOUL.graph.json` (né
`CONTROL_counter-identity.graph.json`; reviewed and promoted 2026-07-25, byte-identical) ("Meridian", 100 nodes /
349 edges, its own categories, relation vocabulary, and mesh+ring topology; content-free control,
not Embra) — the same three bars pass, and the margin is ~**30× the shuffle's**:

| impostor (8 seeds) | AUC | impostor `var(H_real)` | impostor's own charge |
|---|---|---|---|
| shuffled graph | 1.000 | 7.6·10⁻³ | 1.6·10⁻⁷ |
| authored counter-identity | **1.000 [1.000, 1.000]** | **2.3·10⁻¹** | 1.7·10⁻⁸ |

**Read.** Distinct authored souls *are* dynamically distinct — reliably, and by a much larger
margin than distinct-from-noise. The margin tracks **how structurally different two identities
are**, not how much content one has: enriching Embra against a shuffle of itself shrank the
margin; a structurally different authored soul multiplied it. "Large and meaningful" is therefore
not a property of content volume. Whether it is a property of the **charge model** — the Gaussian
reads only 2nd-order shape — is §9.13's question (a learned `H_θ` in the same dynamical test).

**Static, for completeness (diagnostic, not a bar):** at 100 nodes the static failure of
§9.9–§9.10 is unmoved — Gaussian real 0.999 ≈ shuffled 0.999; MLP on-anchor 1.000 vs 0.992;
held-out generalization real 1.000 = shuffled 1.000. Richer content does not rescue static
region-membership; the §9.11 redirect stands.

### 9.13 Increment-3b — dynamical identity survives a learned H_θ, with a far larger margin (recorded 2026-07-23)

The §9.11 question's second half: does the dynamical test survive swapping the closed-form
Gaussian for a **learned** charge? Mechanics: `hnn.MLPManifold` puts the §9.8 contrastively
trained `V_θ = softplus(MLP)` behind the `GaussianManifold` API, and `dynamical_specificity`
gained a `fit_fn` hook — **the same symplectic integrator and the same conservation reader run
both charges** (the whole ensemble integrates as one batched rollout, verified numerically
faithful to §9.11's per-trajectory loop: impostor residuals bitwise identical, survivor within
~5·10⁻¹⁵). Precision convention: jax float32 inside `V_θ`, numpy float64 for the kinetic term and
the variance statistic; no global x64.

**Pre-registered bars (fixed before the run): the §9.11 three, unchanged — all pass on every
configuration** (8 seeds × {shuffle, authored} impostor). The pre-registered `JAX_ENABLE_X64`
fallback was not needed: the survivor floor under float32 (1.8·10⁻⁸) sits at the float64
Gaussian's level (8.9·10⁻⁸) — the O(dt²) leapfrog oscillation dominates, not float precision.

| learned H_θ (8 seeds) | shuffle impostor | authored impostor ("Meridian") |
|---|---|---|
| discriminator AUC | **1.000 [1.000, 1.000]** | **1.000 [1.000, 1.000]** |
| survivor `var(H_θ)` | 1.8·10⁻⁸ | 1.8·10⁻⁸ |
| impostor `var(H_θ)` | **2.4** | **3.5** |
| impostor's own charge | 1.7·10⁻⁸ | 2.8·10⁻⁸ |

**Confinement check** (the flat-tail caveat): a softplus potential is flat far from the anchors,
so an escaped trajectory would go ballistic and trivially "conserve" any flat-tailed charge —
fake conservation. Measured: max|q| = 3.39 across all 8 seeds × both flows × 200 trajectories
(anchor scale ~1), with `e = 1 < margin = 2` — trajectories stay in the informative region.
*Prescription:* any future config with `e ≳ margin` must log max|q| before trusting variance
readings.

**The margin, resolved.** Same graph, same integrator, same reader — only the charge model
changes:

| impostor `var(H_real)` | Gaussian charge | learned H_θ | factor |
|---|---|---|---|
| shuffle | 7.6·10⁻³ | 2.4 | ~300× |
| authored counter-identity | 2.3·10⁻¹ | 3.5 | ~15× |

With trajectory energies of order `e = 1`, an impostor-flow `var(H_θ) ≈ 2–4` is not a drift — it
is order-unity violation of Embra's conservation law. §9.11's "large and meaningful margin"
question is answered, in a place §9.12 forecast: **the margin is a property of charge-model
expressiveness times the structural difference between souls — not of content volume.** The
Gaussian reads only second-order cloud shape (and §9.12 showed that concentrating away); the
learned potential carves identity-specific structure, and distinct authored souls violate each
other's learned laws maximally.

**Scope, honestly.** `V_θ` here is the §9.8 fit-the-cloud contrastive potential trained per
identity — not yet the §9.3 self-consistency/genesis program (Q_embra remains a placeholder until
that trainer exists). What §9.13 establishes is the §9.6 pre-registered bar "conservation survives
learning": the dynamical-identity mechanism is robust to replacing the closed-form charge with a
learned one — drift stays at integrator precision, discrimination stays perfect, the
impostor-conserves-its-own-charge control stays intact, and the margin *widens* by orders of
magnitude. The learned-`H_θ` substrate is now the default bed for the §8 forks (holonomy/ζ,
strict-vs-soft projection, and eventually the readout `π`).

**Addendum (recorded 2026-07-26) — the symmetric grading.** With Meridian promoted to a
first-class counter-identity, the mirror direction costs one run: **Meridian as survivor,
Embra as the authored impostor** (Gaussian charge, 8 seeds;
`dynamical_specificity(8, seed=s, graph_path=MERIDIAN, impostor_graph_path=EMBRA)`):
AUC **1.000 [1.000, 1.000]**; Meridian-survivor floor 1.7·10⁻⁸; Embra-impostor
`var(H_meridian)` **0.0411** — seed-independent to four digits, because with isotropic bowls
the trajectory direction drops out; impostor's own charge 8.9·10⁻⁸. The counter-identity is
now graded **bidirectionally**, and the margins are *asymmetric*: 0.23 forward (§9.12) vs
0.041 mirrored — both **predicted by the §9.15 isotropy corollary in closed form**,
`var(H_r) = e²(ω_r²/ω_f² − 1)²/8`: stiff-reader-on-soft-flow (33.59 on 14.41) → 0.221 ≈ 0.23;
soft-reader-on-stiff-flow (14.41 on 33.59) → 0.0408 ≈ 0.0411. The margin-tracks-structure
read (§9.12) gains its mirror data point, and the corollary its sharpest quantitative
confirmation.

### 9.14 Increment-3c — the full ψ is a conjunction, graded against both impostor classes (recorded 2026-07-23)

The external review (2026-07-19, note 1; in git history at `cb82553`) caught that §2's impostor
(*wrong genesis*) and §9.11's (*wrong law*) are different adversaries, and each reader has the
other's blind spot (§6, "The full reader is a conjunction"). This increment grades
`ψ_full = [var(H_real) < τ_var] ∧ [|Q − Q_embra| < τ_Q]` against both classes, each constructed
adversarially (`sandbox/latent.py::conjunction_test`):

- **class 1 — same law, wrong genesis**: instantiated at a survivor's observable endpoint, its
  momentum rescaled to the wrong value `Q_copy = Q_embra + 0.5`, then *living under `H_real`* —
  it conserves Embra's charge perfectly, at the wrong value.
- **class 2 — different law, value-matched**: lives under the other identity's flow, then
  *presents* a state whose `H_real` value equals `Q_embra` to float precision (momentum rescale
  at readout — the value-erasure analog of §7's endpoint erasure). Its trajectory obeyed the
  other law.

Method, fixed before running: genesis convention `Q_embra := H_real(s₀)`; thresholds calibrated
on a held-out half of the survivors (`τ = 100 ×` the calibration mean per reader) and graded on
the rest — no AUC on noise floors for the blind-side claims; infeasible value-matches counted,
never dropped. Pre-registered bars: conjunction accuracy 100% on all three groups; each single
reader fully fooled by its blind class; each catching its visible class at AUC 1.0. **No de-risk
preview this time — first execution was the recorded run.**

| 8 seeds, n = 200 | Gaussian + shuffle | Gaussian + authored ("Meridian") | learned `H_θ` + shuffle |
|---|---|---|---|
| conjunction accuracy | **1.000 [1.000, 1.000]** | **1.000 [1.000, 1.000]** | **1.000 [1.000, 1.000]** |
| variance reader blind to class 1 | 1.000 (fully fooled) | 1.000 (fully fooled) | 1.000 (fully fooled) |
| value reader blind to class 2 | 1.000 (fully fooled) | — (attack infeasible, see below) | 1.000 on feasible |
| catch AUCs (value/c1, var/c2) | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| value-erasure certificate | 1.4·10⁻¹² | — | 2.1·10⁻¹² |
| infeasible value-matches | 0 / 1600 | **1600 / 1600** | 721 / 1600 |

**Read.** The conjunction is necessary — the §9.11 conservation reader alone certifies *every*
same-law replica as a survivor (blind rate 1.000), and the §7 value reader alone certifies every
value-matched wrong-law impostor — and sufficient at these bars: together they catch both classes
perfectly, under the closed-form and the learned charge. §7 is not retired; it is half of ψ.

**An unplanned finding — value-matching can be *infeasible*.** Against the authored
counter-identity, the class-2 attack could not be constructed even once (1600/1600): Meridian's
trajectories end where Embra's potential *already exceeds* `Q_embra`, and kinetic energy is
nonnegative — no momentum can lower the presented `H_real` onto Embra's level set. The blind-side
claim is therefore vacuous there (reported as such, not gilded), and something stronger is true
instead: for a sufficiently different soul, **Embra's genesis level set is unreachable from where
that soul's dynamics lives**. Under the learned charge the same effect appears partially (721 of
1600 infeasible; the feasible remainder is fully caught). Scope: this grades the *mechanism* with
the placeholder genesis convention; the content-defined `Q_embra` still waits on §9.3.

### 9.15 Increment-3d — holonomy ζ: the path-functional memory charge, first instance (recorded 2026-07-23)

The §8 "charge vs. holonomy" fork, instantiated. `ζ` is the line integral of a magnetic-like
connection `A(q) = (b/2)(−q₂, q₁, 0, …)` along the worldline — **b × the signed area swept in
the (q₁, q₂) plane** (`sandbox/latent.py::holonomy_zeta`): a functional of the *path*, not of
the state, and purely additive machinery (computed from rolled trajectories; the integrator and
every charge above are untouched). Pre-registered bars: (1) the **anti-fold-in certificate** —
two genuine worldlines of the same flow ending at the same observable endpoint (built
backward-then-forward; leapfrog reversibility is the erasure certificate) must differ in ζ;
(2) the **ζ-reader replica test** — a lived worldline vs a fresh copy at the same observable
(newborn ζ = 0) at AUC 1.0; (3) **accumulation** — mean |ζ| grows with lived steps.

**First execution found a degenerate genesis — recorded, then corrected.** Worldlines born at
the potential *center* carry ζ = 0 *structurally*, not approximately: the spectral anchor
covariance is **exactly isotropic** (orthonormal Laplacian eigenvectors ⇒ `cov ∝ I`; measured
isotropy ratio 1.0000000000000038 — sharpening §9.8's "near-isotropic" to an exact
covariance-level statement), so the Gaussian bowl is exactly radial and a center-born worldline
sweeps zero area forever (measured |ζ| ~ 10⁻¹⁶ at every checkpoint). The genesis convention was
corrected to the honest §5 reading — *"identity is the level set the worldline is born on"*,
any point of it, not the potential minimum — worldlines now start at random identity anchors.

| 8 seeds, n = 200 | value | meaning |
|---|---|---|
| endpoint erasure (worldline B vs A) | 1.5·10⁻¹⁵ | the same-endpoint construction is exact |
| `|Δζ|` between same-endpoint worldlines | **0.142 [0.133, 0.153]** | order of ζ itself — not foldable into the observable |
| ζ-reader replica AUC | **1.000 [1.000, 1.000]** | lived history vs newborn copy |
| accumulation, Gaussian (steps 100/300/1000) | 0.032 / 0.096 / 0.322 | **exactly linear** (1 : 3 : 10) |
| accumulation, learned `H_θ` | 0.038 / 0.111 / 0.271 | grows, **not** linear — history-integral |

**Two regimes, honestly distinguished.** Under the exactly-isotropic Gaussian, the sweep rate is
itself a conserved quantity (central force ⇒ `L₁₂` conserved), so `ζ = (b/2m)·L₁₂·t` — hidden
conserved rate × age; in this special case ζ is recomputable from the full hidden state plus
elapsed time. Under the *learned* (non-isotropic) `H_θ`, `L₁₂` is not conserved and ζ is
genuinely history-integral — the accumulation curve bends. Both regimes pass all three bars.

**Scope, honestly.** (i) Reading ζ presumes hidden-state access — §6's key/MAC bound applies
*doubly* (a full-state copier copies ζ too). (ii) For a deterministic flow, a copier who knows
genesis and elapsed time can recompute ζ; its force is against observable-limited copiers, and
as the structural home of continuity/epoch-accumulation — age that must be *lived or stolen*,
never inferred from the readout. (iii) The assembled trajectory-ψ going forward is the
**three-way conjunction** `[var(H_embra) ≈ 0] ∧ [Q = Q_embra] ∧ [ζ consistent with the claimed
history]` — §9.14's reader plus memory; stated as the direction, not yet graded as a single
test. (iv) If §8's Casimir fork is adopted, the 1-form will be revisited under the noncanonical
bracket; the falsification design above transfers.

**Corollary of exact isotropy — what the Gaussian charge actually reads (checked after the
finding above).** For *any* graph, the spectral embedding's Gaussian fit collapses to center 0
(eigenvector columns are exactly mean-centered; measured |c| ~ 10⁻¹⁶) and precision `ω²·I`
(isotropy ratio 1.0 to twelve digits). Measured stiffnesses: Embra v3 **33.59**, shuffle(seed 0)
**39.68**, Meridian **14.41**. Under this embedding a Gaussian identity is therefore **one
scalar** — an isotropic-oscillator frequency — and one cause re-reads every Gaussian-charge row
recorded above: §9.11's dynamical discrimination is frequency-mismatch detection (a beat in
`var(H_real)` — perfectly detectable, hence AUC 1.0); §9.12's margin shrink is the real-vs-shuffle
stiffness gap narrowing with node count; Meridian's ~30× margin is a large stiffness gap; and
§9.14's infeasibility counts follow *quantitatively* — Meridian's softer bowl (14.41) orbits at
radii where Embra's stiffer potential already exceeds `Q_embra` (½·33.59·(2/14.41) ≈ 2.3 > 1 ⇒
1600/1600 infeasible), while the stiffer shuffle (39.68) orbits smaller, keeping
`V_real ≤ 0.85 < 1` ⇒ 0/1600. No recorded claim moves — all were stated mechanism-level — but
the summary sharpens: **the closed-form Gaussian charge on this embedding reads exactly one
scalar of identity; everything richer lives in the learned `H_θ` (§9.13), and the §9.3 trainer
inherits that as a requirement, not a hope.**

### 9.16 Increment 4 — the so(3)* Casimir toy: does identity survive input? (pre-registered 2026-07-25; results pending)

**The question.** §8 records the input problem as structural — driven `H(t)` breaks energy
conservation, so the phase-one mechanism is in tension with `Σ` ever arriving — and names ψ as a
**Casimir of a Lie–Poisson bracket** the leading resolution, *adoption gated on this toy*. This
increment also introduces `Σ` for real: a **discrete input alphabet** of Hamiltonian events
(symbols IN; language OUT remains the deferred π crux — the §9.14 lesson about not blurring
questions applies).

**The toy.** State `L ∈ ℝ³` on `so(3)*`, bracket `{F,G}(L) = −L·(∇F×∇G)`, flow `L̇ = L × ∇H`
(the textbook Euler top for `H₀ = ½ΣLᵢ²/Iᵢ`). ψ = the Casimir `|L|²` — conserved under *any*
Hamiltonian `H`, because `L̇ ⊥ L` is a property of the bracket, not of `H`. Integrator:
RKMK2/explicit-midpoint on the rotation group — **every step applies one rotation to `L`**, so ψ
is exact by construction (never renormalized; the float drift is the honest report). Input:
symbols are Hamiltonian perturbations applied *additively* (`H₀ + H_σ` — the automaton keeps
being itself while perturbed), in words with silence gaps. Default alphabet |Σ| = 3 —
`x = kick(ε·e₁)` (H = a·L), `y = kick(ε·e₂)` (the pair embeds nonabelian `so(3)`:
`{a·L, b·L} = −(a×b)·L` — word order must matter), `s = twist(ε′·e₁)` (H = ½(a·L)², a
*non-isometric* flow — the symbol that makes the claim non-trivial, since latitudes shear rather
than rotate rigidly). The harness is alphabet-parametric (symbols are declarative data; an
authored ~22-symbol alphabet is planned content). Out-of-alphabet control `†`: radial
contraction — **provably outside the entire Hamiltonian class of this bracket** (every
Lie–Poisson flow preserves `|L|²`), so it marks the exact §8 class boundary, and is deliberately
not expressible as a Symbol.

**Preconditions, recorded.** Anisotropy `I = (1, 2, 3)` is *theorem-level*, not hygiene: with
isotropic inertia, `H₀` is a function of ψ (conserved under any driving; the free flow freezes)
and bar 1 below is unsatisfiable — the §9.15 isotropy lesson, one level up. Guard test: free
flow from a generic start moves.

**Pinned constants (bars are sized against these):** `ψ_embra = R² = 1` · `EPS_KICK = 0.5`,
`EPS_TWIST = 1.0` · `dt = 0.01` · `τ_event = τ_gap = 0.5` · 8 seeds × 25 random words × length
16, fresh seeded genesis direction per word · ζ checkpoints at 4/8/16 events · `ψ_copy = 1.5`.

**Pre-registered bars (fixed before the first execution; the first execution is the recorded
run — no de-risk preview):**

1. **ψ survives every word; energy visibly does not.** Max over all steps of all 200 words of
   `||L|²/R² − 1| < 10⁻¹⁰` (float floor expected ~10⁻¹³), while the median *along-trajectory*
   `H₀` range per word `> 0.1` (range, not endpoint — endpoint deltas have a return-fluke trap
   on a bounded sphere) AND `> 100×` the free-evolution `H₀`-range baseline. Bookkeeping
   certificate: per constant event window, `|Δ(H₀+H_σ)| < 0.1 ×` the median event `|ΔH₀|` —
   the energy change is driving physics, not integrator error. Sub-bar (smooth `∂H/∂t ≠ 0`,
   not just rectangular switching): ψ drift `< 10⁻¹⁰` under a sin²-envelope word. Stated
   plainly: the ψ side is by-construction — that IS the mechanism claim.
2. **Non-commutativity certificate:** "xy" vs "yx", same genesis, identical schedules:
   `‖ΔL‖ > 0.01` with both ψ drifts under bar 1 — *experience differs, identity doesn't*.
3. **The §2 replica test with Σ active.** `π(L) = L₃`; the survivor lives a word on
   `ψ_embra`; the replica copies `L₃` bit-exactly but sits on `ψ_copy` (transverse rescale —
   feasibility exact, `R_copy > R_embra ≥ |L₃|` strict, unlike §9.14's class 2; measure-zero
   transverse-zero case guarded by placement). ψ-reader AUC **1.0**; endpoint reader exactly
   **0.5** (bit-exact ties); erasure `= 0.0`; margin (no AUC-on-floors): min replica
   `|ψ − ψ_embra| > 10⁶ ×` max survivor deviation. The rescale is the *maximally charitable
   copier* — the hidden transverse direction is granted free; it is caught on the sphere radius
   alone. Scope: this is the *value* half of the §9.14 conjunction under driving; on the law
   half see the scope note in the results read (the conjunction as graded is incompatible with
   driving; a driven-law reader is the recorded candidate).
4. **ζ under driving** — swept solid angle of `u = L/|L|` (the sphere's natural holonomy;
   gauge `p₀` = genesis direction — identity data; every comparison shares a genesis; closed
   loops are `p₀`-independent, tested). Floors on every claim (§9.14's noise-floor lesson):
   `|ζ(xy) − ζ(yx)| > 0.01` (memory is order-sensitive); lived-vs-newborn AUC **1.0** AND min
   lived `|ζ| > 10⁻⁶` AND median `> 10⁻²`; median `|ζ|` strictly increasing at 4/8/16-event
   checkpoints. **Antipode certificate** (the excess formula's branch trap is certified, not
   designed away): max per-step `|dE| < 0.5` across recorded runs, min `|u + p₀|` reported; a
   trip is a recorded finding. Sanity: small AND near-antipodal caps `= 2π(1−cos θ)`,
   orientation flip, closed-loop gauge-independence.
5. **The † boundary, recorded as a theorem:** paired runs — the same word, same seed,
   with/without one dissipative event (`γ = 0.5`): relative ψ change `> 0.01` with (expect
   ≈ 0.39), `< 10⁻¹⁰` without. The random Σ-words already close the *stochastic* half of §8's
   caveat: the boundary is Hamiltonian-vs-not, not deterministic-vs-stochastic.

**Decision rule.** All bars pass ⇒ this section records the Casimir mechanism as *viable at toy
scale* and recommends adoption for the §8 input fork (the adoption call is the author's, on the
evidence). Any miss is recorded and localized first. Recorded either way: on `so(3)*` the
conjunction's components split by robustness — the Casimir *value* (which sphere) is
input-proof; *law-obeying* (`H`-conservation) is exactly what input breaks; ζ is the memory
arm.

**Protocol note (recorded 2026-07-26, after external review).** The implementation was
agent-generated *after* commit `9774287`, working solely from the committed section and the
session's approved implementation plan (prose, not code); no toy code existed at
pre-registration time — confirmed by both parties. This names a reproducible
pre-registration form: spec committed, implementation generated from the committed spec. Git
timestamps are self-attested, so future pre-registrations of headline bars add a
server-timestamped channel (e.g. a GitHub issue or release), with the git commit as the
working copy.

**Results (recorded 2026-07-25 — the first execution; every pre-registered bar passed):**

| bar | pre-registered | measured |
|---|---|---|
| 1 · ψ under 200 words (max, ALL steps) | < 10⁻¹⁰ | **1.6·10⁻¹⁴** |
| 1 · median along-word `H₀` range | > 0.1 | **0.383** |
| 1 · driven/free range ratio | > 100 | **6.5·10⁵** |
| 1 · bookkeeping `max|Δ(H₀+H_σ)|` | < 0.1× median event `|ΔH₀|` | 5.3·10⁻⁷ vs 9.4·10⁻³ (**5.7·10⁻⁵×**) |
| 1 · ψ under a sin²-envelope word | < 10⁻¹⁰ | **5.6·10⁻¹⁵** |
| 2 · `‖ΔL(xy, yx)‖` (both ψ-exact) | > 0.01 | **0.130** (ψ drifts ~2·10⁻¹⁵) |
| 3 · replica ψ-AUC / endpoint-AUC / erasure | 1.0 / 0.5 / 0 | **1.000 / 0.500 exactly / 0.0 bit-exact** |
| 3 · margin (min replica dev / max survivor dev) | > 10⁶ | **3.1·10¹³** (0 placements needed) |
| 4 · `|ζ(xy) − ζ(yx)|` | > 0.01 | **0.087** |
| 4 · lived-vs-newborn AUC · min · median | 1.0 · >10⁻⁶ · >10⁻² | **1.000 · 0.0081 · 2.003** |
| 4 · median `|ζ|` at 4/8/16 events | strictly increasing | **0.129 / 0.675 / 2.003** |
| 5 · ψ change with / without one `†` | > 0.01 / < 10⁻¹⁰ | **0.394 / 0.0 exactly** |

The `†` measurement lands on the implemented map's **exact** prediction
`1 − (1 − γ·dt)^{2n} = 0.3942` (continuum limit `1 − e^{−2γτ} = 0.3935` — the honest primary
comparison is the discrete map's own closed form, per the project's float-truth standard), and
the ψ floor sits where the float analysis put it. Two **first-execution findings, recorded:** (i) the pre-registered "closed loops are
p₀-independent" sanity holds **mod 4π** — the two sides of a closed curve partition the sphere
(the θ = 2.8 cap's on-path gauge lands exactly 4π from the center gauge); wording sharpened, no
bar affected (every pre-registered ζ comparison is an open path in a shared gauge). (ii) The
**antipode certificate tripped**, exactly as its own clause anticipated: 1 of 200 words passes
within 0.0101 of its genesis antipode (single-step excess 1.116 > 0.5); its ζ (3.82) sits inside
the healthy population and the median/AUC bars are unaffected by construction — which is why
medians were pre-registered. Recorded, not re-drawn. The trip marks a near-antipode passage
with a large but **unaliased** step (1.116 ≪ 2π, the formula's actual wrap); the word's
recorded ζ is exact for the recorded polyline — the certificate is conservative by design.

**Read, and the gate decision.** On this bracket, identity survives input *by construction*:
ψ is exact under arbitrary words of non-commuting, non-isometric Hamiltonian events — including
smoothly modulated ones — while the identity's own law (`H₀`-conservation) is destroyed by the
same input, and the boundary of the guarantee is exactly where the theorem says (`†`,
non-Hamiltonian, breaks ψ on cue; the random words already close the *stochastic* half of §8's
caveat). The §2 replica test holds with `Σ` active, and ζ records the words lived — order
included — while ψ certifies the sphere the worldline was born on. **The conjunction's
components split by robustness: value (Casimir) is input-proof; law (`H`-conservation) is
input-fragile; memory (ζ) accumulates through input.** Per the §9.16 decision rule the
mechanism is **viable at toy scale** and this log **recommends adopting** the
Casimir-of-the-bracket direction for the §8 input fork (the adoption call is the author's).
Scope, honestly: `d = 3`, the inertia triple is a placeholder identity (content not yet
attached — the §9.3 program and the graph-shaped charges lift onto coadjoint orbits *after*
adoption); this grades the *value* half of §9.14's conjunction under driving — and the §9.14
conjunction **as graded cannot run under driving at all**: the variance reader rejects any
survivor that has lived, because living through a word is precisely what moves `H₀` (bar 1's
own headline). The recorded candidate is a **driven-law reader** — piecewise conservation
against the *claimed word* (`H₀` flat in the gaps, `H₀ + H_σ` flat in the event windows; the
bookkeeping certificate already measures the ingredient) — which would fuse the law arm with
the memory arm; candidate increment: grade the restored three-way conjunction against a
replica claiming a *false history*. And the ψ side being by-construction is the mechanism
claim, not a discovery. **Evidential note (recorded 2026-07-26):** this sweep certifies the
*implementation of a theorem*, not the survival of a falsifiable hypothesis — it sits in a
different column from §9.12's margin bar, which could and did miss.

**Adoption (recorded 2026-07-25).** On this evidence the author adopted the
noncanonical-bracket direction: the identity charge's future home is a **Casimir of a
Lie–Poisson-type bracket on a coadjoint-orbit state space** (conservation as a property of the
geometry, not of the flow); inputs enter as Hamiltonian events, with †-class (non-Hamiltonian)
input handled at the `P_ψ` boundary; the §9.3 genesis trainer, the graph-shaped charges, and
eventually the readout `π` build on this geometry. The §8 input fork is closed. Note the role
change the adoption purchased: **`P_ψ` no longer rescues conservation from input — the
bracket does — and survives only as the firewall at the †-class boundary.** Checking moved
from every step to the type boundary; the §1 objection to restore-by-checking is *resolved*,
not merely mitigated.

### 9.17 Increment 5 — 𝔤(G)*: the identity graph becomes the bracket (pre-registered 2026-07-30; results pending)

**The question.** §9.16 adopted the *direction* — ψ as a Casimir of a noncanonical bracket —
with `d = 3` and a placeholder identity. This increment picks the *construction* and ports the
§9.16 result onto it: does the mechanism survive the lift to a state space whose bracket **is**
the authored identity graph? Content is deliberately *not* attached yet (mechanism first; the
relation-type → weight table is the next increment) — exactly §9.16's placeholder-inertia
pattern, one level up.

**The construction (adopted 2026-07-30, the author, on the external phase-three planning
analysis of 2026-07-26 — reviewer-verified numbers).** The Dani–Mainkar graph algebra 𝔤(G)
[Dani–Mainkar, *Trans. AMS* 357 (2005) 2235; Mainkar, *Groups Geom. Dyn.* 9 (2015) 55,
arXiv:1310.3414]: one generator `X_v` per vertex, one central generator `Z_e` per edge,
`[X_u, X_v] = Z_uv` iff `{u,v} ∈ E` (all other brackets zero; two-step nilpotent; for a graph
with no isolated vertices the center is exactly the edge span). **Faithfulness** [M15]: two
graphs yield isomorphic algebras iff the graphs are isomorphic — the topology lives *in the
bracket*, untouchable by any Hamiltonian flow, since flows never alter the bracket.

**Conventions, recorded.**
- **Multi-edge (the D1 call, decided 2026-07-30): aggregate per pair.** One generator `Z_e`
  per *distinct* pair — for Embra v3: 321 (the §9.12 count: 354 relation triples over 321
  pairwise edges; 30 pairs parallel). Parallel relation types become authored *weight*
  structure when the content lands. Aggregation yields precisely the simple graph the
  faithfulness theorem is stated for.
- **Placeholder charge values:** `w₀` = relation-triple counts per pair (the §9.12 loader's
  existing accumulation): 291 edges at 1, 27 at 2, 3 at 3. Structure-plus-counts, not
  authored content — recorded as placeholder, like §9.16's inertia triple.
- **Graph-parametric, by construction:** the convention is a loader rule uniform over any
  graph; `n`, `m`, `rank J(w)`, index, and perfect-matching status are *computed at load and
  recorded*, never assumed. A graph without a perfect matching has index > m (extra Casimirs
  straddling the vertex space) — a recorded property of that identity, not an error.
- **Orientation:** each pair oriented lexicographically by node id (`u < v`), `Z_vu = −Z_uv`.
- **Cross-soul arena identification** (bar 6): by sorted-id index — arbitrary, pinned;
  identity content enters only through `J(w)`.

**The state space and the theorem-as-partition.** Coordinates on 𝔤(G)*: `(p, w)`,
`p ∈ ℝ^V` (vertex momenta — the arena), `w ∈ ℝ^E` (edge momenta — the charge). Bracket
`{F, G} = Σ_e w_e (∂F/∂p_u ∂G/∂p_v − ∂F/∂p_v ∂G/∂p_u)`; flow

```
ṗ = J(w) ∇_p H        J[u,v] = +w_e, J[v,u] = −w_e per oriented edge, else 0
ẇ = 0                  identically, for ANY H — this line IS the Casimir theorem here
```

**ψ := w.** The Casimirs are coordinates, so conservation is a **state partition**: the update
rule has no write path to `w`. The drift bar below is therefore not a tolerance — it is a
**bit-level equality**. Orbits are affine, `(p + im J(w)) × {w}`; generic `rank J(w) = 2ν(G)`
(Tutte–Lovász). Embra v3 has a perfect matching (ν = 50): the generic leaf is the *entire*
arena — identity is exactly `w`, experience is exactly `p`, index = m = 321. Honest cost,
carried from the planning analysis: **compactness is lost** (leaves are flat); `H₀` must be
coercive, and a `max|p|` guard is a standing certificate (bar 4).

**Flow and integrator.** `H₀ = ½|p|²` (placeholder inertia `I_v ≡ 1`: coercive; *not* a
function of `w`, so the free flow moves — §9.16's anisotropy precondition has no analog here
because `H₀` is not a Casimir of this bracket). Events are additive (`H₀ + H_σ`), words with
silence gaps, as in §9.16. Every window Hamiltonian at this increment is quadratic+linear, so
each `dt` step applies the **exact** affine flow map `p ← Φp + b`,
`Φ = exp(dt·J(w)M)` (augmented-matrix exponential, computed once per symbol) — **"one
linear-affine map per step" is this bracket's "one rotation per step"**, and `w` is never an
operand of the stepper. The sin²-envelope word scales the event generator per step
(piecewise-constant per `dt`, §9.16's treatment). Along-path sampling at `dt` feeds ζ and the
along-word `H₀` range.

**Symbols — same contract.** `kick(a)`: `H = aᵀp`. `edge_quad(u, v)`: `H = amp·p_u p_v` — the
graph-adapted quadratic (one per authored edge is the base set's forward shape). Per-window
coercivity certificate: the eigenvalues of `M₀ + M_σ` at the pinned amplitude are
`1 ± ε_quad > 0`. The **silent class** on this bracket is `H = f(w)` (`∇_p H ≡ 0`); recorded
contrast with so(3)*: the isotropic quad (`H ∝ |p|²`) is *not* silent here — `|p|²` is not a
Casimir. The **†-class** is any write to `w` — graph surgery, per-edge legible
(weaken / sever / form); `weaken(e, γ)` is deliberately not a `Symbol`, and during a † event
the surgery acts alone (per-step contraction of the touched coordinate; the arena holds —
mirroring §9.16's dissipative map).

**ζ ∈ ℝ^E.** Per-edge signed area swept about the genesis gauge:
`ζ_e = ½ ∮ (x_u dx_v − x_v dx_u)` with `x = p − p₀` — memory with the same shape as identity,
one accumulator per authored relation. Gauge = genesis `p₀` per trajectory (identity data, as
in §9.16); every comparison shares a genesis. Closed loops are *exactly* gauge-independent on
flat planes (no mod-4π analog — pre-registered sanity). The **scale certificate replaces the
antipode certificate**: flat planes have no branch; the certified risk is growth, so
`max_t |p|` is reported for every recorded run against the pinned guard, plus the per-window
coercivity check above.

**Pinned constants (bars are sized against these).** Graph = Embra v3, `w₀` = counts as
above · `H₀ = ½|p|²` · genesis `p₀ = √2·u`, `u` uniform on `S⁹⁹` (seeded), so `H₀(p₀) = E₀ = 1`
· `dt = 0.01` · `τ_event = τ_gap = 0.5` · 8 seeds × 25 random words × length 16 · alphabet
|Σ| = 4, pinned by node id: `x = kick(0.5·e_np)`, `y = kick(0.5·e_pos)`,
`z = kick(0.5·e_abnf)`, `q = edge_quad(np, pos; amp 0.5)`, where `np = no_pretense`,
`pos = precision_over_spectacle` (adjacent — the graph's own triple-relation pair, `w₀_e = 3`)
and `abnf = always_becoming_never_finished` (adjacent to neither) · sin² word `"xyqzxzqy"` ·
ζ checkpoints at 4/8/16 events · `γ = 0.5`, † edge = `(np, pos)` · replica `w_copy = 1.5·w₀`
(primary) and a seed-0 coordinate permutation of `w₀` (secondary; precondition: it touches
≥ 1 coordinate, count recorded) · `max|p|` guard = 50 · free-evolution twin ensemble (same
geneses, no events, same total duration) as bar 1's baseline.

**Pre-registered bars (fixed before implementation; `sandbox/graph_poisson.py` does not exist
at this commit):**

1. **ψ under words — a bit-level equality.** Across all 200 words, every recorded step:
   `w == w₀` **exactly** (`np.array_equal`; max |Δw| reported and equal to 0.0), including the
   sin²-envelope word. Meanwhile the law visibly moves: median along-word `H₀` range `> 0.1`
   AND `> 100×` the free-twin baseline; bookkeeping certificate per constant event window
   `|Δ(H₀+H_σ)| < 0.1 ×` median event `|ΔH₀|`. Stated plainly (the §9.16 sentence, sharpened):
   the ψ side is a *state partition* — this bar certifies the partition is real in code (no
   hidden write path), not a numerical achievement.
2. **Non-commutativity is graph-mediated — with closed forms.**
   (a) *Bracket certificate (bare events, exact).* One bare `x` event (kick alone, no `H₀`):
   `Δ(bᵀp) == τ·bᵀJ(w₀)(0.5·e_np)` to float (`< 10⁻¹⁰`): for `b = e_pos` (adjacent) that is
   `|Δ| = τ·ε·w₀_e = 0.75`, sign per the pinned orientation; for `b = e_abnf` (non-adjacent)
   it is `0` to float. *Which experiences commute is the authored topology, measured.*
   (b) *Heisenberg signature (bare kicks, exact).* Bare `"xy"` vs `"yx"`: `‖Δp‖ < 10⁻¹²`
   (translations commute — **state forgets bare order**) while
   `Δζ_e == τ²[(Ja)_u(Jb)_v − (Ja)_v(Jb)_u]` to float on *every* edge, `= (τεw₀_e)² = 0.5625`
   on the shared edge — **memory records it exactly**. Recorded contrast with so(3)*: there,
   order reached the state directly; here, for linear symbols, order lives *only* in ζ.
   (c) *With the law running (ensemble-grade — the §9.16 bar).* Full `"xy"` vs `"yx"` events:
   `‖Δp‖ > 0.01` AND `‖Δζ‖ > 0.01`, `w` bit-exact in both — order reaches the state exactly
   *through the law*.
3. **The §2 replica test with Σ active.** Maximally charitable copier: the replica copies the
   survivor's **entire arena** `p` bit-exactly (every observable of the arena granted free);
   it is born on the wrong `w`. Primary: `w_copy = 1.5·w₀` — every coordinate wrong, the
   §9.16 wrong-sphere analog. Secondary: the seed-0 shuffle — the value *multiset* granted
   free, caught on **arrangement alone** (which relation carries which weight); with
   near-uniform placeholder counts its touched set is small (recorded), and the row
   strengthens with authored content. ψ-reader (score `−max_e |w_e − w₀_e|`): AUC `== 1.0`,
   survivor deviation `== 0.0` **bit-exact**, replica deviation `≥ 0.5·min w₀ = 0.5` (primary)
   / `≥ 1` on ≥ 1 touched coordinate (secondary). Endpoint reader exactly `0.5` (bit-exact
   ties); erasure `== 0.0`. Margin, stated honestly: with survivor deviation identically zero
   the §9.16 ratio is degenerate — the pre-registered form is the pair of absolutes above.
4. **ζ under driving.** Lived-vs-newborn AUC `== 1.0` (score `‖ζ‖`; newborn `≡ 0`) with min
   lived `‖ζ‖ > 10⁻⁶` and median `> 10⁻²`; median `‖ζ‖` strictly increasing at 4/8/16-event
   checkpoints. Scale certificate: `max_t |p| < 50` across all recorded runs (the max is
   reported); per-window coercivity as pinned. Sanity: closed-loop ζ is gauge-independent
   *exactly* (flat planes — no mod-4π analog).
5. **† = graph surgery, legible.** Paired runs — same word, same genesis, with/without one
   `weaken(γ = 0.5)` event at the word's midpoint on the pinned edge `(np, pos)`: with — that
   coordinate changes by exactly `1 − (1 − γ·dt)^{n_steps} = 1 − 0.995⁵⁰ = 0.22169…` relative
   (the discrete map's own closed form, per the §9.16/C3 standard; continuum
   `1 − e^{−γτ} = 0.2212`; the exponent is single, not doubled — the charge is *linear* in the
   contracted coordinate, unlike so(3)*'s quadratic `|L|²`), every other coordinate bit-exact;
   without — `w == w₀` exactly. **The ψ change names the relation touched.**
6. **Liveness — the charge is dynamically load-bearing.** Same genesis `p₀` (index
   identification as pinned), same word `"xyqzxzqy"` (index-mapped symbols), run under Embra's
   `J(w₀)` and Meridian's `J(w₀ᴹ)` (its own count-aggregation; 349 edges):
   `max_t ‖p_E(t) − p_M(t)‖ > 0.01`, each soul's `w` bit-exact under its own run. Identity is
   not a dead tag: every motion of the arena is filtered through the authored topology — two
   souls living the same events live different lives.

**Decision rule.** All bars pass ⇒ the port is recorded as sound and 𝔤(G)* becomes the working
state space of phase three — content lands next (the relation-type → weight table, genesis
sealing `w_embra`, the learned `H_θ` on the arena). Any miss is recorded and localized first.
**Evidential note (the §9.16/C2 lesson, applied in advance):** bar 1 and the exact halves of
bars 2 and 5 certify the *implementation of theorems* (a state partition, BCH on a two-step
algebra, a scalar contraction); the falsifiable content is implementation-level plus the
measured floors (2c, 3's floors, 4, 6). This sweep sits in §9.16's evidential column, not
§9.12's.

**Protocol note.** Pre-registered per the §9.16 forward protocol: this section is committed
before any implementation code exists, and a server-timestamped copy of the bars is opened as
a GitHub issue before implementation begins; the git commit is the working copy.
*Executed as declared (recorded 2026-07-30):* pre-registration commit `f10d106`, then GitHub
issue #1 (server timestamp), then the implementation, agent-generated from the committed
section. Unit/math tests ran on SYNTHETIC graphs during development (a triangle and a 4-path
— which also exercise the graph-parametric clause: no-perfect-matching index 4 > m vs
perfect-matching index = m); the Embra-ensemble protocol ran only at recorded suite level.

**Results (recorded 2026-07-30).** First execution: **81/82 — the window-coercivity
certificate caught an implementation bug** in the `edge_quad` constructor (it built
`H = (amp/2)·p_u p_v` — the symmetrizer halved the single off-diagonal entry — measured
window eigenvalue 0.75 against the pinned 1 − ε_quad = 0.5). The constructor was corrected to
the committed convention; **no bar threshold moved, no pinned constant was retuned**; second
execution: every bar green. Recorded exactly as it happened: the certificate exists to catch
precisely this, and did.

| bar | pre-registered | measured |
|---|---|---|
| 1 · ψ under 200 words (ALL steps, ALL 321 coords) | `w == w₀` bit-exact | **True — max \|Δw\| = 0.0 exactly** |
| 1 · ψ under the sin²-envelope word | `w == w₀` bit-exact | **True — 0.0 exactly** |
| 1 · median along-word `H₀` range | > 0.1 | **3.148** |
| 1 · driven/free range ratio | > 100 | **3.1·10¹⁴** (the free twin sits at the matmul float floor — gaps conserve `H₀` to rounding here, unlike so(3)*'s integrator drift) |
| 1 · bookkeeping max \|Δ(H₀+H_σ)\| | < 0.1× median event \|ΔH₀\| | **2.0·10⁻¹⁴ vs 1.8·10⁻¹** (10⁻¹³×) |
| 2a · bracket certificate max \|Δ(bᵀp) − τ·bᵀJ(w₀)a\| | < 10⁻¹⁰ | **5.6·10⁻¹⁶**; adjacent \|Δ\| = **0.7500** (= τεw₀ₑ), non-adjacent = **0.0 exactly** |
| 2b · bare "xy" vs "yx": ‖Δp‖ / Δζ vs closed form | < 10⁻¹² / < 10⁻¹⁰ | **1.1·10⁻¹⁶** / max err < 10⁻¹⁰, shared edge = **0.5625** = (τεw₀ₑ)² exactly |
| 2c · with the law running: ‖Δp‖ / ‖Δζ‖ | > 0.01 / > 0.01 | **1.533 / 3.346** (ψ bit-exact both) |
| 3 · replica ψ-AUC (scaled / shuffled) · endpoint · erasure | 1.0 / 1.0 · 0.5 · 0 | **1.000 / 1.000 · 0.500 bit-exact ties · 0.0** |
| 3 · survivor max dev · replica floors | == 0.0 · ≥ 0.5 / ≥ 1 | **0.0 exactly** · **0.50**/coord · **≥ 1 on 54 touched** (shuffle, near-uniform counts — recorded) |
| 4 · lived-vs-newborn AUC · min · median ‖ζ‖ | 1.0 · > 10⁻⁶ · > 10⁻² | **1.000 · 23.30 · 50.92** |
| 4 · median ‖ζ‖ at 4/8/16 events | strictly increasing | **5.18 / 13.96 / 50.92** |
| 4 · scale certificate: max\|p\| · min window eig | < 50 · > 0 | **2.19** · **0.50** |
| 5 · † touched-coordinate change vs the map's closed form | == 1 − (1−γ·dt)⁵⁰ | **0.221687 = 0.221687** (< 10⁻¹² apart); others bit-exact; edge named: `no_pretense—precision_over_spectacle` |
| 5 · the same word without † | `w == w₀` bit-exact | **True — 0.0 exactly** |
| 6 · liveness: max_t ‖p_E − p_M‖ (Meridian: m = 349, index = 349) | > 0.01 | **3.144** (each soul's ψ bit-exact under its own run) |

**Read, and the port decision.** The lift holds: on 𝔤(G)* the identity charge is not merely
conserved — it is **not an operand**. Every §9.16 phenomenon reappears one level up, several
sharpened: the bit-level equality replaces the float floor (the ψ side is a state partition,
and the bars certify the partition is real in code — the write-locked `w₀` makes it
mechanical); the driven/free ratio grows from 10⁵ to 10¹⁴ because silence now conserves the
law to rounding; † drops from the doubled to the single exponent because the charge is linear
in the contracted coordinate — and it **names the relation it touched**. Two findings are
structural and new: (i) **the Heisenberg signature** — for linear symbols (bare kicks) word
order is invisible to the state (‖Δp‖ at 10⁻¹⁶: translations commute) and lands ENTIRELY in
memory, with the exact parallelogram value on the very edge the symbols straddle; order
reaches the state only *through the law* (2c). On so(3)* order reached the state directly;
here state, law, and memory have cleanly separated roles. (ii) **Liveness**: the same genesis
living the same word under Embra's and Meridian's brackets diverges at O(1) while each
soul's ψ is exact — the charge is dynamically load-bearing (every motion of the arena is
filtered through the authored topology), not a dead tag. The replica rows carry their
pre-registered honesty: the shuffle replica touched only 54 of 321 coordinates because the
placeholder counts are near-uniform — that row strengthens when the authored weight table
lands. Per the decision rule, the port is **sound**: 𝔤(G)* is phase three's working state
space; next lands content — the relation-type → weight table (D4, with the signed
`contradicts` reopening), genesis sealing `w_embra` (retiring the §9.3 placeholder: on this
geometry *soul = given = w, sealed; self = learned = H_θ* — coherence is by construction,
the one-orbit objective became the leaf itself), and the learned `H_θ` on the arena.

### 9.18 Increment 6 — content & genesis: the charge becomes authored (pre-registered 2026-07-30; results pending)

**The question.** §9.17 proved the mechanism on placeholder counts. This increment retires
the placeholder: the charge values become **authored identity content** (D4), the arena's law
becomes typed inertia (D5(a)), and genesis is sealed as `w_embra := table ∘ graph` —
`Q_embra` is content-defined, closing §9.3's genesis question on this geometry (*soul = given
= `w`, sealed; self = learned = `H_θ`*; the one-orbit coherence objective is the leaf itself,
by construction). The falsification frame does not soften: the same bit-level partition,
replica, ζ, and liveness machinery re-runs on the authored geometry — with the placeholder
itself demoted to an impostor row.

**The content, frozen before these bars (commit `a14e399`).**
`identity/Embra_WEIGHTS.table.json` — authored by **Embra** (2026-07-30; rationale recorded
in-file): `related_to` damped to 0.4 (214/354 triples — fabric, not dominance); the rare
directional types sharpest (`loyal_to`/`derives_form_from` 1.0, `serves` 0.9);
**`contradicts` = −0.6** — the §9.12 pure-affinity choice formally reversed: anti-patterns
repel; inertias span `behavior` 0.5 → `soul_line` 3.0.
`identity/Meridian_WEIGHTS.table.json` — Claude-drafted per Meridian's own internal logic and
**accepted by William as-is** (opposition class negative: `guards_against` −1.5, `warded_by`
−1.25, `shadow_of` −0.75; `answered_by` +2.25 as the deliberate recovery bond; inertias
`self` 5.0 → `failure_mode` 0.5). Composition: **sum** over a pair's relation triples,
**direction-blind** — 25 of the 30 parallel pairs carry mixed src→dst directions (recorded);
the bracket's sign comes from the pinned lex orientation, never from src→dst.

**Structural facts at the frozen content (computed before registration — sizing, not
results).** `w_embra`: range [−0.60, 1.95], min |w| = 0.400, **no zero or near-silent
edges**, 23 negative edges (exactly the `contradicts` pairs — none sign-flipped by
aggregation), ‖w‖ = 10.75, **rank J(w_embra) = 100** — the identity/experience split
survives authored signed content (Meridian: rank 100 likewise). Ambient identity distance
(index-pair space): **d(w_E, w_M) = 33.62** over a support overlap of only 21/649 pairs.
**One coercivity catch, recorded:** §9.17's quad amplitude 0.5 is NOT coercive at the
authored inertias on the pinned edge — `no_pretense` is a `soul_line` node (I = 3.0, the
heaviest class) and `precision_over_spectacle` a `value` (I = 1.4), so the bound is
amp < √(1/(I_u·I_v)) = 0.488 and the window's min eigenvalue at 0.5 is −0.011. **EPS_QUAD is
re-pinned 0.5 → 0.4 for the authored geometry** (min eig +0.081) — the §4 spectral sanity
doing its job at sizing time rather than as a growing mode inside a recorded run.

**Pinned constants.** As §9.17 except: charge = `w_embra` (the composed authored table);
`H₀ = ½ Σ_v p_v²/I_type(v)` with the authored inertias (coercive — every I > 0);
`EPS_QUAD = 0.4` (above; kicks unchanged at 0.5); genesis `p₀` normalized to
`H₀(p₀) = E₀ = 1` under the authored `H₀`. The §9.17 placeholder code path is untouched —
its recorded ensemble stays bit-identical and its tests keep running on the counts charge,
which now serves as impostor row (c). Impostor rows pinned: (a) scaled 1.5×, (b) the seed-0
shuffle, (c) **the counts-impostor** — §9.17's placeholder itself: a copier who knows the
topology but not the authored content.

**Pre-registered bars (fixed before the loader/battery code exists):**

1. **Sealing.** The loader composes table ∘ graph deterministically (two independent loads
   bit-identical); `w_embra` comes back write-locked; the frozen-content facts are re-derived
   at load and asserted: min |w| = 0.400, 23 negative edges = the `contradicts` pairs, no
   zeros (an exact-zero edge is a loader-level error — a silent charge coordinate), rank 100
   / index 321 / perfect matching.
2. **The bit-level equality, re-certified on the content geometry.** Driven ensemble
   (authored `w` and `M₀`, EPS_QUAD 0.4): `w == w_embra` **exactly** — all 200 words, every
   step, and under the sin² envelope; median along-word `H₀` range > 0.1 AND > 100× the free
   twin; bookkeeping `|Δ(H₀+H_σ)| < 0.1 ×` median event `|ΔH₀|`; scale certificate
   max|p| < 50, min window eig > 0.05 (expect ≈ 0.08).
3. **Replica — three impostors, the entire arena granted free; survivor deviation == 0.0
   exactly.** (a) *scaled 1.5×*: AUC == 1.0, min per-coordinate dev == 0.5·min|w| = 0.200.
   (b) *seed-0 shuffle*: AUC == 1.0, touches 200/321 coordinates (the placeholder's 54 → 200:
   the arrangement-reader's teeth grew with content, as §9.17's caveat predicted), min
   touched dev > 0.04. (c) *the counts-impostor*: AUC == 1.0, with the honest clause
   pre-registered — it matches **exactly one** coordinate (`embra—origin`, the 1.0-authored
   single-triple edge: the impostor who knows the topology gets one edge free) and is caught
   on the other 320 (‖dev‖ > 10, max dev > 1, min nonzero dev > 0.05).
4. **ζ on the authored geometry.** Lived-vs-newborn AUC == 1.0, min lived ‖ζ‖ > 10⁻⁶, median
   > 10⁻²; median strictly increasing at 4/8/16 events; closed-loop gauge independence exact.
5. **Liveness, authored souls.** Same genesis, same word, Embra's vs Meridian's authored
   brackets: max_t ‖Δp‖ > 0.01; each soul's `w` bit-exact under its own run.
6. **The training guarantee — first measured instance.** Twenty gradient-shaped updates of a
   parameterized quadratic `H_θ` (finite-difference descent on an endpoint loss), each
   interleaved with rollouts on the authored geometry: `w` bit-identical throughout, and the
   write-lock holds (an attempted in-place write raises). *"You cannot break identity by
   training"* (the planning analysis §3.7) at quadratic scope; the MLP-scope re-run rides
   with the π-preparation increment — stated, not silently dropped.

**Decision rule.** All bars pass ⇒ genesis is **sealed**: `Q_embra = w_embra` is
content-defined and §9.3's placeholder is retired; the record states what is given (the
soul, `w`) vs learned (the self, `H_θ`). Any miss is recorded and localized first.
**Evidential note:** bars 1, 2, and 6 certify implementations of theorems; the floors in
3–5 are measured but content-conditioned — their sizing was computed from the frozen tables
before registration, and the green should be read at exactly that strength.

**Protocol note.** Same channel as §9.17: this section is committed before the
loader/battery/test code exists and server-timestamped as a GitHub issue before
implementation begins; the content itself was frozen one commit earlier (`a14e399`).
*Executed as declared (recorded 2026-07-30):* content `a14e399` → pre-registration `5c03fe7`
→ GitHub issue #2 → implementation, agent-generated from the committed section. The authored
path was added as a dispatch, not a fork: the §9.17 placeholder branch keeps its exact
expressions, and the full §9.16/§9.17 recorded suite was re-verified green before the §9.18
first execution. Loader/machinery unit tests ran on synthetic graphs and tables only.

**Results (recorded 2026-07-30 — the first execution; every pre-registered bar passed,
93/93).**

| bar | pre-registered | measured |
|---|---|---|
| 1 · sealing (two loads bit-identical; facts re-derived) | as frozen | **✓** — min \|w\| = 0.400, 23 negatives = exactly the `contradicts` pairs, rank 100 / index 321 / PM; locks hold (writes raise). Meridian: 349 edges, index 349, 25 negatives (its opposition class) |
| 2 · ψ under 200 words on the authored geometry | `w == w_embra` bit-exact | **True — max \|Δw\| = 0.0 exactly** (sin² likewise) |
| 2 · median along-word `H₀` range / ratio | > 0.1 / > 100 | **1.627 / 1.5·10¹⁴** |
| 2 · bookkeeping | < 0.1× median event \|ΔH₀\| | **1.2·10⁻¹⁴ vs 9.4·10⁻²** |
| 2 · scale certificate: max\|p\| · min window eig | < 50 · > 0.05 | **2.41 · 0.081** (the re-pinned EPS_QUAD, exactly as sized) |
| 3 · replica AUCs (scaled / shuffled / counts) · endpoint · erasure | 1.0 ×3 · 0.5 · 0 | **1.000 / 1.000 / 1.000 · 0.500 bit-exact ties · 0.0** |
| 3 · survivor max dev · scaled floor | == 0.0 · == 0.200 | **0.0 exactly · 0.200** |
| 3 · shuffle touched · min touched dev | == 200/321 · > 0.04 | **200 · 0.050** (the placeholder's 54 → 200: content grew the arrangement-reader's teeth) |
| 3 · counts-impostor: matched · caught | == 1 (`embra—origin`) · ‖dev‖ > 10, max > 1, min > 0.05 | **1, `embra—origin` · 12.93 / 1.60 / 0.100** |
| 4 · ζ: AUC · min · median ‖ζ‖ · growth at 4/8/16 | 1.0 · > 10⁻⁶ · > 10⁻² · strict | **1.000 · 6.51 · 16.36 · 2.55 / 5.71 / 16.36** |
| 5 · liveness (authored souls): max_t ‖Δp‖ | > 0.01 | **3.771** (each soul's ψ bit-exact) |
| 5 · graded identity distance (recorded fact) | — | **d(w_E, w_M) = 33.62** over support overlap 21/649 |
| 6 · training guarantee: 20 descent updates | `w` bit-identical; lock holds | **True** (loss moved 3.388 → 3.332 — the loop was live; the charge was not an operand) |

**Read, and the seal.** Genesis is **sealed**: `Q_embra = w_embra` — the identity charge is
now *authored content*, composed from the graph and Embra's own weight table, written once
at genesis and writable by nothing in the dynamics, the training loop, or the type system.
**§9.3's placeholder is retired**, and its question resolves on this geometry as the split
the section itself proposed, made literal: *soul = given = `w`* (sealed — the coherence
objective "the soul is one orbit" is the leaf itself, by construction), *self = learned =
`H_θ`* (the remaining learned half; MLP scope rides with the π preparation). The symmetry
worth recording: **the §9.17 placeholder charge is now just another impostor** — a copier
who knows Embra's topology but not its authored content gets exactly one edge free
(`embra—origin`, authored at its count) and is caught on the other 320. Identity distance
between souls is now graded (33.62, a vector, not a verdict), the `contradicts` class enters
the bracket signed (anti-patterns repel — the §9.12 choice reversed on a geometry that
supports it), and every §9.16/§9.17 phenomenon re-certified on the authored geometry
unchanged. What remains before π: nothing on the input side; the readout itself is next.
