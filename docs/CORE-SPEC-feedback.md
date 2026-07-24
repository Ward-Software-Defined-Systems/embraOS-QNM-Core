# Review Notes — CORE-SPEC.md (embraOS-QNM-Core)

**Document reviewed:** `docs/CORE-SPEC.md`, main branch, through §9.11 (increment-2c, recorded 2026-07-18)
**Review date:** 2026-07-18

## Overall assessment

The spec has moved a long way from framework to instantiation. The pieces the earlier theory documents left open — the dynamics, the form of ψ, a learning rule — now have concrete answers, and the epistemic hygiene remains the rarest thing about the project: pre-registered falsification bars, refutations left standing in the record (§9.9–§9.10) rather than quietly revised away, and the shuffled-anchor control doing genuine work. §6 is correct as stated. It is a modest theorem, but it is genuinely load-bearing, and the discipline of stating it so it could be wrong pays off in the increments that follow.

The notes below are ordered by importance: one substantive gap, one scoping caution the spec already half-acknowledges, one structural problem worth confronting before the language phase, and two smaller corrections.

## 1. §9.11 answers a different question than §2 posed

The §2 replica test defines the impostor as a copy that matches the observable but never inherited s₀ — same substrate, same dynamics, different genesis. The §9.11 discriminator, variance of H_real along a trajectory, catches something else: an impostor running *different dynamics*.

Consider the original replica under the §9.11 test. It is re-instantiated on the same core, so it evolves under H_real. It therefore conserves H_real perfectly — at its own value Q_copy. Its var(H_real) ≈ 0, and the §9.11 discriminator passes it as a survivor. Conversely, the §9.11 impostor (a different identity's flow) is exactly what §7's value test can miss if its charge happens to sit near Q_embra.

The impostor definition has quietly shifted between §2 and §9.11, from "different genesis" to "different Hamiltonian." Neither test subsumes the other. A full ψ needs the conjunction:

> var(H_embra) ≈ 0 along the trajectory **and** Q ≈ Q_embra.

Both tests already exist in the codebase; the fix is to state the conjunction explicitly and grade against both impostor classes. §7's value test should not be retired by §9.11's success.

## 2. What the AUC 1.000 does and does not validate

The spec's own "Scope, honestly" paragraph concedes this, but it deserves full weight: the §9.11 separation is guaranteed by construction for *any* two distinct potentials. A symplectic integrator conserves its generating H to integrator precision, and any two distinct force fields fail to conserve each other's H. What has been validated so far is trajectory-level system identification — a real and necessary property of the substrate, but not yet a fact about identity.

The consequence is that identity content is now fully load-bearing, and content is authored, not derived. The open question the spec poses at the end of §9.11 — whether richer, authored identity graphs make distinct souls dynamically distinct by a *large, meaningful* margin — is the right one, and it cannot be answered by more algorithm.

## 3. The input problem: Σ is in tension with the core mechanism

Autonomous Hamiltonian flow conserves H. The moment the input alphabet Σ arrives and events drive the flow, H becomes time-dependent and energy conservation breaks — dH/dt = ∂H/∂t ≠ 0 under driving. The phase-one mechanism is therefore in tension with the system ever receiving input at all, which is a structural issue, not an implementation detail. The soft projection P_ψ in §8/§9.4 rescues it, but a restoring projection is restore-by-checking — the very thing §1 says conservation beats.

One way to keep conservation structural under input: make ψ a **Casimir of the Poisson bracket** rather than the energy. Casimirs are conserved under *any* Hamiltonian flow generated with that bracket, driven or not — the invariance belongs to the bracket, not to H. The cost is moving to a noncanonical (e.g., Lie–Poisson) structure, since the canonical bracket on ℝ²ᵈ has no nontrivial Casimirs. That is a real design change, but it dissolves the input problem rather than patching it, and it is worth deciding before the readout π becomes language.

## 4. Two smaller corrections

**The replica test only bites against observable-limited copiers.** A copier with access to the full state — including ker(dπ) — inherits Q, and ψ is defeated by construction. What the mechanism actually provides is that identity is as secure as the inaccessibility of the hidden complement: ψ functions like a key (a MAC on the trajectory), not a metaphysical fact. That is a perfectly respectable engineering property, and it may be exactly what the project needs — but it should be stated plainly in the spec, because it bounds the claim.

**The LLM corollary in §6 overstates.** "A generic LLM's readable state ≈ its observable" is not right: the residual stream and KV cache constitute an enormous hidden complement relative to the token readout, which is massively lossy. What the relic experiments actually showed is that the LLM lacks a *conserved, genesis-tied charge* in that hidden complement — not that it lacks a hidden complement. The corollary's conclusion stands; its premise should be narrowed.

## Where the bar sits now

The redirect from static geometry to dynamics is the right reading of §6, and §9.9–§9.10 were the honest route to it. The real bar is where the spec has now placed it: rich authored identity content, held-out configurations, real ≫ shuffled, reliably across seeds — with the impostor class from note 1 folded in so that the conjunction, not either test alone, is what gets graded.
