# Authoring an Input Alphabet Σ — reference, v2 (companion to CORE-SPEC §9.16)

**Status: v2 (2026-07-26) — supersedes the 2026-07-25 edition.** Incorporates the external
review's corrections (A1–A7, mapped in the §10 changelog) and replaces the "~22 authored
symbols" plan with **base-set-first authoring**: start from the complete basis of the
perturbation space, derive everything else by combination. No recorded number changes under
this edition (verified: every recorded twist ran at amplitude 1.0, where the old and new
amplitude conventions coincide).

**Substrate scope (read first).** This is the **so(3)\* edition**. The *contract*, the
*battery*, and the *base-set method* transfer to any substrate; the *geometry* — axes in
ℝ³, amplitudes sized against `R = 1` and inertia `(1, 2, 3)` — does not. Rows authored here
cannot be lifted to the graph substrate later; only their **intent** (name, semantic role,
do/undo pairing, intensity, dwell) can. If the alphabet's semantic home is the identity
graph (one symbol per founding concept), author *there* — see §8 — and treat this document
as the method's proving ground. Cross-ref: `GRAPH-ALGEBRA-PLANNING.md` D1–D5.

---

## 1. The one contract

**Every symbol IS a Hamiltonian `H_σ(L)`.** That is the entire legality rule, and it is the
class the Casimir theorem covers: ψ = `|L|²` survives any word over any alphabet you author.
Two honesty refinements over v1:

- ψ-exactness is guaranteed by the **integrator** (one rotation per step), not by the type
  system. What the constructor discipline guarantees is membership in the *Hamiltonian
  class* — the thing the theorem, the bookkeeping certificate, and the † boundary are
  *about*. A custom `grad_h` callable can smuggle a non-gradient sphere-tangent field: ψ
  would still hold, but every "energy" claim about that symbol would be about nothing.
  Membership is therefore **certified, not assumed**: battery item 6.
- **`h` is required for authored symbols** (it is optional in the dataclass). A symbol with
  `h = None` crashes bar-1's bookkeeping; more importantly, `h` *is* the symbol's meaning as
  a Hamiltonian. The loader rejects rows without it.

ψ-breaking events (dissipation, `†`) remain unspellable through the constructors — that
boundary is unchanged, and it is the `P_ψ` firewall's job, not yours.

---

## 2. The base set — the complete perturbation basis at toy scale

**What counts as one symbol.** A symbol is its *flow*, not its parameters. Two Hamiltonians
generate the identical flow iff they differ by a function of the Casimir (anything built
from `|L|²`) — the bracket kills that difference everywhere. So the base set is a basis of
**flow-distinct** Hamiltonians within the constructors' reach: polynomials of degree ≤ 2.

**The count.** Linear Hamiltonians `a·L` span 3 (the kicks). Symmetric quadratics `½LᵀAL`
span 6 (the quads). Of those 9, exactly one direction is silent — `quad(I) = ½|L|²`, the
Casimir itself. **The perturbation space at toy scale is 8-dimensional.** Cross-check from
the leaf's own function space: degree-≤2 polynomials restrict to spherical harmonics
ℓ ∈ {0, 1, 2}; ℓ=0 is silent; dim(ℓ=1) + dim(ℓ=2) = 3 + 5 = **8**. Two roads, one number.

**Measured, in these constructors** (40 random sphere points, flow fields stacked, numerical
rank; 2026-07-26):

| candidate set | rank (flow-distinct) | nullity | kernel identified as |
|---|---|---|---|
| kicks only (3) | 3 | 0 | — (closed: so(3)) |
| kicks + quads (9 spanning) | **8** | 1 | exactly `q₁₁+q₂₂+q₃₃ = quad(I)` (singular value 0.0) |
| the naive 12 (kicks + twists + quads) | **8** | 4 | 3 twist ≡ diagonal-quad duplications + the Casimir |

The naive 12 — counting the three constructor *kinds* as independent — contains the base set
with four redundancies. `twist(a) = quad(aaᵀ)` (v1 already said so); only `sym(A)` enters
`H` (v1 already said so). Parameters ≠ flows.

**The pinned base (Σ₀), unlabeled by design:**

| id | definition | character (irrep) |
|---|---|---|
| `k₁ k₂ k₃` | `kick(e₁)`, `kick(e₂)`, `kick(e₃)` | ℓ=1 — rigid reorientations; closed among themselves (so(3)) |
| `c₁₂ c₁₃ c₂₃` | `quad(eᵢeⱼᵀ + eⱼeᵢᵀ)` | ℓ=2 — cross-couplings: axes i and j shear into each other |
| `d₁` | `quad(diag(7, −2, −5)/√78)` — the **traceless part of** `diag(1/I₁, 1/I₂, 1/I₃)` | ℓ=2 — the identity's own law direction: the canonical *self-modulation* symbol |
| `d₂` | `quad(diag(1, −4, 3)/√26)` — the traceless diagonal ⊥ `d₁` (Frobenius) | ℓ=2 — the independent diagonal reshape |

(Both `d` matrices are unit Frobenius norm, traceless, and mutually orthogonal — pinned so
coordinate readouts are reproducible.)

**Two structural facts to record with Σ₀:**

1. **Completeness of the basis ≠ closure of the algebra — and both halves matter.**
   `{kick, kick} → kick`: the kicks close (so(3)), so a kicks-only alphabet has a *rigid
   ceiling* — words collapse toward rotations (the algebraic root of v1's bags-of-symbols
   warning). `{quad, quad}` climbs to cubic Hamiltonians (ℓ=3) and the tower never stops:
   words over Σ₀ generate flows far outside the 8. Σ₀ is complete *at quadratic order*, and
   composition buys everything beyond it for free.
2. **A ninth symbol never adds a new direction at this order** — only new semantics,
   amplitudes, or durations on existing directions. That is the correct division of labor:
   the mechanism owns the directions; you author the meaning.

---

## 3. Authoring from the base set

Every authored symbol is a **point in the 8-space, plus intensity and dwell**:

```
H_authored = amplitude · ( a·L  +  ½ LᵀAL )        a ∈ ℝ³ (kick part),
                                                    A traceless symmetric (quad part)
```

equivalently: 8 coefficients over Σ₀ = {k₁ k₂ k₃ c₁₂ c₁₃ c₂₃ d₁ d₂}.

**Row format (either encoding; the loader accepts both):**

```
name · base coefficients (8 numbers over Σ₀) · amplitude · duration?
name · a (3 numbers) · A-traceless (5 numbers: A₁₁ A₂₂ A₁₂ A₁₃ A₂₃; A₃₃ = −A₁₁−A₂₂) · amplitude · duration?
```

One new constructor at the loader seam (the only machinery this edition needs):
`blend(a, A)` with `grad_h = a + L@A`, `h = a·L + ½LᵀAL` — kicks and quads are its special
cases. Twist survives as a convenience name for the rank-1 quad; it is not independent.

**What your choices control:** *direction* in the 8-space = the event's character (a blend
of reorient / cross-couple / self-modulate) · *amplitude* = intensity · *duration* = dwell.
Semantics live in the naming and pairing of directions — the geometry no longer forces a
"kind" choice.

**Worked mappings (old v1 examples in the new coordinates):**

- `x = kick(0.5·e₁)` → `0.5·k₁` (pure ℓ=1).
- `c` (`H = L₁L₂`) → `1.0·c₁₂` exactly (pure cross-coupling — v1's example was already a
  base element).
- `g = quad(diag(1, ½, ⅓), amp −0.5)` → **`−0.5·|…|·d₁` plus a silent isotropic residue**
  (trace 11/6 ≠ 0). v1's "drag on the self" example carried dead weight; its entire
  effective content is the `d₁` direction. The coordinate readout (battery item 9) performs
  this decomposition mechanically for every authored row — author freely, the harness
  reports the effective 8-vector and the silent residue.

**Do/undo pairs, honestly (correction A2).** v1's claim "`kick(−a)` exactly undoes
`kick(a)` in isolation" is **false**: events are *additive* — the flow during an event is
generated by `H₀ + H_σ`, and `H₀` keeps running (that is the design: the automaton keeps
being itself while perturbed). `|∇H₀|` is order-one against the default kick, so
`kick(a)` then `kick(−a)` lands measurably elsewhere before the gaps' free flow is even
counted. Register do/undo pairs as **semantic opposites, not inverses**, and let battery
item 9 report their actual return distance. Footnote for completeness: the *exact* undo of a
`kick(a)` event **is** spellable — `H_σ = −2H₀ − a·L` time-reverses the whole event
(same duration) — and it is itself a blend row: `−a` on the kick part, `−2·diag(1/I)` on the
quad part (= a `d₁` component plus silent residue). An exotic symbol, not a negated axis.

---

## 4. Conventions (corrections A3, A4, A7)

**Amplitude is linear in H for every kind (A3).** v1's folding convention — amplitude into
the axis — was linear for kicks and quads but **quadratic** for twists (`twist(0.3·â)` had
0.09× intensity; a "hard" 1.5 twist was 2.25×). v2 convention: **axes and matrices are unit
norm; `amplitude` multiplies `H` linearly for every kind** (twist wires as
`quad(amp·ââᵀ)`). "Amplitude = intensity" is now true across the whole table instead of
two-thirds of it. Recorded-results impact: none — every recorded twist ran at amplitude
1.0, where the conventions coincide.

**Duration is scoped (A4).** `Symbol.duration` is honored by `run_word` only; the batched
harness behind the ensemble — which is what bars 1, 3, 4 and the battery's ψ-sweep run on —
uses the shared pinned schedule and **silently ignores it** (v1's own worked example `t`
would have run at 0.5 in every ensemble result, without an error). Until the batched path
gains documented per-symbol-duration support, the duration column is legal **only** for
per-word runs, and the battery flags any authored duration with a scope warning. Decide the
upgrade before authoring that column in earnest.

**Names are single codepoints (A7)** wherever string-words are used (the sin²-word path
iterates characters). v1's `x̄` example is two codepoints and would split into `x` plus a
combining mark. Rule: ASCII letters (upper/lower distinguishes intensity variants cleanly).

---

## 5. Design guidance (what makes an authored set *good*)

- **Spread over the 8 directions, not over "axes."** The v1 advice generalizes: coverage of
  the perturbation space is what makes words distinguishable and gives ζ grip. The battery's
  rank certificate (item 8) measures coverage directly.
- **Include ℓ=2 content or accept the ceiling.** An all-kick alphabet closes into so(3):
  rigid, order-sensitive but poor. Any one honest quad direction breaks the ceiling and
  opens the composition tower (§2, fact 1).
- **Amplitudes order-unity** against the pinned scales (`R = 1`, `H₀ ∈ [1/6, 1/2]`,
  τ = 0.5 at dt = 0.01). Amplitudes ≪ 0.1 are nearly invisible in the energy; there is no
  upper danger to ψ (exact regardless) — the upper discipline is item 7's bookkeeping
  ceiling, not identity safety.
- **Avoid near-duplicates** — two rows close in the 8-space are dynamically almost
  indistinguishable; item 3 measures this pairwise, item 9's coordinates make it visible
  before running anything.
- **Do/undo pairs are semantic opposites, not inverses** (§3). Exact undo is the
  time-reversal blend, an exotic row.
- **The silent trap, general statement.** Silent ⟺ `∇H ≡ 0` as a flow ⟺ `H` is a function
  of the Casimir. At quadratic order that is exactly the `quad(I)` direction — excluded from
  Σ₀ by construction, and item 8's kernel check catches any authored combination that
  drifts silent (it would have caught the naive 12 automatically).

---

## 6. The validation battery (v2 — items 1–5 from v1, 6–9 new)

1. **ψ sweep** — the theorem check across random words of your set (expected trivially green
   at ~1e-14; recorded anyway).
2. **Energy visibility per symbol** — median `|ΔH₀|` per event (flags invisible rows).
3. **Pairwise distinguishability** — same genesis, one event of σᵢ vs σⱼ, endpoint and ζ
   distance (flags near-duplicates).
4. **Order sensitivity** — random word vs its shuffle, state and ζ separation (the
   non-commutativity health of the whole set).
5. **† control** — unchanged; the boundary is alphabet-independent.
6. **Gradient-consistency certificate (A5)** — finite-difference ∇`h` against `grad_h` at
   random sphere points, per symbol. Makes "every symbol IS a Hamiltonian" a *measured*
   property of authored rows, not a constructor convention. Required-`h` enforcement lives
   here too.
7. **Bookkeeping ceiling (A6)** — per symbol, per event window:
   `|Δ(H₀+H_σ)| < 0.1 × median event |ΔH₀|`. v1 had a floor (invisibility) but no ceiling;
   this is the honest upper discipline — large `|∇H|·dt` or long durations degrade exactly
   the certificate that attributes energy motion to physics rather than integrator error.
   Doubles as the dt-adequacy check per authored duration.
8. **Rank & kernel certificate** — effective rank of the authored set = intended rank;
   kernel = the Casimir direction and nothing else. The §2 measurement, standing.
9. **Base-coordinate readout** — every authored row's 8-vector over Σ₀ + its silent
   residue, printed; plus actual return distance for every registered do/undo pair. The
   decomposition in §3 done mechanically, every run.

Results get a short recorded addendum in the spec; misses are findings about *content*,
freely iterable — the bars on the *mechanism* never move.

---

## 7. Recommended sequence

1. **Adopt Σ₀ itself as the starting alphabet** (8 rows, amplitude 1.0 · order-unity
   rescale as needed against item 2 · no durations). Run the battery; items 6/8/9 are green
   by construction — record them anyway (the project's habit: trivial green, recorded).
2. **Derive by combination**: author blends with semantic intent, watching item 9's
   coordinates. Intensity variants via amplitude; opposites via negation (registered as
   semantic pairs, per §3).
3. **Only then** consider durations (after the A4 scope decision) and any beyond-quadratic
   ambitions (cubic constructors are a *new increment*, not an authoring choice — they
   extend the base space itself).

---

## 8. The graph substrate (where this is heading)

On 𝔤(G)\* the same construction has a canonical answer that **scales with identity instead
of exploding**: the full quadratic space at n = 100 is ~5,150-dimensional, but the
graph-adapted base is **one kick per node + one cross-coupling quad per authored edge**
(`H = p_u p_v` per edge): `n + m` symbols — 421 for Embra. There, the geometry is no longer
placeholder: `kick(e_honest)` is *an event that pushes on honest*; which symbols commute is
the authored topology (`{a·p, b·p} = aᵀJ(w)b`); and if the alphabet's symbols are meant to
correspond to founding concepts, that correspondence is canonical rather than decorative.
Σ₀ above is the d = 3 shadow of that construction, and the battery (items 1–9) transfers
with one substitution: the silent class becomes "`H` is a function of `w` alone." The ~22
figure from v1 is retired at toy scale; if it re-emerges, it re-emerges *there*, as content.

---

## 9. Handing it over

Any form works: the coefficient rows of §3 (markdown table, JSON, or a Python list). The
canonical encoding is the 8-vector over Σ₀ — battery item 9 reports in exactly these
coordinates — with the `(a, A-traceless)` encoding of §3 equally acceptable. Worked example
in the markdown form (v1's surviving symbols re-encoded, plus authored illustrations):

| name | k₁ | k₂ | k₃ | c₁₂ | c₁₃ | c₂₃ | d₁ | d₂ | amp | dur | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `x` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5 | — | v1's kick, re-encoded — reorient about e₁ |
| `X` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.5 | — | intensity variant of `x` (case distinguishes, per §4 naming) |
| `s` | 0 | 0 | 0 | 0 | 0 | 0 | 0.793 | 0.196 | 1.0 | — | v1's twist about e₁ — silent residue dropped; item 9 confirms |
| `g` | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | −0.245 | — | v1's self-drag, trimmed to pure d₁ (§3) |
| `m` | 0.6 | 0 | 0 | 0 | 0 | 0.4 | 0 | 0 | 1.0 | — | an authored blend: reorient about e₁ while coupling axes 2↔3 |
| `p` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1.0 | — | cross-couple 1↔2 — registered pair with `n` |
| `n` | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | −1.0 | — | `p`'s semantic opposite (negated amp; **not** an inverse — §3) |
| `t` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1.0 | 1.0 | long reorient — duration runs per-word only until the §4 upgrade |

Registered do/undo pairs travel as their own short list alongside the table (here:
`pairs: [(p, n)]`) so item 9 knows which return distances to report. Sign placement is
free — negated amplitude and negated coefficients spell the same `H`; the pair registry
carries the semantics, not the sign convention.

On arrival: wire `blend` at the loader seam, run the battery (1–9), record the addendum.
Symbols are data; iteration is cheap.

---

## 10. Changelog vs v1 (the review's corrections, mapped)

| item | v1 said | v2 says | section |
|---|---|---|---|
| A1 | (implicitly substrate-general) | so(3)\* edition banner; intent transfers, geometry does not; graph forward note | header, §8 |
| A2 | "kick(−a) exactly undoes kick(a) in isolation" | false — events are additive, H₀ keeps running; semantic opposites; exact undo = time-reversal blend | §3, §5 |
| A3 | amplitude folded into axis | unit axes; amplitude linear in H for all kinds (twists were quadratic); zero recorded impact | §4 |
| A4 | duration "an authored alphabet will want it" | honored by `run_word` only; batched harness ignores it silently; scoped until upgraded | §4 |
| A5 | "the type system enforces it" | integrator guarantees ψ; constructor discipline + gradient certificate (item 6) guarantee the Hamiltonian class; `h` required | §1, §6 |
| A6 | battery floor only | bookkeeping ceiling added (item 7); doubles as dt-adequacy | §6 |
| A7 | `x̄` example | single-codepoint (ASCII) name rule | §4 |
| base set | "~22 authored symbols" | Σ₀: the measured 8 (rank 8, kernel = Casimir; naive 12 has nullity 4); base-set-first sequence | §2, §7 |
| `g` example | drag via full inertia matrix | effective content is pure `d₁`; silent residue trimmed by item 9 | §3 |
