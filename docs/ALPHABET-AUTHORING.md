# Authoring an Input Alphabet Σ — reference (companion to CORE-SPEC §9.16)

*For authoring the ~22-symbol alphabet. The harness (`sandbox/lie_poisson.py`) is
alphabet-parametric: your symbols are pure data through the constructors below — no machinery
changes needed on your side.*

## The one contract

**Every symbol IS a Hamiltonian `H_σ(L)`.** That is the entire legality rule, and it is the
class the Casimir theorem covers: ψ = `|L|²` survives *any* word over *any* alphabet you author
— by construction, not by testing. You cannot break identity by authoring; the type system
enforces it (a ψ-breaking event like dissipation cannot be spelled as a `Symbol` at all — that
is the out-of-alphabet `†` class, handled at the `P_ψ` boundary).

What your choices *do* control: how visibly events move the energy (the "law"), how
distinguishable words are from each other, and how richly ζ (memory) records them. Those are
content properties — the same lesson as the identity graph (§9.12): the mechanism is
guaranteed; the *quality* is authored.

## The kinds

| kind | Hamiltonian | flow character | you choose |
|---|---|---|---|
| `kick(a)` | `H = a·L` | **reorients** — rigid rotation of `L` about the axis `a`, rate `|a|` | axis direction + amplitude |
| `twist(a)` | `H = ½(a·L)²` | **shears** — latitudes about `a` rotate at rate ∝ `(a·L)`: layered, non-rigid | axis direction + amplitude |
| `quad(A)` | `H = ½LᵀAL` | **reshapes** — the rotation axis is the state-dependent `A·L`: a modified-*inertia* event | a symmetric 3×3 matrix + amplitude |

All three are implemented (`lie_poisson.kick`/`twist`/`quad`). The family nests: `twist(a)` is
the rank-1 special case `quad(a·aᵀ)`, and the identity's own free dynamics is itself
`quad(diag(1, ½, ⅓))` — so a quad event literally perturbs the automaton's *sense of inertia*.
Only the symmetric part of `A` enters `H` (the constructor symmetrizes, so a slightly
asymmetric authored matrix is handled consistently). Anything smooth beyond these is also
legal via a custom `grad_h` callable — still a Hamiltonian, just not from a constructor.

## The row format

One symbol = one row:

```
name · kind ∈ {kick, twist} · axis (3 numbers, any direction) · amplitude · duration (optional)
name · quad               · matrix A (symmetric 3×3 — six numbers: A₁₁ A₂₂ A₃₃ A₁₂ A₁₃ A₂₃) · amplitude · duration (optional)
```

(For quad rows I wire `quad(amplitude · A)` — same one-source-of-truth convention as folding
amplitude into a kick/twist axis.)

Worked examples (the current default three, plus illustrative extensions):

| name | kind | axis | amplitude | duration | reading |
|---|---|---|---|---|---|
| `x` | kick | (1, 0, 0) | 0.5 | — | reorient about e₁ |
| `y` | kick | (0, 1, 0) | 0.5 | — | reorient about e₂ (does not commute with `x`) |
| `s` | twist | (1, 0, 0) | 1.0 | — | shear about e₁ |
| `z` | kick | (0, 0, 1) | 0.5 | — | third reorientation axis |
| `X` | kick | (1, 0, 0) | 1.5 | — | a *hard* x-kick (intensity via amplitude) |
| `x̄` | kick | (−1, 0, 0) | 0.5 | — | the **inverse** of `x` — negated axis undoes it |
| `d` | kick | (1, 1, 0)/√2 | 0.5 | — | a diagonal-axis reorientation |
| `t` | twist | (0, 1, 0) | 1.0 | 1.0 | a *long* shear (duration override) |
| `w` | quad | diag(0.4, 0.9, 1.5) | 1.0 | — | reshape: the world's inertia felt differently along each axis |
| `c` | quad | A₁₂ = A₂₁ = 1 (else 0), i.e. `H = L₁L₂` | 1.0 | — | cross-coupling: axes 1 and 2 shear *into each other* |
| `g` | quad | diag(1, ½, ⅓) | −0.5 | — | a *drag* on the identity's own inertia (negative multiple of `H₀`'s matrix — partially "suspends the self") |

Semantic freedom: **axis = where in `L`-space the event acts · kind = its character (reorient /
shear / reshape) · amplitude = intensity · duration = dwell.** Negated axes give you inverse
pairs for free (`kick(−a)` exactly undoes `kick(a)` in isolation) — a useful semantic device if
some of your 22 come in do/undo pairs.

## Design guidance (what makes an alphabet *good*)

- **Spread the axes.** Kicks about the *same* axis commute with each other — words made only of
  parallel kicks collapse toward bags of symbols. Non-commutativity (⇒ word order matters, ⇒ ζ
  and any future language work have grip) comes from axis diversity. The default pair {x, y}
  is the minimum; 22 symbols across varied axes is far richer.
- **Keep amplitudes order-unity.** The pinned scales: sphere `R = 1`, `H₀ ∈ [1/6, 1/2]`
  (inertias 1, 2, 3), defaults `kick 0.5`, `twist 1.0`, `τ = 0.5` (50 steps at dt 0.01).
  Amplitudes ≪ 0.1 make events nearly invisible in the energy; there is no upper danger to ψ
  (it is exact regardless), only integrator-resolution common sense (`|∇H|·dt` comfortably
  below ~1).
- **Avoid near-duplicates.** Two symbols with nearly the same kind+axis+amplitude are
  dynamically almost indistinguishable — words differing only in them will barely separate in
  state or ζ. The validation battery (below) measures this pairwise, so near-duplicates are
  caught, not guessed.
- **Mix kinds.** Twists (non-isometric) enrich the dynamics in a way kicks alone cannot;
  quads add the third character (state-dependent rotation axes — reshaped inertia).
- **The silent-quad trap (the isotropy lesson, third appearance).** An isotropic
  `quad(c·I)` does *exactly nothing*: `H = c|L|²/2` is a function of ψ itself, so the flow
  freezes for the event's whole duration — a legal but dead symbol. The same lesson that
  collapsed the Gaussian charge (§9.15) and that made anisotropic inertia a theorem-level
  precondition (§9.16) applies to authored matrices: keep `A`'s **eigenvalues distinct** (or
  at least not all equal). Near-isotropic `A` = near-silent symbol; the battery's
  energy-visibility check flags both. Eigenvalue scale: order-unity, like the other
  amplitudes.
- **No zero axes / zero amplitudes** (a symbol that does nothing is legal but pointless — and
  the battery will flag it as a duplicate of silence).

## What we run when your 22 land (the validation battery)

1. **ψ sweep** — the theorem check across random words of *your* alphabet (expected trivially
   green at ~1e-14; recorded anyway).
2. **Energy visibility per symbol** — each symbol's median `|ΔH₀|` per event (flags invisible
   symbols).
3. **Pairwise distinguishability** — a 22×22 matrix: same genesis, one event of σᵢ vs σⱼ,
   distance in endpoint state and in ζ (flags near-duplicates).
4. **Order sensitivity** — random word vs its shuffle: state and ζ separation (the
   non-commutativity health of the whole alphabet).
5. **† control** — unchanged; the boundary is alphabet-independent.

Results get a short recorded addendum in the spec; misses are findings about *content*, freely
iterable (edit rows, re-run in seconds) — the bars on the *mechanism* never move.

## Handing it over

Any form works: a markdown table like the above, JSON, or a Python list. On arrival I wire it
as `make_authored_alphabet()` (adding a JSON loader if your format calls for it — `kick`,
`twist`, and `quad` are all implemented and tested), run the battery, and record. Symbols are
data; iteration is cheap.
