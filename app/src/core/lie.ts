/**
 * The so(3)* Lie–Poisson core — a port of sandbox/lie_poisson.py.
 *
 * State L lives on the sphere ψ = |L|² = 1. Flow: L̇ = L × ∇H.
 * Integrator: RKMK2, one rotation per step ⇒ ψ is exact and never renormalized.
 * That is the mechanism claim the visualizer's drift counter reports live.
 *
 * This module also owns Σ₀ — the eight-direction perturbation basis. Σ₀ is the
 * COORDINATE SYSTEM, pinned by the mathematics (docs/ALPHABET-AUTHORING.md §2:
 * the perturbation space has rank 8, with the Casimir quad(I) as the sole kernel
 * direction). It is not authored content and never comes from a file. The
 * alphabet — named symbols, each a point in this 8-space — is authored content
 * and lives in alphabet.ts. Conflating the two is what the prototype did.
 */

export type Vec3 = [number, number, number]

// ---------------------------------------------------------------------------
// vectors
// ---------------------------------------------------------------------------
export const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
export const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
export const add3 = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
export const scale3 = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s]
export const norm3 = (a: Vec3) => Math.sqrt(dot(a, a))

// ---------------------------------------------------------------------------
// the pinned scales — recorded results are measured against these, so they are
// constants here rather than anything an alphabet file can set.
// ---------------------------------------------------------------------------
export const INERTIA: Vec3 = [1, 2, 3]
export const DT = 0.01
export const TAU_EVENT = 0.5
export const TAU_GAP = 0.5

/** The free law H₀ = ½ Σ Lᵢ²/Iᵢ. On ψ = 1 it runs over [1/6, 1/2]. */
export const gradH0 = (L: Vec3): Vec3 => [L[0] / INERTIA[0], L[1] / INERTIA[1], L[2] / INERTIA[2]]
export const h0 = (L: Vec3) =>
  0.5 * ((L[0] * L[0]) / INERTIA[0] + (L[1] * L[1]) / INERTIA[1] + (L[2] * L[2]) / INERTIA[2])

/** max − min of H₀ over ψ = 1: 1/2 − 1/6. The yardstick every H_σ span is read against. */
export const H0_SPAN = 1 / 2 - 1 / 6

// ---------------------------------------------------------------------------
// the group step
// ---------------------------------------------------------------------------

/**
 * Rodrigues rotation of x by rotation-vector v (sinc form, small-angle guard).
 * Exactly norm-preserving — this is the mechanism that makes ψ structural rather
 * than checked.
 */
export function rotate(v: Vec3, x: Vec3): Vec3 {
  const th = norm3(v)
  let sa: number, sb: number
  if (th < 1e-8) {
    sa = 1 - (th * th) / 6
    sb = 0.5 - (th * th) / 24
  } else {
    sa = Math.sin(th) / th
    sb = (1 - Math.cos(th)) / (th * th)
  }
  const c = Math.cos(th)
  const cv = cross(v, x)
  const vd = dot(v, x)
  return [
    x[0] * c + cv[0] * sa + v[0] * vd * sb,
    x[1] * c + cv[1] * sa + v[1] * vd * sb,
    x[2] * c + cv[2] * sa + v[2] * vd * sb,
  ]
}

/** One explicit-midpoint step on the rotation group. L̇ = L × ω = −ω × L, so the
 *  rotation vector is −ω·dt. */
export function rkmk2(gradFn: (L: Vec3) => Vec3, L: Vec3, dt: number): Vec3 {
  const Lh = rotate(scale3(gradFn(L), -0.5 * dt), L)
  return rotate(scale3(gradFn(Lh), -dt), L)
}

// ---------------------------------------------------------------------------
// Σ₀ — the pinned base (docs/ALPHABET-AUTHORING.md §2)
// ---------------------------------------------------------------------------

const SQ78 = Math.sqrt(78)
const SQ26 = Math.sqrt(26)

/** The traceless part of diag(1/I₁, 1/I₂, 1/I₃), normalized: the identity's own
 *  law direction, i.e. the canonical self-modulation symbol. For I = (1,2,3),
 *  diag(1, ½, ⅓) − (11/18)·I = (1/18)·diag(7, −2, −5). */
export const D1: Vec3 = [7 / SQ78, -2 / SQ78, -5 / SQ78]

/** The independent traceless diagonal, Frobenius-orthogonal to D1. */
export const D2: Vec3 = [1 / SQ26, -4 / SQ26, 3 / SQ26]

export const SIGMA0 = [
  { key: 'k1', label: 'k₁', group: 'ℓ=1 · reorient' },
  { key: 'k2', label: 'k₂', group: 'ℓ=1 · reorient' },
  { key: 'k3', label: 'k₃', group: 'ℓ=1 · reorient' },
  { key: 'c12', label: 'c₁₂', group: 'ℓ=2 · cross-couple' },
  { key: 'c13', label: 'c₁₃', group: 'ℓ=2 · cross-couple' },
  { key: 'c23', label: 'c₂₃', group: 'ℓ=2 · cross-couple' },
  { key: 'd1', label: 'd₁', group: 'ℓ=2 · reshape (d₁ = the self direction)' },
  { key: 'd2', label: 'd₂', group: 'ℓ=2 · reshape (d₁ = the self direction)' },
] as const

export type Sigma0Key = (typeof SIGMA0)[number]['key']
export type Coeffs = Record<Sigma0Key, number>

export const SIGMA0_KEYS: readonly Sigma0Key[] = SIGMA0.map((b) => b.key)
export const SIGMA0_LABEL: Readonly<Record<Sigma0Key, string>> = Object.fromEntries(
  SIGMA0.map((b) => [b.key, b.label]),
) as Record<Sigma0Key, string>

/** Σ₀'s ℓ-groups in display order, each with its members. */
export const SIGMA0_GROUPS: { group: string; keys: Sigma0Key[] }[] = SIGMA0.reduce(
  (acc, b) => {
    const last = acc[acc.length - 1]
    if (last && last.group === b.group) last.keys.push(b.key)
    else acc.push({ group: b.group, keys: [b.key] })
    return acc
  },
  [] as { group: string; keys: Sigma0Key[] }[],
)

export const ZERO: Coeffs = Object.fromEntries(SIGMA0_KEYS.map((k) => [k, 0])) as Coeffs
export const unit = (k: Sigma0Key): Coeffs => ({ ...ZERO, [k]: 1 })

/** Euclidean norm of the 8-vector in Σ₀ coordinates — the coordinates battery
 *  item 9 reports in. Note Σ₀ is not Frobenius-orthonormal as matrices (the c's
 *  have norm √2), so this is a coordinate norm, not an operator norm. */
export const coeffsNorm = (c: Coeffs) =>
  Math.sqrt(SIGMA0_KEYS.reduce((s, k) => s + c[k] * c[k], 0))

export const coeffsFinite = (c: Coeffs) => SIGMA0_KEYS.every((k) => Number.isFinite(c[k]))

/** The (a, A) pair a Σ₀ point denotes. A is symmetric and traceless by construction. */
export function symMatrix(c: Coeffs): { a: Vec3; A: number[][] } {
  const A = [
    [c.d1 * D1[0] + c.d2 * D2[0], c.c12, c.c13],
    [c.c12, c.d1 * D1[1] + c.d2 * D2[1], c.c23],
    [c.c13, c.c23, c.d1 * D1[2] + c.d2 * D2[2]],
  ]
  return { a: [c.k1, c.k2, c.k3], A }
}

// ---------------------------------------------------------------------------
// symbols
// ---------------------------------------------------------------------------

export interface Generator {
  grad: (L: Vec3) => Vec3
  h: (L: Vec3) => number
}

/** H_σ = amp · ( a·L + ½ LᵀAL ) — the one legality rule (§1): every symbol IS a
 *  Hamiltonian. Both the live slider blend and a loaded row come through here. */
export function buildSymbol(c: Coeffs, amp = 1): Generator {
  const { a, A } = symMatrix(c)
  const m = amp
  return {
    grad: (L) => [
      m * (a[0] + A[0][0] * L[0] + A[0][1] * L[1] + A[0][2] * L[2]),
      m * (a[1] + A[1][0] * L[0] + A[1][1] * L[1] + A[1][2] * L[2]),
      m * (a[2] + A[2][0] * L[0] + A[2][1] * L[1] + A[2][2] * L[2]),
    ],
    h: (L) => {
      const AL: Vec3 = [
        A[0][0] * L[0] + A[0][1] * L[1] + A[0][2] * L[2],
        A[1][0] * L[0] + A[1][1] * L[1] + A[1][2] * L[2],
        A[2][0] * L[0] + A[2][1] * L[1] + A[2][2] * L[2],
      ]
      return m * (dot(a, L) + 0.5 * dot(L, AL))
    },
  }
}

/** Fibonacci sphere sampler — the flow-arrow field and the readouts below. */
export function fibSphere(n: number): Vec3[] {
  const pts: Vec3[] = []
  const ga = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * i + 1) / n
    const r = Math.sqrt(Math.max(0, 1 - y * y))
    const th = ga * i
    pts.push([Math.cos(th) * r, y, Math.sin(th) * r])
  }
  return pts
}

const PROBE = fibSphere(1024)

/**
 * Two numbers that decide whether an authored symbol is worth having, both read
 * off the same sphere sample:
 *
 *  - `span`: max − min of H_σ over ψ = 1. This is "is it visible in the energy"
 *    (§5), directly comparable to H0_SPAN = 1/3. span = 0 ⟺ the symbol is
 *    SILENT: ∇H ≡ 0 and H is a function of the Casimir, so it does nothing.
 *  - `maxGrad`: max |∇H_σ|. Times dt this is the per-step rotation angle, which
 *    is the honest form of the bookkeeping-ceiling question (item 7) — a coarse
 *    step, not a large energy, is what degrades the integrator.
 */
export function probe(gen: Generator): { span: number; maxGrad: number } {
  let lo = Infinity
  let hi = -Infinity
  let maxGrad = 0
  for (const L of PROBE) {
    const v = gen.h(L)
    if (v < lo) lo = v
    if (v > hi) hi = v
    const g = norm3(gen.grad(L))
    if (g > maxGrad) maxGrad = g
  }
  return { span: hi - lo, maxGrad }
}
