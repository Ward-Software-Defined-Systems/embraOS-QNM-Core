import { describe, it, expect } from 'vitest'
import type { Vec3 } from './lie.ts'
import {
  buildSymbol,
  D1,
  D2,
  dot,
  gradH0,
  h0,
  H0_SPAN,
  INERTIA,
  norm3,
  probe,
  rkmk2,
  SIGMA0,
  SIGMA0_KEYS,
  unit,
  ZERO,
} from './lie.ts'

const frob = (v: Vec3) => Math.sqrt(dot(v, v))

describe('Σ₀ — the pinned base (docs/ALPHABET-AUTHORING.md §2)', () => {
  it('D1 and D2 are unit Frobenius and mutually orthogonal', () => {
    // §2 pins both "so coordinate readouts are reproducible" — this is that claim.
    expect(frob(D1)).toBeCloseTo(1, 12)
    expect(frob(D2)).toBeCloseTo(1, 12)
    expect(dot(D1, D2)).toBeCloseTo(0, 12)
  })

  it('both are traceless', () => {
    expect(D1[0] + D1[1] + D1[2]).toBeCloseTo(0, 12)
    expect(D2[0] + D2[1] + D2[2]).toBeCloseTo(0, 12)
  })

  it('D1 IS the normalized traceless part of diag(1/I) — the identity’s own law direction', () => {
    const inv: Vec3 = [1 / INERTIA[0], 1 / INERTIA[1], 1 / INERTIA[2]]
    const tr = (inv[0] + inv[1] + inv[2]) / 3
    const tl: Vec3 = [inv[0] - tr, inv[1] - tr, inv[2] - tr]
    const n = frob(tl)
    expect(tl[0] / n).toBeCloseTo(D1[0], 12)
    expect(tl[1] / n).toBeCloseTo(D1[1], 12)
    expect(tl[2] / n).toBeCloseTo(D1[2], 12)
  })

  it('has 8 directions in three ℓ-groups, keys unique', () => {
    expect(SIGMA0).toHaveLength(8)
    expect(new Set(SIGMA0_KEYS).size).toBe(8)
    expect(new Set(SIGMA0.map((b) => b.group)).size).toBe(3)
  })
})

describe('H₀ — the free law', () => {
  it('spans [1/6, 1/2] on ψ = 1', () => {
    const { span } = probe({ grad: gradH0, h: h0 })
    expect(span).toBeCloseTo(H0_SPAN, 3)
    expect(h0([1, 0, 0])).toBeCloseTo(1 / 2, 12)
    expect(h0([0, 0, 1])).toBeCloseTo(1 / 6, 12)
  })
})

describe('RKMK2 — ψ is structural, not checked', () => {
  it('holds |L|² exact over 2000 steps under a nontrivial generator', () => {
    // One rotation per step is exactly norm-preserving, so drift can only be
    // floating-point noise — never a slow leak. This is the mechanism claim.
    const sym = buildSymbol({ ...ZERO, k1: 0.6, c23: 0.4, d1: -0.3 }, 1.2)
    const grad = (L: Vec3): Vec3 => {
      const g0 = gradH0(L)
      const gs = sym.grad(L)
      return [g0[0] + gs[0], g0[1] + gs[1], g0[2] + gs[2]]
    }
    let L: Vec3 = [0.3, -0.5, 0.81]
    L = [L[0] / norm3(L), L[1] / norm3(L), L[2] / norm3(L)]
    let maxDrift = 0
    for (let i = 0; i < 2000; i++) {
      L = rkmk2(grad, L, 0.01)
      maxDrift = Math.max(maxDrift, Math.abs(dot(L, L) - 1))
    }
    expect(maxDrift).toBeLessThan(1e-12)
  })

  it('is never renormalized — a norm-breaking field would show up', () => {
    // Sanity on the test itself: a radial (non-tangent) field is not what rkmk2
    // integrates, and rotation still preserves the norm. If this ever failed,
    // `rotate` would be the culprit, not the physics.
    let L: Vec3 = [1, 0, 0]
    for (let i = 0; i < 500; i++) L = rkmk2((x) => [x[0] * 3, x[1] - 2, x[2] + 1], L, 0.02)
    expect(Math.abs(dot(L, L) - 1)).toBeLessThan(1e-12)
  })
})

describe('buildSymbol — every symbol IS a Hamiltonian (§1)', () => {
  it('grad is the actual gradient of h', () => {
    const sym = buildSymbol({ ...ZERO, k1: 0.7, k3: -0.2, c12: 0.5, c13: 0.1, d1: 0.9, d2: -0.4 }, 1.3)
    const L: Vec3 = [0.4, -0.6, 0.69]
    const eps = 1e-6
    const g = sym.grad(L)
    for (let i = 0; i < 3; i++) {
      const p: Vec3 = [...L]
      const m: Vec3 = [...L]
      p[i] += eps
      m[i] -= eps
      expect((sym.h(p) - sym.h(m)) / (2 * eps)).toBeCloseTo(g[i], 7)
    }
  })

  it('is linear in amplitude, for every direction (§4/A3)', () => {
    for (const k of SIGMA0_KEYS) {
      const L: Vec3 = [0.5, 0.5, Math.SQRT1_2]
      const one = buildSymbol(unit(k), 1).h(L)
      const two = buildSymbol(unit(k), 2.5).h(L)
      expect(two).toBeCloseTo(2.5 * one, 12)
    }
  })

  it('flags the isotropy trap: a zero point is SILENT', () => {
    const { span, maxGrad } = probe(buildSymbol(ZERO, 1))
    expect(span).toBe(0)
    expect(maxGrad).toBe(0)
  })

  it('each Σ₀ direction is audible on its own (rank 8, no silent basis vector)', () => {
    for (const k of SIGMA0_KEYS) {
      expect(probe(buildSymbol(unit(k), 1)).span).toBeGreaterThan(0.1)
    }
  })
})
