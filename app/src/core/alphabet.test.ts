import { describe, it, expect } from 'vitest'
import defaultJson from '../../public/alphabet.json'
import type { Coeffs } from './lie.ts'
import type { Diagnostic, Level } from './alphabet.ts'
import { buildSymbol, SIGMA0_KEYS, symMatrix, ZERO } from './lie.ts'
import {
  BLEND,
  buildTimeline,
  coeffsFromAA,
  formatCoeffs,
  intensity,
  parseAlphabet,
  scanWord,
  totalSteps,
  wordOf,
} from './alphabet.ts'

const levels = (d: readonly Diagnostic[], l: Level) => d.filter((x) => x.level === l)
const parseDefault = () => parseAlphabet(defaultJson, 'public/alphabet.json')

// A minimal valid file to mutate per-case.
const file = (symbols: unknown[], extra: Record<string, unknown> = {}) => ({
  schema: 'embraos.alphabet/1',
  symbols,
  ...extra,
})

describe('coeffsFromAA — §3’s alternate encoding, and battery item 9’s readout', () => {
  it('reproduces §9’s `s` row from v1’s twist(e₁) = quad(diag(1,0,0))', () => {
    // The doc's number, derived rather than asserted: the traceless projection of
    // diag(1,0,0) is d₁ = 7/√78, d₂ = 1/√26, with an isotropic residue of 1/3.
    const { coeffs, residue } = coeffsFromAA(
      [0, 0, 0],
      [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
    )
    expect(coeffs.d1).toBeCloseTo(7 / Math.sqrt(78), 12)
    expect(coeffs.d2).toBeCloseTo(1 / Math.sqrt(26), 12)
    expect(coeffs.d1).toBeCloseTo(0.793, 3) // §9's tabulated value
    expect(coeffs.d2).toBeCloseTo(0.196, 3) // §9's tabulated value
    expect(residue).toBeCloseTo(1 / 3, 12)
  })

  it('reproduces §9’s `g` row from v1’s quad(diag(1, ½, ⅓)) at amp −0.5', () => {
    // §3 claims this trims to PURE d₁. It does — d₂ is exactly zero, because
    // diag(1/I) traceless IS the D1 direction.
    const { coeffs, residue } = coeffsFromAA(
      [0, 0, 0],
      [
        [1, 0, 0],
        [0, 1 / 2, 0],
        [0, 0, 1 / 3],
      ],
    )
    expect(coeffs.d2).toBeCloseTo(0, 12)
    expect(coeffs.d1).toBeCloseTo(Math.sqrt(78) / 18, 12)
    expect(residue).toBeCloseTo(11 / 18, 12) // the doc's "trace 11/6", over 3
    expect(-0.5 * coeffs.d1).toBeCloseTo(-0.245, 3) // §9's tabulated amp
  })

  it('c_ij = A_ij exactly — the two conventions already agree', () => {
    const { coeffs } = coeffsFromAA(
      [0, 0, 0],
      [
        [0, 0.7, -0.2],
        [0.7, 0, 0.4],
        [-0.2, 0.4, 0],
      ],
    )
    expect(coeffs.c12).toBeCloseTo(0.7, 12)
    expect(coeffs.c13).toBeCloseTo(-0.2, 12)
    expect(coeffs.c23).toBeCloseTo(0.4, 12)
  })

  it('round-trips Σ₀ → (a, A) → Σ₀ for every row of the shipped default', () => {
    for (const row of parseDefault().alphabet.symbols) {
      const { a, A } = symMatrix(row.coeffs)
      const back = coeffsFromAA(a, A)
      expect(back.residue).toBeCloseTo(0, 12) // Σ₀ points are traceless by construction
      for (const k of SIGMA0_KEYS) expect(back.coeffs[k]).toBeCloseTo(row.coeffs[k], 12)
    }
  })

  it('accepts the 5-number traceless form and the 6-number form identically', () => {
    const five = parseAlphabet(file([{ name: 'a', A: [0.3, -0.1, 0.5, 0, 0], amp: 1 }]), 't')
    const six = parseAlphabet(file([{ name: 'a', A: [0.3, -0.1, -0.2, 0.5, 0, 0], amp: 1 }]), 't')
    expect(five.fatal).toBe(false)
    expect(six.fatal).toBe(false)
    for (const k of SIGMA0_KEYS) {
      expect(six.alphabet.symbols[0].coeffs[k]).toBeCloseTo(five.alphabet.symbols[0].coeffs[k], 12)
    }
  })
})

describe('the shipped default (docs §9) — this file IS the fixture', () => {
  const res = parseDefault()

  it('parses with no errors', () => {
    expect(levels(res.diagnostics, 'error')).toEqual([])
    expect(res.fatal).toBe(false)
  })

  it('yields exactly the eight §9 symbols, in order', () => {
    expect(res.alphabet.symbols.map((s) => s.name)).toEqual([
      'x', 'X', 's', 'g', 'm', 'p', 'n', 't',
    ])
  })

  it('carries exactly one warning: the §4/A4 duration scope note on `t`', () => {
    // Pre-registered: the default is meant to exercise every feature, and `dur`
    // legitimately warns. Anything ELSE warning is a regression.
    const warns = levels(res.diagnostics, 'warn')
    expect(warns).toHaveLength(1)
    expect(warns[0].where).toContain('"t"')
    expect(warns[0].message).toContain('dur')
  })

  it('reads x and X as distinct symbols at 3× intensity — case is significant', () => {
    const { byName } = res.alphabet
    expect(byName.get('x')!.amp).toBe(0.5)
    expect(byName.get('X')!.amp).toBe(1.5)
    expect(intensity(byName.get('X')!) / intensity(byName.get('x')!)).toBeCloseTo(3, 12)
  })

  it('notes x/X as a deliberate intensity variant, but not p/n (a registered pair)', () => {
    const infos = levels(res.diagnostics, 'info')
    expect(infos.some((d) => d.where.includes('"x"') && d.where.includes('"X"'))).toBe(true)
    expect(infos.some((d) => d.where.includes('"p"') && d.where.includes('"n"'))).toBe(false)
  })

  it('registers the p/n pair', () => {
    expect(res.alphabet.pairs).toEqual([
      { do: 'p', undo: 'n', note: expect.stringContaining('not inverses') },
    ])
  })

  it('agrees with §9’s reading of `s` — non-unit norm is the dropped residue, not an error', () => {
    // ‖s‖ = √6/3: exactly what survives after quad(diag(1,0,0)) loses its
    // isotropic third. amp alone is therefore NOT intensity.
    const s = res.alphabet.byName.get('s')!
    expect(intensity(s)).toBeCloseTo(Math.sqrt(6) / 3, 3)
    expect(s.amp).toBe(1)
  })

  it('every row is audible', () => {
    for (const r of res.alphabet.symbols) expect(r.span).toBeGreaterThan(0.02)
  })
})

describe('validation — hand-edited JSON must fail readably, never blank the page', () => {
  it('rejects a non-object top level', () => {
    expect(parseAlphabet([1, 2, 3], 't').fatal).toBe(true)
    expect(parseAlphabet('nope', 't').fatal).toBe(true)
  })

  it('rejects a missing symbols array', () => {
    expect(parseAlphabet({ schema: 'embraos.alphabet/1' }, 't').fatal).toBe(true)
  })

  it('drops a bad row but keeps the rest', () => {
    const res = parseAlphabet(
      file([
        { name: 'xx', sigma0: { k1: 1 } }, // two codepoints — §4/A7
        { name: 'a', sigma0: { k1: 1 } },
      ]),
      't',
    )
    expect(res.fatal).toBe(false)
    expect(res.alphabet.symbols.map((s) => s.name)).toEqual(['a'])
    expect(levels(res.diagnostics, 'error')[0].message).toContain('single ASCII letter')
  })

  it('rejects duplicates case-sensitively, but not across case', () => {
    const dup = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1 } }, { name: 'a', sigma0: { k2: 1 } }]), 't')
    expect(dup.alphabet.symbols).toHaveLength(1)
    expect(levels(dup.diagnostics, 'error')[0].message).toContain('duplicate')

    const cased = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1 } }, { name: 'A', sigma0: { k2: 1 } }]), 't')
    expect(cased.alphabet.symbols).toHaveLength(2)
    expect(levels(cased.diagnostics, 'error')).toEqual([])
  })

  it('rejects both encodings at once, and neither', () => {
    const both = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1 }, a: [1, 0, 0] }]), 't')
    expect(both.fatal).toBe(true)
    expect(levels(both.diagnostics, 'error')[0].message).toContain('both encodings')

    const none = parseAlphabet(file([{ name: 'a', amp: 1 }]), 't')
    expect(levels(none.diagnostics, 'error')[0].message).toContain('neither encoding')
  })

  it('rejects an unknown Σ₀ key rather than silently zeroing it', () => {
    const res = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1, c33: 0.5 } }]), 't')
    expect(res.fatal).toBe(true)
    expect(levels(res.diagnostics, 'error')[0].message).toContain('c33')
  })

  it('warns on a misspelled row key instead of defaulting behind your back', () => {
    const res = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1 }, amplitude: 2 }]), 't')
    expect(levels(res.diagnostics, 'warn').some((d) => d.message.includes('amplitude'))).toBe(true)
    expect(res.alphabet.symbols[0].amp).toBe(1) // and it really did default
  })

  it('rejects a _comment divider that also carries real content', () => {
    // The house rule from sandbox/latent.py: a typo'd row must not vanish silently.
    const res = parseAlphabet(file([{ _comment: 'group', name: 'a', sigma0: { k1: 1 } }]), 't')
    expect(res.fatal).toBe(true)
    expect(levels(res.diagnostics, 'error')[0].message).toContain('only that key')
  })

  it('skips a pure _comment divider silently', () => {
    const res = parseAlphabet(file([{ _comment: 'group' }, { name: 'a', sigma0: { k1: 1 } }]), 't')
    expect(res.alphabet.symbols).toHaveLength(1)
    expect(levels(res.diagnostics, 'error')).toEqual([])
    expect(levels(res.diagnostics, 'warn')).toEqual([])
  })

  it('flags a silent row before you ever play it', () => {
    const res = parseAlphabet(file([{ name: 'a', sigma0: {}, amp: 1 }]), 't')
    expect(levels(res.diagnostics, 'warn')[0].message).toContain('SILENT')
  })

  it('flags a faint row against H₀’s span', () => {
    const res = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1 }, amp: 0.001 }]), 't')
    expect(levels(res.diagnostics, 'warn')[0].message).toContain('faint')
  })

  it('flags a coarse integrator step — the honest form of the ceiling question', () => {
    const res = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1 }, amp: 40 }]), 't')
    expect(levels(res.diagnostics, 'warn').some((d) => d.message.includes('coarse step'))).toBe(true)
  })

  it('flags genuine near-duplicates', () => {
    const res = parseAlphabet(
      file([
        { name: 'a', sigma0: { k1: 1 }, amp: 1 },
        { name: 'b', sigma0: { k1: 1.001 }, amp: 1 },
      ]),
      't',
    )
    expect(levels(res.diagnostics, 'warn').some((d) => d.message.includes('near-duplicate'))).toBe(true)
  })

  it('rejects a non-positive dur', () => {
    const res = parseAlphabet(file([{ name: 'a', sigma0: { k1: 1 }, dur: 0 }]), 't')
    expect(res.fatal).toBe(true)
    expect(levels(res.diagnostics, 'error')[0].message).toContain('dur')
  })

  it('honors a permuted basis for array-form rows, with a warning', () => {
    const res = parseAlphabet(
      file([{ name: 'a', sigma0: [1, 0, 0, 0, 0, 0, 0, 0], amp: 1 }], {
        basis: ['d2', 'k2', 'k3', 'c12', 'c13', 'c23', 'd1', 'k1'],
      }),
      't',
    )
    expect(res.alphabet.symbols[0].coeffs.d2).toBe(1)
    expect(res.alphabet.symbols[0].coeffs.k1).toBe(0)
    expect(levels(res.diagnostics, 'warn').some((d) => d.message.includes('permutation'))).toBe(true)
  })

  it('is fatal when nothing survives', () => {
    expect(parseAlphabet(file([{ name: '9', sigma0: { k1: 1 } }]), 't').fatal).toBe(true)
  })
})

describe('words — case-preserving, unknown characters surfaced not swallowed', () => {
  const known = new Set(['x', 'X', 's', 'g'])

  it('keeps case: x and X are different symbols', () => {
    expect(scanWord('xX', known).every((t) => t.ok)).toBe(true)
    expect(wordOf(scanWord('xX', known))).toEqual(['x', 'X'])
  })

  it('marks unknown characters instead of deleting them', () => {
    const scan = scanWord('xQs', known)
    expect(scan.map((t) => t.ok)).toEqual([true, false, true])
    expect(wordOf(scan)).toEqual(['x', 's'])
  })

  it('accepts the blend pseudo-symbol, which no authored name can collide with', () => {
    expect(scanWord(BLEND, known)[0].ok).toBe(true)
    expect(BLEND).not.toMatch(/[A-Za-z]/)
  })

  it('is codepoint-safe', () => {
    expect(scanWord('x🙂', known).map((t) => t.ok)).toEqual([true, false])
  })
})

describe('timelines — per-symbol dwell', () => {
  const { alphabet } = parseDefault()
  const live = () => ({ coeffs: { ...ZERO, k1: 1 } as Coeffs, amp: 1 })

  it('is gap–event–gap per symbol', () => {
    const tl = buildTimeline(['x', 's'], alphabet.byName, live)
    expect(tl.map((s) => s.phase)).toEqual(['gap', 'event', 'gap', 'event', 'gap'])
    expect(tl.filter((s) => s.phase === 'event').map((s) => s.name)).toEqual(['x', 's'])
  })

  it('gives `t` twice the dwell of a default-schedule symbol (dur 1.0 vs τ 0.5)', () => {
    const tl = buildTimeline(['x', 't'], alphabet.byName, live)
    const ev = tl.filter((s) => s.phase === 'event')
    expect(ev[1].steps).toBe(2 * ev[0].steps)
  })

  it('resolves the blend against the live sliders at build time', () => {
    const tl = buildTimeline([BLEND], alphabet.byName, live)
    const ev = tl.find((s) => s.phase === 'event')!
    expect(ev.name).toBe(BLEND)
    expect(ev.sym!.h([1, 0, 0])).toBeCloseTo(1, 12)
  })

  it('totalSteps accounts for dur — this is what the trail buffer must be sized against', () => {
    const short = totalSteps(buildTimeline(['x'], alphabet.byName, live))
    const long = totalSteps(buildTimeline(['t'], alphabet.byName, live))
    expect(long).toBe(short + 50)
  })
})

describe('display helpers', () => {
  it('formats a blend in canonical Σ₀ order', () => {
    expect(formatCoeffs({ ...ZERO, k1: 0.6, c23: 0.4 })).toBe('0.60·k₁ + 0.40·c₂₃')
    expect(formatCoeffs({ ...ZERO, d1: -0.5 })).toBe('− 0.50·d₁')
    expect(formatCoeffs(ZERO)).toBe('0 (no Hσ)')
    // Display rounding is toFixed(2), inherited from the prototype: it is a
    // 2-dp READOUT, not the stored value. 0.245 sits just below the tie in
    // binary float, so it shows as 0.24 while the coefficient stays exact.
    expect(formatCoeffs({ ...ZERO, d1: -0.245 })).toBe('− 0.24·d₁')
  })

  it('agrees with buildSymbol about what is silent', () => {
    expect(formatCoeffs(ZERO)).toContain('no Hσ')
    expect(buildSymbol(ZERO, 1).h([1, 0, 0])).toBe(0)
  })
})
