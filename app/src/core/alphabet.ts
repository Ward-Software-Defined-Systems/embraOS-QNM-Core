/**
 * The authored input alphabet Σ — schema, loader, and the word/timeline machinery.
 *
 * Contract (docs/ALPHABET-AUTHORING.md §1): every symbol IS a Hamiltonian
 * H_σ = amp · ( a·L + ½ LᵀAL ). A row is a point in the 8-space Σ₀, plus
 * intensity and dwell. §3 specifies two equivalent encodings and says the loader
 * accepts both; both land here and normalize to Σ₀ coordinates, which is what
 * battery item 9 reports in.
 *
 * parseAlphabet is a TOTAL function: it never throws. Hand-edited JSON is the
 * input, so every malformed shape has to come back as a diagnostic the author can
 * read, not an exception that blanks the page.
 */

import type { Coeffs, Generator, Sigma0Key, Vec3 } from './lie.ts'
import {
  buildSymbol,
  coeffsNorm,
  D1,
  D2,
  DT,
  probe,
  SIGMA0_KEYS,
  SIGMA0_LABEL,
  TAU_EVENT,
  TAU_GAP,
  ZERO,
} from './lie.ts'

export const SCHEMA_ID = 'embraos.alphabet/1'
export const SUBSTRATE = 'so(3)*'

/** §4/A7: names are single ASCII letters, and case is significant — upper/lower
 *  is how the doc distinguishes intensity variants. */
export const NAME_RE = /^[A-Za-z]$/

// ---------------------------------------------------------------------------
// types
// ---------------------------------------------------------------------------

export type Level = 'error' | 'warn' | 'info'

export interface Diagnostic {
  level: Level
  where: string
  message: string
}

export interface SymbolRow {
  name: string
  coeffs: Coeffs
  amp: number
  dur?: number
  reading?: string
  /** The isotropic part split off when a row arrived as (a, A) with tr A ≠ 0.
   *  H_res = residue · ½|L|² is a function of ψ, so it is SILENT — invisible to
   *  the flow. Reported rather than hidden; §3's whole `g` discussion is about it. */
  residue: number
  /** max − min of H_σ over ψ = 1, and max |∇H_σ|. See lie.probe. */
  span: number
  maxGrad: number
  sym: Generator
}

export interface Pair {
  do: string
  undo: string
  note?: string
}

export interface Alphabet {
  schema?: string
  substrate?: string
  symbols: SymbolRow[]
  pairs: Pair[]
  byName: Map<string, SymbolRow>
  source: string
}

export interface ParseResult {
  alphabet: Alphabet
  diagnostics: Diagnostic[]
  fatal: boolean
}

export const emptyAlphabet = (source: string): Alphabet => ({
  symbols: [],
  pairs: [],
  byName: new Map(),
  source,
})

// ---------------------------------------------------------------------------
// (a, A) ⇄ Σ₀
// ---------------------------------------------------------------------------

/**
 * §3's alternate encoding, decomposed into Σ₀ coordinates.
 *
 * The trace splits off first: the isotropic part of A contributes (tr/3)·½|L|²,
 * a function of the Casimir, so it can never move the flow. What remains is the
 * traceless diagonal, Frobenius-projected onto the pinned D1, D2 (both unit norm
 * and mutually orthogonal, which is exactly why §2 pins them — it makes this
 * readout reproducible).
 *
 * c_ij = A_ij exactly: §2 defines c₁₂ as quad(e₁e₂ᵀ + e₂e₁ᵀ), giving H = L₁L₂,
 * and buildSymbol places c12 at both A[0][1] and A[1][0]. The conventions agree.
 */
export function coeffsFromAA(
  a: readonly number[],
  A: readonly (readonly number[])[],
): { coeffs: Coeffs; residue: number } {
  const residue = (A[0][0] + A[1][1] + A[2][2]) / 3
  const dg: Vec3 = [A[0][0] - residue, A[1][1] - residue, A[2][2] - residue]
  return {
    coeffs: {
      k1: a[0],
      k2: a[1],
      k3: a[2],
      c12: A[0][1],
      c13: A[0][2],
      c23: A[1][2],
      d1: dg[0] * D1[0] + dg[1] * D1[1] + dg[2] * D1[2],
      d2: dg[0] * D2[0] + dg[1] * D2[1] + dg[2] * D2[2],
    },
    residue,
  }
}

/** Accepts the 5-number traceless form, the 6-number form, or a 3×3 (symmetrized). */
function toSym3(A: readonly number[] | readonly (readonly number[])[]): number[][] | null {
  if (Array.isArray(A) && A.length === 3 && Array.isArray(A[0])) {
    const M = A as readonly (readonly number[])[]
    if (M.some((r) => !Array.isArray(r) || r.length !== 3 || r.some((v) => !Number.isFinite(v)))) {
      return null
    }
    return [
      [M[0][0], 0.5 * (M[0][1] + M[1][0]), 0.5 * (M[0][2] + M[2][0])],
      [0.5 * (M[0][1] + M[1][0]), M[1][1], 0.5 * (M[1][2] + M[2][1])],
      [0.5 * (M[0][2] + M[2][0]), 0.5 * (M[1][2] + M[2][1]), M[2][2]],
    ]
  }
  const v = A as readonly number[]
  if (!Array.isArray(v) || v.some((x) => !Number.isFinite(x))) return null
  if (v.length === 5) {
    // [A₁₁ A₂₂ A₁₂ A₁₃ A₂₃], A₃₃ = −A₁₁−A₂₂
    const [a11, a22, a12, a13, a23] = v
    return [
      [a11, a12, a13],
      [a12, a22, a23],
      [a13, a23, -a11 - a22],
    ]
  }
  if (v.length === 6) {
    // [A₁₁ A₂₂ A₃₃ A₁₂ A₁₃ A₂₃] — trace is split off as the silent residue
    const [a11, a22, a33, a12, a13, a23] = v
    return [
      [a11, a12, a13],
      [a12, a22, a23],
      [a13, a23, a33],
    ]
  }
  return null
}

// ---------------------------------------------------------------------------
// parse
// ---------------------------------------------------------------------------

const TOP_KEYS = new Set(['schema', 'substrate', 'basis', 'symbols', 'pairs', '_comment'])
const ROW_KEYS = new Set(['name', 'sigma0', 'a', 'A', 'amp', 'dur', 'reading', '_comment'])

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v)

/** Thresholds for the "is this row worth having" diagnostics. Heuristics, and
 *  labelled as such in the messages — not claims. */
const SILENT_SPAN = 1e-12
/** §5: "Amplitudes ≪ 0.1 are nearly invisible in the energy". Against the pinned
 *  scales a unit-direction symbol at amp 0.1 spans ~0.1–0.2, so ≪ that. */
const FAINT_SPAN = 0.02
/** Per-step rotation angle at the pinned dt. Coarse steps, not big energies, are
 *  what degrade the integrator (item 7). */
const COARSE_RAD = 0.05
/** Relative distance below which two amp-scaled rows do nearly the same thing. */
const DUPLICATE_REL = 0.02

export function parseAlphabet(raw: unknown, source: string): ParseResult {
  const diagnostics: Diagnostic[] = []
  const push = (level: Level, where: string, message: string) =>
    diagnostics.push({ level, where, message })
  const bail = (message: string): ParseResult => {
    push('error', source, message)
    return { alphabet: emptyAlphabet(source), diagnostics, fatal: true }
  }

  if (!isObj(raw)) {
    return bail('the top level must be a JSON object (with a "symbols" array), not an array or scalar.')
  }

  for (const k of Object.keys(raw)) {
    if (!TOP_KEYS.has(k)) {
      push('warn', source, `unknown top-level key "${k}" — ignored. Expected one of: ${[...TOP_KEYS].join(', ')}.`)
    }
  }

  const schema = typeof raw.schema === 'string' ? raw.schema : undefined
  if (schema !== undefined && schema !== SCHEMA_ID) {
    push('warn', source, `schema is "${schema}"; this app reads "${SCHEMA_ID}". Parsing anyway.`)
  }
  const substrate = typeof raw.substrate === 'string' ? raw.substrate : undefined
  if (substrate !== undefined && substrate !== SUBSTRATE) {
    push('warn', source, `substrate is "${substrate}"; this toy is ${SUBSTRATE}. The coefficients will be read as Σ₀ regardless.`)
  }

  // `basis` pins the ordering used by the array form of `sigma0`.
  let basis: Sigma0Key[] = [...SIGMA0_KEYS]
  if (raw.basis !== undefined) {
    const b = raw.basis
    if (!Array.isArray(b) || b.length !== SIGMA0_KEYS.length || b.some((k) => typeof k !== 'string')) {
      return bail(`"basis" must be an array of the ${SIGMA0_KEYS.length} Σ₀ keys; got ${JSON.stringify(b)}.`)
    }
    const seen = new Set(b as string[])
    if (seen.size !== b.length || !SIGMA0_KEYS.every((k) => seen.has(k))) {
      return bail(`"basis" must be a permutation of ${SIGMA0_KEYS.join(', ')}; got ${(b as string[]).join(', ')}.`)
    }
    basis = b as Sigma0Key[]
    if (basis.some((k, i) => k !== SIGMA0_KEYS[i])) {
      push('warn', source, `"basis" is a permutation of the canonical order (${SIGMA0_KEYS.join(' ')}). Honored for array-form rows, but readouts use the canonical order.`)
    }
  }

  if (!Array.isArray(raw.symbols)) {
    return bail('"symbols" is missing or not an array.')
  }

  const symbols: SymbolRow[] = []
  const byName = new Map<string, SymbolRow>()

  raw.symbols.forEach((entry: unknown, i: number) => {
    const at = `symbols[${i}]`
    if (!isObj(entry)) {
      push('error', at, 'not an object — dropped.')
      return
    }

    // House convention (cf. sandbox/latent.py, tests/test_identity_graph.py):
    // divider objects are skipped, but must carry ONLY _comment, so a typo'd real
    // row can never vanish silently.
    if ('_comment' in entry) {
      const others = Object.keys(entry).filter((k) => k !== '_comment')
      if (others.length === 0) return
      push('error', at, `a "_comment" divider must carry only that key; this one also has ${others.map((k) => `"${k}"`).join(', ')}. Dropped rather than silently half-read.`)
      return
    }

    for (const k of Object.keys(entry)) {
      if (!ROW_KEYS.has(k)) {
        push('warn', at, `unknown key "${k}" — ignored. (Expected: ${[...ROW_KEYS].filter((x) => x !== '_comment').join(', ')}.)`)
      }
    }

    const name = entry.name
    if (typeof name !== 'string' || !NAME_RE.test(name)) {
      push('error', at, `name must be a single ASCII letter (§4/A7 — the string-word path iterates characters); got ${JSON.stringify(name)}. Dropped.`)
      return
    }
    if (byName.has(name)) {
      push('error', `${at} "${name}"`, `duplicate name — names are case-significant, and "${name}" is already defined. Dropped.`)
      return
    }
    const where = `${at} "${name}"`

    const hasSigma0 = entry.sigma0 !== undefined
    const hasAA = entry.a !== undefined || entry.A !== undefined
    if (hasSigma0 && hasAA) {
      push('error', where, 'has both encodings ("sigma0" and "a"/"A"). Give exactly one — §3. Dropped.')
      return
    }
    if (!hasSigma0 && !hasAA) {
      push('error', where, 'has neither encoding. Give "sigma0" (8 Σ₀ coefficients) or "a"/"A" (§3). Dropped.')
      return
    }

    let coeffs: Coeffs
    let residue = 0

    if (hasSigma0) {
      const s = entry.sigma0
      if (Array.isArray(s)) {
        if (s.length !== basis.length) {
          push('error', where, `"sigma0" array must have ${basis.length} numbers, in the order ${basis.join(' ')}; got ${s.length}. Dropped.`)
          return
        }
        if (s.some((v) => !Number.isFinite(v))) {
          push('error', where, '"sigma0" contains a non-finite value. Dropped.')
          return
        }
        coeffs = { ...ZERO }
        basis.forEach((k, j) => (coeffs[k] = s[j] as number))
      } else if (isObj(s)) {
        coeffs = { ...ZERO }
        let bad = false
        for (const [k, v] of Object.entries(s)) {
          if (!SIGMA0_KEYS.includes(k as Sigma0Key)) {
            push('error', where, `unknown Σ₀ key "${k}" in "sigma0". Expected one of: ${SIGMA0_KEYS.join(', ')}. Dropped.`)
            bad = true
            break
          }
          if (!Number.isFinite(v)) {
            push('error', where, `"sigma0.${k}" is not a finite number. Dropped.`)
            bad = true
            break
          }
          coeffs[k as Sigma0Key] = v as number
        }
        if (bad) return
      } else {
        push('error', where, '"sigma0" must be an object of Σ₀ coefficients or an array of 8 numbers. Dropped.')
        return
      }
    } else {
      let a: number[] = [0, 0, 0]
      if (entry.a !== undefined) {
        const av = entry.a
        if (!Array.isArray(av) || av.length !== 3 || av.some((v) => !Number.isFinite(v))) {
          push('error', where, '"a" must be 3 finite numbers (the kick part). Dropped.')
          return
        }
        a = av as number[]
      }
      let A: number[][] = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]
      if (entry.A !== undefined) {
        const M = toSym3(entry.A as number[])
        if (!M) {
          push('error', where, '"A" must be 5 numbers [A₁₁ A₂₂ A₁₂ A₁₃ A₂₃], 6 numbers [A₁₁ A₂₂ A₃₃ A₁₂ A₁₃ A₂₃], or a 3×3 matrix, all finite. Dropped.')
          return
        }
        A = M
      }
      const dec = coeffsFromAA(a, A)
      coeffs = dec.coeffs
      residue = dec.residue
      if (Math.abs(residue) > 1e-12) {
        push('info', where, `isotropic residue ${residue.toFixed(6)} split off from A (H_res = ${residue.toFixed(6)}·½|L|², a function of ψ — SILENT, so it changes nothing). The Σ₀ coefficients below are what remains.`)
      }
    }

    let amp = 1
    if (entry.amp === undefined) {
      push('info', where, 'no "amp" — defaulted to 1.0.')
    } else if (!Number.isFinite(entry.amp)) {
      push('error', where, `"amp" must be a finite number; got ${JSON.stringify(entry.amp)}. Dropped.`)
      return
    } else {
      amp = entry.amp as number
    }

    let dur: number | undefined
    if (entry.dur !== undefined) {
      if (!Number.isFinite(entry.dur) || (entry.dur as number) <= 0) {
        push('error', where, `"dur" must be a positive finite number; got ${JSON.stringify(entry.dur)}. Dropped.`)
        return
      }
      dur = entry.dur as number
      push('warn', where, `"dur" is scoped (§4/A4): run_word honors it, but the batched harness behind the ensemble uses the shared pinned schedule and SILENTLY IGNORES it. This console honors it, so what you see here is the per-word behaviour only.`)
    }

    let reading: string | undefined
    if (entry.reading !== undefined) {
      if (typeof entry.reading !== 'string') {
        push('warn', where, '"reading" is not a string — ignored.')
      } else {
        reading = entry.reading
      }
    }

    const sym = buildSymbol(coeffs, amp)
    const { span, maxGrad } = probe(sym)
    const row: SymbolRow = { name, coeffs, amp, dur, reading, residue, span, maxGrad, sym }
    symbols.push(row)
    byName.set(name, row)

    if (span < SILENT_SPAN) {
      push('warn', where, 'SILENT: ∇H ≡ 0, so H is a function of the Casimir and this symbol does nothing to the flow. (§5 — the isotropy trap at symbol level.)')
    } else if (span < FAINT_SPAN) {
      push('warn', where, `faint: H_σ spans ${span.toExponential(2)} over the sphere, against H₀'s pinned span of 1/3. §5's "amplitudes ≪ 0.1 are nearly invisible in the energy" — likely too small to read.`)
    }
    if (maxGrad * DT > COARSE_RAD) {
      push('warn', where, `coarse step: max |∇H_σ| = ${maxGrad.toFixed(2)}, so RKMK2 rotates up to ${(maxGrad * DT).toFixed(3)} rad per dt = ${DT}. ψ stays exact regardless, but the trajectory resolution does not (item 7).`)
    }
  })

  if (symbols.length === 0) {
    return bail('no valid symbols survived — see the errors above.')
  }

  // ---- pairs -------------------------------------------------------------
  const pairs: Pair[] = []
  const pairKey = new Set<string>()
  if (raw.pairs !== undefined) {
    if (!Array.isArray(raw.pairs)) {
      push('warn', source, '"pairs" is not an array — ignored.')
    } else {
      raw.pairs.forEach((p: unknown, i: number) => {
        const at = `pairs[${i}]`
        let d: unknown, u: unknown, note: string | undefined
        if (Array.isArray(p) && p.length === 2) {
          ;[d, u] = p
        } else if (isObj(p)) {
          d = p.do
          u = p.undo
          if (typeof p.note === 'string') note = p.note
        } else {
          push('warn', at, 'must be ["do","undo"] or {"do":…,"undo":…} — ignored.')
          return
        }
        if (typeof d !== 'string' || typeof u !== 'string') {
          push('warn', at, 'both members must be symbol names — ignored.')
          return
        }
        for (const n of [d, u]) {
          if (!byName.has(n)) push('warn', at, `names "${n}", which is not a symbol in this alphabet.`)
        }
        pairs.push({ do: d, undo: u, note })
        pairKey.add([d, u].sort().join(' '))
      })
    }
  }

  // ---- cross-row checks (§5's "avoid near-duplicates"; item 3's static cousin)
  const vecs = symbols.map((r) => SIGMA0_KEYS.map((k) => r.amp * r.coeffs[k]))
  const nrm = vecs.map((v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)))
  for (let i = 0; i < symbols.length; i++) {
    for (let j = i + 1; j < symbols.length; j++) {
      if (nrm[i] < 1e-12 || nrm[j] < 1e-12) continue
      const diff = Math.sqrt(vecs[i].reduce((s, x, k) => s + (x - vecs[j][k]) ** 2, 0))
      const rel = diff / Math.max(nrm[i], nrm[j])
      const a = symbols[i].name
      const b = symbols[j].name
      if (rel < DUPLICATE_REL) {
        push('warn', `symbols "${a}" / "${b}"`, `near-duplicate: the amp-scaled Σ₀ vectors differ by ${(rel * 100).toFixed(1)}%. They do the same thing (§5 — avoid near-duplicates).`)
        continue
      }
      if (pairKey.has([a, b].sort().join(' '))) continue
      const cos = vecs[i].reduce((s, x, k) => s + x * vecs[j][k], 0) / (nrm[i] * nrm[j])
      if (Math.abs(cos) > 0.999) {
        push('info', `symbols "${a}" / "${b}"`, cos > 0
          ? `collinear, same sign — an intensity variant (${a}: ‖amp·c‖ ${nrm[i].toFixed(3)}, ${b}: ${nrm[j].toFixed(3)}). §4 endorses this; noted so it is deliberate.`
          : `collinear, opposite sign — a semantic-opposite shape. Consider registering it in "pairs" so item 9 reports its return distance (§3: opposites are not inverses).`)
      }
    }
  }

  return {
    alphabet: { schema, substrate, symbols, pairs, byName, source },
    diagnostics,
    fatal: false,
  }
}

// ---------------------------------------------------------------------------
// words and timelines
// ---------------------------------------------------------------------------

/** The live-slider pseudo-symbol. Non-letter by design: names are validated
 *  against /^[A-Za-z]$/, so this can never collide with an authored symbol —
 *  unlike the prototype's `q`, which collides the moment someone authors one. */
export const BLEND = '*'

export interface Token {
  ch: string
  ok: boolean
}

/** The single place a word is checked against the alphabet — it replaced both the
 *  hard-coded /[^xyzuvwdeq]/ input filter and the playback-time filter, which had
 *  to agree by hand. Note there is no case folding anywhere: `x` and `X` are
 *  different symbols (§4/A7). */
export function scanWord(text: string, known: ReadonlySet<string>): Token[] {
  return Array.from(text).map((ch) => ({ ch, ok: ch === BLEND || known.has(ch) }))
}

export const wordOf = (scan: readonly Token[]): string[] =>
  scan.filter((t) => t.ok).map((t) => t.ch)

export interface Segment {
  phase: 'gap' | 'event'
  steps: number
  name: string | null
  sym: Generator | null
}

export interface LiveRow {
  coeffs: Coeffs
  amp: number
}

const stepsFor = (tau: number, dt: number) => Math.max(1, Math.round(tau / dt))

/** gap–event–gap, with each event's dwell taken from its row's `dur` when set. */
export function buildTimeline(
  names: readonly string[],
  byName: ReadonlyMap<string, SymbolRow>,
  liveRow: () => LiveRow,
  opts: { dt?: number; tauEvent?: number; tauGap?: number } = {},
): Segment[] {
  const dt = opts.dt ?? DT
  const tauEvent = opts.tauEvent ?? TAU_EVENT
  const tauGap = opts.tauGap ?? TAU_GAP
  const gap = (): Segment => ({ phase: 'gap', steps: stepsFor(tauGap, dt), name: null, sym: null })

  const tl: Segment[] = [gap()]
  for (const n of names) {
    if (n === BLEND) {
      const r = liveRow()
      tl.push({
        phase: 'event',
        steps: stepsFor(tauEvent, dt),
        name: BLEND,
        sym: buildSymbol(r.coeffs, r.amp),
      })
    } else {
      const row = byName.get(n)
      if (!row) continue
      tl.push({ phase: 'event', steps: stepsFor(row.dur ?? tauEvent, dt), name: n, sym: row.sym })
    }
    tl.push(gap())
  }
  return tl
}

export const totalSteps = (tl: readonly Segment[]) => tl.reduce((s, seg) => s + seg.steps, 0)

// ---------------------------------------------------------------------------
// display helpers
// ---------------------------------------------------------------------------

/** "0.60·k₁ + 0.40·c₂₃" — the canonical Σ₀ reading of a point. */
export function formatCoeffs(c: Coeffs): string {
  const terms: string[] = []
  for (const k of SIGMA0_KEYS) {
    const v = c[k]
    if (Math.abs(v) <= 1e-9) continue
    const prefix = terms.length && v > 0 ? '+ ' : v < 0 ? '− ' : ''
    terms.push(`${prefix}${Math.abs(v).toFixed(2)}·${SIGMA0_LABEL[k]}`)
  }
  return terms.length ? terms.join(' ') : '0 (no Hσ)'
}

const round6 = (v: number) => Math.round(v * 1e6) / 1e6

/** A ready-to-paste schema row for the current blend — closes the authoring loop:
 *  tune the sliders, copy, paste into your alphabet file. */
export function toJsonRow(c: Coeffs, amp: number, name = '?'): string {
  const sigma0: Record<string, number> = {}
  for (const k of SIGMA0_KEYS) if (Math.abs(c[k]) > 1e-9) sigma0[k] = round6(c[k])
  return JSON.stringify({ name, sigma0, amp: round6(amp), reading: '' })
}

/** Effective intensity. §4/A3's convention assumes unit-norm axes, but §9's own
 *  rows are not unit-norm (‖s‖ = √6/3), so `amp` alone is not intensity —
 *  |amp|·‖c‖ is. Never auto-normalize: for `s` the shortfall is the dropped
 *  residue and is meaningful. */
export const intensity = (r: { coeffs: Coeffs; amp: number }) =>
  Math.abs(r.amp) * coeffsNorm(r.coeffs)
