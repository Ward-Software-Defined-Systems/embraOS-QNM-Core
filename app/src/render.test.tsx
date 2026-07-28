/**
 * A render smoke test — no jsdom, no WebGL.
 *
 * renderToString runs the whole component body (state initialisers, useMemo,
 * every JSX branch) while skipping effects, so the three.js scene never starts.
 * That is exactly the half a unit test can't otherwise reach: a stale identifier
 * left behind by the Σ₀ refactor would blank the page in a browser and pass every
 * other test here.
 */
import { describe, it, expect } from 'vitest'
import { renderToString } from 'react-dom/server'
import defaultJson from '../public/alphabet.json'
import { parseAlphabet } from './core/alphabet.ts'
import AlphabetPanel from './ui/AlphabetPanel.tsx'
import AlphabetVisualizer from './viz/AlphabetVisualizer.jsx'

const { alphabet, diagnostics } = parseAlphabet(defaultJson, 'public/alphabet.json')

/** React marks interpolation boundaries with <!-- --> in SSR output; strip them
 *  so assertions can read the text a user actually sees. */
const render = (el: React.ReactElement) => renderToString(el).replaceAll('<!-- -->', '')

describe('AlphabetVisualizer renders', () => {
  const html = render(<AlphabetVisualizer />)

  it('keeps every panel the prototype had', () => {
    for (const panel of ['Presets', 'Blend — Hσ over Σ₀', 'Display', 'Word console', 'Notes']) {
      expect(html).toContain(panel)
    }
    expect(html).toContain('Alphabet') // and the one new section
  })

  it('keeps all twelve preset chips, including the v1 reference blends', () => {
    for (const chip of ['k₁', 'k₂', 'k₃', 'c₁₂', 'c₁₃', 'c₂₃', 'd₁', 'd₂']) {
      expect(html).toContain(chip)
    }
    for (const chip of ['x (v1)', 's (v1)', 'g (v1, trimmed)', 'clear']) {
      expect(html).toContain(chip)
    }
  })

  it('keeps the three ℓ-group headings and the Hσ readout', () => {
    expect(html).toContain('ℓ=1 · reorient')
    expect(html).toContain('ℓ=2 · cross-couple')
    expect(html).toContain('ℓ=2 · reshape (d₁ = the self direction)')
    expect(html).toContain('Hσ = ')
    expect(html).toContain('0 (no Hσ)') // the blend starts blank
  })

  it('keeps the Display toggles and the playback controls', () => {
    expect(html).toContain('Idle view includes H₀')
    expect(html).toContain('Flow arrows')
    expect(html).toContain('Ambient particles')
    expect(html).toContain('▶ play')
    expect(html).toContain('reset')
  })

  it('no longer advertises the Σ₀ letters as an alphabet', () => {
    // The old legend hard-coded "x y z → k₁ k₂ k₃ …". That mapping was the bug.
    expect(html).not.toContain('→ k₁ k₂ k₃')
    expect(html).toContain('Case matters')
  })
})

describe('AlphabetPanel renders a loaded alphabet', () => {
  const html = render(
    <AlphabetPanel
      alphabet={alphabet}
      diagnostics={diagnostics}
      sourceLabel="public/alphabet.json"
      isDefault
      onAppend={() => {}}
      onSolo={() => {}}
      onLoadText={() => {}}
      onReset={() => {}}
    />,
  )

  it('shows every symbol with its Σ₀ reading', () => {
    for (const row of alphabet.symbols) expect(html).toContain(`>${row.name}<`)
    expect(html).toContain('0.60·k₁ + 0.40·c₂₃') // m
    expect(html).toContain('an authored blend')
  })

  it('shows the diagnostics count and the registered pair', () => {
    expect(html).toContain('0 error')
    expect(html).toContain('1 warning')
    expect(html).toContain('p n')
  })

  it('shows the dwell of the one row that sets it', () => {
    expect(html).toContain('dur')
  })
})
