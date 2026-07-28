import { memo, useRef, useState } from 'react'
import type { Alphabet, Diagnostic, SymbolRow } from '../core/alphabet.ts'
import { formatCoeffs, intensity } from '../core/alphabet.ts'
import { H0_SPAN } from '../core/lie.ts'

/** The prototype's chip styling, shared so the two panels can't drift apart. */
export const CHIP = 'px-2 py-1 rounded border text-xs mono transition-colors'

const LEVEL_COLOR: Record<Diagnostic['level'], string> = {
  error: '#E2694A',
  warn: '#d8b45a',
  info: '#6FA0E8',
}

interface Props {
  alphabet: Alphabet
  diagnostics: Diagnostic[]
  sourceLabel: string
  isDefault: boolean
  onAppend: (names: string[]) => void
  onSolo: (row: SymbolRow) => void
  onLoadText: (name: string, text: string) => void
  onReset: () => void
}

function Row({
  row,
  onAppend,
  onSolo,
}: {
  row: SymbolRow
  onAppend: (names: string[]) => void
  onSolo: (row: SymbolRow) => void
}) {
  const silent = row.span < 1e-12
  return (
    <div className="py-1 border-b hair last:border-b-0">
      <div className="flex items-center gap-2">
        <button
          className={CHIP + ' hair hover:border-[#3FD68C] hover:text-[#3FD68C] w-7 text-center shrink-0'}
          onClick={() => onAppend([row.name])}
          title={`append "${row.name}" to the word`}
        >
          {row.name}
        </button>
        <span className="mono text-xs flex-1 min-w-0 truncate" title={formatCoeffs(row.coeffs)}>
          {formatCoeffs(row.coeffs)}
        </span>
        <button
          className="dim2 hover:text-[#3FD68C] shrink-0"
          onClick={() => onSolo(row)}
          title="load into the sliders — inspect the field without playing"
        >
          solo
        </button>
      </div>
      <div className="flex flex-wrap items-center gap-x-2.5 mono dim2 mt-0.5 pl-9">
        <span>
          amp <span className="val">{row.amp}</span>
        </span>
        <span title="|amp|·‖c‖ in Σ₀ coordinates — amp alone is not intensity when ‖c‖ ≠ 1">
          ‖amp·c‖ <span className="val">{intensity(row).toFixed(3)}</span>
        </span>
        <span title={`max − min of H_σ over ψ = 1, against H₀'s span of ${H0_SPAN.toFixed(3)}`}>
          span <span className="val">{row.span.toFixed(3)}</span>
        </span>
        {row.dur !== undefined && (
          <span title="per-word dwell (§4/A4 — honored here, ignored by the batched harness)">
            dur <span className="val">{row.dur}</span>
          </span>
        )}
        {Math.abs(row.residue) > 1e-12 && (
          <span title="isotropic part split off from A — a function of ψ, so silent">
            residue <span className="val">{row.residue.toFixed(3)}</span>
          </span>
        )}
        {silent && <span style={{ color: '#d8b45a' }}>SILENT</span>}
      </div>
      {row.reading && (
        <div className="dim2 pl-9 leading-snug" title={row.reading}>
          {row.reading}
        </div>
      )}
    </div>
  )
}

function AlphabetPanel({
  alphabet,
  diagnostics,
  sourceLabel,
  isDefault,
  onAppend,
  onSolo,
  onLoadText,
  onReset,
}: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const errors = diagnostics.filter((d) => d.level === 'error')
  const warns = diagnostics.filter((d) => d.level === 'warn')
  const [open, setOpen] = useState(false)
  const expanded = open || errors.length > 0

  const pick = async (f: File | undefined | null) => {
    if (!f) return
    onLoadText(f.name, await f.text())
  }

  return (
    <section>
      <div className="eyebrow mb-1.5">Alphabet</div>

      <div className="flex items-center gap-1.5 mb-1.5">
        <button
          className={CHIP + ' hair hover:border-[#3FD68C]'}
          onClick={() => fileRef.current?.click()}
        >
          load JSON…
        </button>
        <button
          className={CHIP + ' hair hover:border-[#6FA0E8]'}
          onClick={onReset}
          disabled={isDefault}
          style={isDefault ? { opacity: 0.4 } : undefined}
        >
          reset
        </button>
        <input
          ref={fileRef}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(e) => {
            void pick(e.target.files?.[0])
            e.target.value = '' // so re-picking the same file after an edit still fires
          }}
        />
      </div>
      <div className="dim2 mb-1.5 truncate" title={sourceLabel}>
        source: <span className="val">{sourceLabel}</span>
        {isDefault && ' — edit it and refresh, or drop a file anywhere on this page'}
      </div>

      <button
        className="w-full text-left mono text-xs px-2 py-1 rounded panel mb-1.5"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="val">{alphabet.symbols.length}</span>
        <span className="dim"> symbols · </span>
        <span style={{ color: errors.length ? LEVEL_COLOR.error : undefined }}>
          {errors.length} error{errors.length === 1 ? '' : 's'}
        </span>
        <span className="dim"> · </span>
        <span style={{ color: warns.length ? LEVEL_COLOR.warn : undefined }}>
          {warns.length} warning{warns.length === 1 ? '' : 's'}
        </span>
        <span className="dim float-right">{expanded ? '▾' : '▸'}</span>
      </button>

      {expanded && (
        <div className="mb-2 flex flex-col gap-1.5">
          {diagnostics.length === 0 && <div className="dim2">nothing to report.</div>}
          {diagnostics.map((d, i) => (
            <div key={i} className="text-xs leading-snug">
              <span className="mono" style={{ color: LEVEL_COLOR[d.level] }}>
                {d.level}
              </span>
              <span className="dim2 mono"> {d.where}</span>
              <div className="dim">{d.message}</div>
            </div>
          ))}
        </div>
      )}

      <div className="panel rounded px-2 py-0.5">
        {alphabet.symbols.map((row) => (
          <Row key={row.name} row={row} onAppend={onAppend} onSolo={onSolo} />
        ))}
        {alphabet.symbols.length === 0 && (
          <div className="dim2 py-2">no symbols loaded.</div>
        )}
      </div>

      {alphabet.pairs.length > 0 && (
        <div className="mt-2">
          <div className="dim2 mb-1">registered do/undo pairs (§3 — opposites, not inverses)</div>
          <div className="flex flex-wrap gap-1.5">
            {alphabet.pairs.map((p, i) => (
              <button
                key={i}
                className={CHIP + ' hair hover:border-[#3FD68C] hover:text-[#3FD68C]'}
                onClick={() => onAppend([p.do, p.undo])}
                title={p.note ?? `append "${p.do}${p.undo}"`}
              >
                {p.do} {p.undo}
              </button>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

export default memo(AlphabetPanel)
