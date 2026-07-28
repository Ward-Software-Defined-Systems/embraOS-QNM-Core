/**
 * The impure edge: everything that touches the network, the filesystem, or
 * localStorage. core/alphabet.ts stays pure so it can be tested in a plain node
 * env; all the ways a file can fail to arrive live here.
 */

import type { ParseResult } from './core/alphabet.ts'
import { emptyAlphabet, parseAlphabet } from './core/alphabet.ts'

export const DEFAULT_URL = '/alphabet.json'
const LS_KEY = 'qnm-alphabet:file'

export interface StoredFile {
  name: string
  text: string
}

const fatal = (source: string, message: string): ParseResult => ({
  alphabet: emptyAlphabet(source),
  diagnostics: [{ level: 'error', where: source, message }],
  fatal: true,
})

export function parseText(text: string, source: string): ParseResult {
  let raw: unknown
  try {
    raw = JSON.parse(text)
  } catch (e) {
    return fatal(source, `not valid JSON — ${(e as Error).message}`)
  }
  return parseAlphabet(raw, source)
}

/**
 * Fetch the bundled default.
 *
 * Two guards, both earned:
 *
 * 1. cache: 'no-store' + a buster. public/ assets are unhashed, so both the
 *    browser and Cloudflare's asset layer will happily serve a stale copy — and
 *    the author concludes their edit didn't save. The whole point of loading at
 *    runtime is edit-then-refresh, so a stale read defeats the feature.
 * 2. an HTML sniff. Under `not_found_handling: "single-page-application"` a
 *    missing asset can come back as index.html with HTTP 200, and JSON.parse
 *    then reports a syntax error at "<" that explains nothing. (The deployed
 *    Worker's 404 stub usually catches this first; `vite dev` and `vite preview`
 *    do not, so the sniff stays.)
 */
export async function fetchDefault(url = DEFAULT_URL): Promise<ParseResult> {
  let res: Response
  try {
    res = await fetch(`${url}?t=${Date.now()}`, { cache: 'no-store' })
  } catch (e) {
    return fatal(url, `could not be fetched — ${String(e)}`)
  }
  if (!res.ok) return fatal(url, `HTTP ${res.status} ${res.statusText}`)

  const text = await res.text()
  const ct = res.headers.get('content-type') ?? ''
  if (ct.includes('text/html') || /^\s*</.test(text)) {
    return fatal(
      url,
      'was served as HTML, not JSON — the single-page-application fallback returned index.html for a file that is not there. Check that public/alphabet.json exists.',
    )
  }
  return parseText(text, url)
}

// ---------------------------------------------------------------------------
// a dropped/picked file, remembered across refreshes so the edit loop survives
// ---------------------------------------------------------------------------

export function readStored(): StoredFile | null {
  try {
    const s = localStorage.getItem(LS_KEY)
    if (!s) return null
    const v = JSON.parse(s) as StoredFile
    return typeof v?.name === 'string' && typeof v?.text === 'string' ? v : null
  } catch {
    return null
  }
}

export function writeStored(file: StoredFile): void {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(file))
  } catch {
    /* quota or a private window — the file still loaded, it just won't persist */
  }
}

export function clearStored(): void {
  try {
    localStorage.removeItem(LS_KEY)
  } catch {
    /* ignore */
  }
}
