# Σ₀ on the sphere — an alphabet authoring aid

An interactive visualizer for the `so(3)*` Lie–Poisson toy of `sandbox/lie_poisson.py`, built to
make authoring an input alphabet Σ (`docs/ALPHABET-AUTHORING.md`) a matter of editing a file and
looking at the result.

The sphere is the Casimir leaf ψ = |L|² = 1. Colour bands are level sets of the displayed
generator — on the sphere those *are* its streamlines. The word console plays a word as a
gap–event–gap sequence and reports ψ drift live: the integrator is RKMK2, one rotation per step, so
ψ is exact and never renormalized. That drift counter is the mechanism claim, running.

## Run it

```
npm install
npm run dev       # vite dev server
npm test          # vitest — Σ₀'s pinned identities, the loader, a render smoke test
npm run build     # tsc -b && vite build
npm run lint      # oxlint
npm run deploy    # build && wrangler deploy   (needs `wrangler login` once)
```

Node 22 (`.node-version` pins 22.23.1).

`worker-configuration.d.ts` is generated, not committed — `prebuild` runs `wrangler types` before
every build. If your editor complains about `Env` in `worker/index.ts` on a fresh clone, run
`npm run cf-typegen` once.

## The alphabet

`public/alphabet.json` holds the alphabet the word console plays. It ships with the worked example
from `docs/ALPHABET-AUTHORING.md` §9 "Handing it over" — that is example content and a schema
reference, not an authored alphabet.

Two ways to use your own:

- **Editing locally**: change `public/alphabet.json` and refresh the browser. No rebuild — the file
  is fetched at runtime, with cache-busting, precisely so this loop is short.
- **On the deployed page**: drop a `.json` file anywhere on the window, or use *load JSON…*. It is
  read in the browser and remembered in `localStorage`; nothing is uploaded. *reset* goes back to
  the bundled default.

### Schema — `embraos.alphabet/1`

Every row IS a Hamiltonian (§1): `H_σ = amp · ( a·L + ½ LᵀAL )`, given as a point in the pinned
base Σ₀ = {k₁ k₂ k₃ | c₁₂ c₁₃ c₂₃ | d₁ d₂} plus intensity and dwell.

```jsonc
{
  "schema": "embraos.alphabet/1",
  "substrate": "so(3)*",
  "basis": ["k1","k2","k3","c12","c13","c23","d1","d2"],   // optional; pins array-form order
  "symbols": [
    { "_comment": "divider objects are skipped — but must carry ONLY _comment" },
    { "name": "m",                       // single ASCII letter, CASE-SIGNIFICANT (§4/A7)
      "sigma0": { "k1": 0.6, "c23": 0.4 },  // sparse: missing keys are 0
      "amp": 1.0,                        // optional, defaults to 1; sign free
      "dur": 0.5,                        // optional; per-word runs only (§4/A4)
      "reading": "reorient about e₁ while coupling axes 2↔3" }
  ],
  "pairs": [ { "do": "p", "undo": "n", "note": "opposites, not inverses" } ]
}
```

`sigma0` also accepts an ordered array of 8 numbers. §3's alternate `(a, A)` encoding works too —
`"a": [a₁,a₂,a₃]` with `"A"` as 5 numbers `[A₁₁ A₂₂ A₁₂ A₁₃ A₂₃]`, 6 numbers including `A₃₃`, or a
3×3 matrix. Give exactly one encoding per row. When `A` carries a trace, the isotropic part is split
off and reported as the **silent residue**: it contributes `(tr/3)·½|L|²`, a function of ψ, so it
can never move the flow. That readout is battery item 9, live.

### What the panel tells you

Per row: the canonical Σ₀ coefficients, `amp`, `‖amp·c‖`, the **span** of `H_σ` over the sphere
(read against H₀'s pinned span of 1/3), `dur`, any residue, and a SILENT flag. Note `amp` alone is
not intensity — §9's own rows are not unit-norm (‖s‖ = √6/3), so `|amp|·‖c‖` is the number that
matters. The app never auto-normalizes: for `s` the shortfall is the dropped residue and is
meaningful.

Diagnostics distinguish **errors** (that row is dropped, the rest still load), **warnings** (the row
loads — silent, faint, coarse-stepping, near-duplicate, scoped `dur`), and **info**. A malformed
file never blanks the page.

This is an authoring aid, not the battery. The authoritative checks are the Python ones in
`docs/ALPHABET-AUTHORING.md` §6.

## Layout

- `src/core/` — pure, typed, tested. `lie.ts` is the port of `sandbox/lie_poisson.py` plus Σ₀ itself;
  `alphabet.ts` is the schema, parser and word machinery. Σ₀ is the *coordinate system*, pinned by
  the mathematics; the alphabet is authored content. Keeping those apart is the point.
- `src/loadAlphabet.ts` — the impure edge (fetch, file drop, localStorage).
- `src/ui/`, `src/viz/` — the panel and the three.js scene.
- `worker/index.ts` — a stub. The app is fully static, so Cloudflare's asset server answers first
  and this only sees unmatched routes; its 404 is what gives a missing `alphabet.json` an honest
  error instead of `index.html` with a 200.

## Deploy

Cloudflare Workers with static assets (not Pages), via `@cloudflare/vite-plugin`. `npm run build`
writes `dist/client/` (the assets) and `dist/qnm_alphabet/` (the worker plus a generated
`wrangler.json` with `"directory": "../client"` injected); `.wrangler/deploy/config.json` points a
bare `wrangler deploy` at that generated config. `wrangler.jsonc` has no `routes`, so it deploys to
`workers.dev`; a custom domain is one entry when wanted.
