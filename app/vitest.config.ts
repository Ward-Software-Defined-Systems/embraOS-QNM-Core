import { defineConfig } from 'vitest/config'

// The core (Σ₀ algebra, the alphabet loader) is pure and framework-agnostic, so it
// runs in a plain Node env — no jsdom, no WebGL. Deliberately a SEPARATE config from
// vite.config.ts so the Cloudflare plugin never loads during tests.
export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
})
