import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { cloudflare } from '@cloudflare/vite-plugin'

// cloudflare() last — it reads wrangler.jsonc itself and owns the build output layout.
export default defineConfig({
  plugins: [react(), tailwindcss(), cloudflare()],
})
