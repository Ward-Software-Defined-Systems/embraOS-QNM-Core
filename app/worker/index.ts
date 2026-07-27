/**
 * The app is fully static: Cloudflare's asset server matches built assets BEFORE this
 * script runs, so a request only reaches here when nothing in dist/client matched.
 *
 * That makes the 404 below load-bearing rather than decorative. A client-side
 * `fetch('/alphabet.json')` for a file that isn't there gets an honest 404 instead of
 * index.html with a 200 (which is what the single-page-application fallback would
 * otherwise hand back, and which turns a missing file into a baffling JSON parse error).
 */
export default {
  fetch(request) {
    const url = new URL(request.url)

    if (url.pathname.startsWith('/api/')) {
      return Response.json({ error: 'not implemented' }, { status: 501 })
    }
    return new Response(null, { status: 404 })
  },
} satisfies ExportedHandler<Env>
