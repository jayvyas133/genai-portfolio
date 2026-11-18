# ADR (Architecture Decision Record) 0021: One web page and gateway for all three projects

- **Status:** accepted
- **Date:** 2026-11-18
- **Scope:** portfolio

## Context

The three projects are HTTP (Hypertext Transfer Protocol) APIs on
three ports (8000, 8001, 8002). A reviewer should be able to try all of
them from one URL in a browser, without curl. The forces: no changes to
the finished APIs, nothing a reviewer must build, and the Deep Research
progress stream (SSE, Server-Sent Events) must still arrive live.

## Options considered

1. **Each API serves its own page.** Three URLs, and HTML mixed into
   three backends.
2. **One page calling the three ports directly.** The browser then makes
   cross-origin requests, so every API needs CORS (Cross-Origin Resource
   Sharing) configuration, and the page hard-codes ports.
3. **One page behind a reverse proxy** (chosen). nginx serves the page
   and forwards `/api/documind/`, `/api/research/` and `/api/sql/` to
   the three services on the compose network. One origin, so no CORS;
   this is how several backend services are usually put behind one URL.

For the page itself, a framework (React and a build step) was rejected:
the page is three forms and a chat, about 250 lines of plain
JavaScript.

## Decision

- `web/site/` (`index.html`, `app.js`, `style.css`) is served by
  `nginx:1.27-alpine` with `web/nginx.conf`; compose service `web` on
  port 8080 depends on all three APIs, so
  `docker compose -f infra/docker-compose.yml up web` starts everything.
- Proxy timeouts follow each API's own limit: 60 s for DocuMind (30 s
  limit), 180 s for Text-to-SQL (120 s), and an unbuffered, 1-hour read
  timeout for the Deep Research event stream (the API sends heartbeats).
- Model text is inserted with `textContent`, never `innerHTML`, so a
  model reply cannot inject markup into the page.
- The chat disables "send" while a turn runs, matching the API's one
  turn per conversation (409).
- The nginx config sits outside the served folder, so it is not
  downloadable.

## Trade-offs

- **No authentication or rate limiting** at the gateway; local use only.
- **No automated browser tests.** Verified by calling every route the
  page uses through the gateway (Cloudflare models): DocuMind answered
  with citations in 3.1 s, two Text-to-SQL turns in 3.4 s and 4.4 s, and
  a Deep Research job streamed 9 events live and finished in 18 s.
- **DocuMind needs the Cloudflare embedding model** its index was built
  with, so the page needs `.env`'s Cloudflare provider, not Ollama.

## Addendum (2026-11-18): a React version alongside the plain page

Jay asked for a React version as well, to show the skill, while keeping
the plain page.

- `web/react/`: Vite + React 19 + TypeScript (strict), written by hand
  (no scaffolder): 3 tab components, a typed API module mirroring the
  Pydantic response models, and the plain page's stylesheet, shared. 24
  npm packages; the build is 71 KB gzipped.
- Served by the same gateway at `/react/` (single-page-app fallback to
  its `index.html`); the plain page stays at `/`. Same API calls, same
  safety: React escapes text by default, as `textContent` does.
- `web/Dockerfile` is multi-stage: a Node stage runs `npm ci` and
  `npm run build` (type check, then bundle); the final image is plain
  nginx with both pages and no Node. The `web` compose service now builds
  this image instead of mounting files.
- `npm run dev` gives hot reload, proxying `/api` to the running gateway.

Trade-off: a Node toolchain and a build step to maintain, which the plain
page avoids. Tabs stay mounted when hidden, so each keeps its state
(a running research stream, the chat thread) when switching.
