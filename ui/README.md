# equate review UI

A browser UI for reviewing what `equate` matched: paste two collections, get the optimal one-to-one matching, and correct it. It talks to the HTTP service in `equate/service.py`.

## Run it

```bash
pip install 'equate[service]'
python -m equate.service serve            # the API on http://127.0.0.1:8000

cd ui
npm install
npm run dev                               # the UI on http://localhost:5173 (proxies /api)
```

To serve the built UI from the service itself (one process, one origin, no dev server):

```bash
cd ui && npm install && npm run build && cd ..
python -m equate.service serve ui=ui/dist   # UI at http://127.0.0.1:8000/, API under /api
```

(`equate.service.mk_app(ui_dir="ui/dist")` is the same app to mount elsewhere.) The wheel does not ship a prebuilt bundle yet; build it from a checkout.

`EQUATE_API_TARGET=http://host:port npm run dev` points the dev proxy at another service; `VITE_EQUATE_API=https://... npm run build` builds a UI that calls a service directly (the service allows CORS from the vite dev origins by default; see `mk_app(cors_origins=...)`).

## What it does

- **Review queue**: every left item with its current match, score, and the margin between its top two candidates; the most uncertain items come first, confirmed ones last.
- **Pair detail**: the selected item's candidates, each with a side-by-side field diff (per-field similarity; a missing field is shown as missing, not as different). *Confirm* the current match, *Use this match* for another candidate, *Reject*, or undo.
- **Optimistic edits**: an edit shows at once; the server then re-solves the assignment from the cached scores and the rows it changed flash. Edits are sent in order; a failed one re-reads the session from the server.
- **Is this a match?**: the active-learning widget asks about the least certain open pair; the answer is applied as a confirm or a reject.
- The session is in the URL (`?session=<id>`), so a reload or a shared link reopens it and Back returns to the setup screen. The setup form's draft survives a reload.

## How it is built

- `src/schema/equate.ts`: Zod schemas for every payload, mirroring the Python dataclasses. `src/schema/fields.json` is the Python side's field list (`python -m equate.service fields > ui/src/schema/fields.json`); a test on each side fails when they drift.
- `src/review/collection.ts`: the review queue as a zodal collection; the table's columns come from `toColumnDefs`, cells from the `@zodal/ui-shadcn` registry.
- `src/state/`: a zustand + immer store; `optimistic.ts` holds the pure predict/reconcile functions; the draft goes through a zodal `DataProvider` (`@zodal/store-localstorage`).
- `src/route.ts`: the only module that touches `history`.

`npm test` runs the unit tests; `npm run build` typechecks and builds to `dist/`; `npm run ci-local` runs both. Hosted CI runs the same on every change under `ui/` (`.github/workflows/ui-ci.yml`, the wads npm workflow, node 20 and 22; publishing to npm is off).

`@zodal/ui-shadcn@0.1.0` still declares `@zodal/core`/`@zodal/ui` `^0.1` as peers while this app uses `0.2`; `package.json` bridges that with `overrides` until the renderer is republished.
