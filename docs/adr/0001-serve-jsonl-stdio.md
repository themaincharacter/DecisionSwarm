# 0001. Long-lived packs serve JSONL decisions over stdio

Date: 2026-10-24
Status: Accepted

## Context

Real-time consumers (the planned ViZDoom/Laya demo) need swarm decisions about
6x/sec. `decision-swarm run` is one-shot: it spawns workers, loads weights,
answers one payload, exits — so every call would pay model load. A built pack
must be runnable as a long-lived decision service with a warm swarm.

Packaging shapes considered:

- **(a) JSONL over stdio (`serve`)** — one long-lived process per pack, one
  JSON object per line; no new dependencies; reuses the existing stdio framing
  from `decision_swarm.runtime.protocol`; any language can drive it, including
  the Tauri app via the sidecar.
- **(b) Python import of `orchestrator.Swarm`** — works but couples every
  consumer to Python plus a repo/pack checkout.
- **(c) HTTP/WebSocket** — extra dependency and process lifecycle for a
  consumer that lives on the same machine; revisit only if a browser client
  appears.

The Doom consumer is a local Python script (ViZDoom + PIL + imageio), so (a)
fits; (b) remains available as the no-subcommand fallback (import the runtime
package and call the orchestrator directly).

## Decisions

1. **`serve` speaks the (a) transport.** `decision-swarm serve --pack DIR
   [--models-dir DIR] [--envs-dir DIR] [--offline]` (also `<pack-slug> serve`
   from a built pack's runtime) plans cells and spawns workers once, then loops
   over JSON lines on stdio. Wire format:

   ```jsonl
   <- {"type": "ready", "name": "...", "cells": [...], "views": [...], "warnings": []}
   -> {"type": "query", "id": 1, "views": {...}, "questions": {...}}
   <- {"type": "result", "id": 1, "cells": {...}}
   <- {"type": "error", "id": 1, "message": "..."}
   -> {"type": "shutdown"}
   ```

   - `query` is exactly the `run --input` payload plus `type` and `id`
     (`id` is echoed back for correlation; it may be omitted).
   - `result` carries the same `cells` object as `run`'s JSON output.
   - `error` answers one query (or the session when `id` is null) without
     ending the session; malformed lines never kill the service.
   - `shutdown`, or end-of-input, exits cleanly and reaps the workers.

2. **The worker protocol gains a `ready` message** (`{"type": "ready"}`),
   emitted after a worker has loaded its model. `Swarm.start()` waits for it
   (workers load in parallel), so "started" means "warm" and `serve` can emit
   its own `ready` truthfully. Workers that fail to start surface their error
   as per-cell warnings, keeping one dead cell from taking down the swarm.

3. **The generated pack shim exposes `serve` and a real `run`.** Its `run`
   delegates to `orchestrator.run` (the generated README claimed `run` worked
   while it was a stub — that is now true); its `serve` runs the same loop.

4. **Per-decision episode logs stay consumer-side.** The engine is a pure
   decision service and answers only; frame/reward/episode semantics belong to
   the consumer, which can tee the JSONL stream for replay.

## Consequences

- Consumers integrate by spawning one process and exchanging JSONL lines; no
  Python or repo checkout is required beyond `pip install ./runtime` in a pack.
- The first query no longer pays model load inside `run` either — the cost
  moves to swarm start, where it is visible and parallel.
- A browser-only consumer would need (c) later; the serve loop is small enough
  to sit behind an HTTP shim without changing the engine.
- Cell envs still install only `cell.runtime.package_source` (CPU torch and no
  torchvision on Windows); that is a separate concern from transport and is
  tracked in the handoff notes.
