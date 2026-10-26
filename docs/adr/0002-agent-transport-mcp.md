# 0002. Agents reach a pack through a tools-only MCP server on stdio

Date: 2026-10-26
Status: Accepted

## Context

ADR-0001 gave programs a transport: a warm swarm behind JSONL on stdio. An agent
cannot use it. It has nowhere to keep a pipe between turns, it cannot learn a
pack's views, cells and questions without pulling a README into its context, and
it does not own the process lifecycle it would need to keep models warm.

Shapes considered:

- **(a) CLI plus instructions** — `decision-swarm run --input obs.json` documented
  in a skill/AGENTS.md. Works in every host, costs almost nothing, but discovery
  is context-expensive and every call reloads the models.
- **(b) MCP server over stdio, spawned by the agent host** — the host keeps the
  process alive for the session (the same lifetime `serve` already assumes), tool
  schemas carry the pack's shape, and it works in opencode, Cursor, Claude Code
  and VS Code without per-host integration.
- **(c) HTTP** — a second process to supervise and a port to allocate for a client
  that is already local.
- **(d) Generated Python runtime only** — already exists, but it is a Python API,
  not an agent surface.

(b) is the only option that is both language-agnostic and host-native.

Costs considered: the official `mcp` SDK pulls pydantic, anyio and httpx into an
engine that otherwise depends only on `huggingface_hub`, and the engine ships as a
PyInstaller binary *and* a Tauri sidecar — both of which cannot `pip install` an
optional extra. Tool-only MCP is a small, stable slice of the protocol
(`initialize`, `ping`, `tools/list`, `tools/call`), so it is implemented by hand
and checked against the real SDK by a skippable conformance test.

## Decisions

1. **`decision-swarm mcp` speaks MCP over stdio, tools only.** No resources,
   prompts or sampling. The server is scoped by `--packs-dir` (repeatable) with an
   optional `--pack` default, so one config entry can judge any pack in a
   directory — and can build one it then judges. `<pack-slug> mcp` from a generated
   runtime forwards to the same command for its own pack.

2. **Judging is two tools, split by cost.** `describe` returns views, cells,
   questions and criteria without loading a model, so schema exploration is free.
   `judge` takes one observation and returns per-cell answers, loading the models on
   its first call for a pack. With no `pack` argument, `describe` lists the packs.

3. **Results project, they do not aggregate.** `judge` defaults to
   `detail: "compact"` (cell → question → `{value, confidence}`) because an agent
   pays for every token of a tool result; `full` keeps the model, latency and
   warnings. v1 still returns raw per-cell answers — no consensus.

4. **One warm swarm at a time** (`--max-warm`, default 1). A swarm is gigabytes of
   weights across 2–8 worker processes; switching packs evicts the oldest.

5. **Authoring is exposed through the same server, behind a registry-only trust
   gate.** `list_models`, `build`, `fetch`, `install`, `export` and `job_status`.
   A manifest is LLM-authored data, and `envs.build_env` executes a cell's
   `package_source` with pip — so unless the server runs with `--allow-unvetted`,
   every model id and package source must resolve in the model registry. `install`
   re-validates the pack it was handed, because a pack directory can be edited
   after it was built.

6. **`build` emits `mcp.json`.** A ready-to-paste client config with absolute
   paths, so a freshly built pack is one config copy away from being callable. The
   absolute paths go stale if the pack moves, so `mcp --print-config --pack DIR`
   regenerates the snippet.

7. **Hand-rolled JSON-RPC, no runtime dependency.** The wire is ~120 lines; the
   `conformance` extra installs the official SDK so
   `tests/test_mcp_conformance.py` can check the handshake and a tool call against
   it. stdout is the transport, so `serve()` redirects `sys.stdout` to stderr for
   the session: a stray `print`, a pip run or a chatty download can never corrupt
   the stream.

## Consequences

- Programs keep ADR-0001's JSONL transport untouched; agents get a tool surface on
  top of the same `answer()` seam. HTTP remains a third adapter, not a rewrite.
- The trust gate is the *only* thing between an LLM-authored manifest and an
  arbitrary pip source. It is a flag, not a sandbox: `--allow-unvetted` is for an
  operator who knows what the manifest says.
- Image views stay file paths, which is correct for a local stdio server and wrong
  for a remote one. MCP image content blocks are a later addition.
- Long operations do not fit a tool call: `export` is a job pair, `fetch`/`install`
  are idempotent and resumable, and job records live in memory, so a restarted
  server loses them (`job_status` says so and you re-run).
