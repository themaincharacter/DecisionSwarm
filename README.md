# Decision Swarm

Assemble a **swarm of Laya decision-model sensors** over one observation, download the
required models once into a shared local cache, and package the configured swarm as a
runnable artifact: an installable Python library and a bash-callable CLI.

> **Status: Phases 0–4 done.** Core (manifest, registry, cache, packs), runtime
> (workers, protocol, orchestrator), the CLI + per-OS launcher builds, and the
> desktop GUI are in place. Functionality lands per the [Roadmap](#roadmap).

## Concept

| Term | Meaning |
|---|---|
| **Observation** | The thing being judged: a bundle of named **views** (screenshot, DOM, HTML class names, plain text, …). |
| **Cell** | One sensor: a model + bindings to named views + a question set. |
| **Swarm** | 2–8 cells observing the same observation. |
| **Pack** | The build artifact: `swarm.json` + `models.lock.json` + generated `runtime/` + README. |

Each cell answers typed questions — `choice`, `score`, or `noul` (yes/no probability) —
over the views it is bound to, in a single forward pass with calibrated probabilities.
**v1 returns raw per-cell answers;** aggregation/consensus is deliberately deferred.

### Example: browser control

| Cell | Model | Views | Question |
|---|---|---|---|
| `screenshot_sensor` | `thaitea/laya-vision` | `screenshot` | "Do you see a submit button?" (`noul`) |
| `dom_sensor` | `laya` (English) | `dom` | "Which element is the submit button?" (`choice`) |
| `class_sensor` | `laya-typed-decisions` | `classes` | "Which class label belongs to submit?" (`choice`) |

## Download

The GitHub Releases page carries ready-to-run apps built by CI on every
`v*` tag — Windows (portable `.zip`, just unzip and run), macOS (`.dmg`) and
Linux (`.deb`, `.rpm`, `.AppImage`), for x64 and arm64 — alongside the standalone
`decision-swarm-cli-*` CLI binaries. The GUI bundles the engine, so building a
swarm needs no Python and no git checkout (models download on first use into the
shared cache). See `gui/README.md` for how bundles are built and signed.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
decision-swarm --help
```

## CLI

```text
decision-swarm inspect --pack ./my-swarm            # cells, models, sizes, cache status
decision-swarm fetch   --pack ./my-swarm            # pre-download models to the shared cache
decision-swarm install --pack ./my-swarm            # build per-package envs
decision-swarm run     --pack ./my-swarm --input obs.json [--offline] [--json|--pretty]
```

All four commands are implemented. `install` builds one venv per unique
`package_source` (required in practice: the `laya-vision` fork cannot coexist with PyPI
`laya`); `run` executes warm per-cell workers and prints the result schema (`--pretty` for
humans, `--offline` to refuse uncached models). Binaries are built per-OS with PyInstaller
via `decision-swarm.spec` (small ~20 MB launchers; weights resolve from the shared cache).

## Desktop GUI

`gui/` is a Tauri v2 app: a thin Rust shell around a vanilla-JS UI that drives the same
`decision-swarm` engine through the CLI (`models --json`, `build --json`, `fetch
--manifest`), so GUI and CLI produce identical packs. Define observation views and 2–8
cells (models with per-model licenses, view bindings, typed `choice`/`score`/`noul`
questions), watch the live cluster panel (checkpoints, params, disk/RAM estimates,
workers), Download the models, and Submit — then run the pack from the CLI. The UI also
runs in a plain browser against a mock bridge; see `gui/README.md` for dev setup and
Linux build dependencies.

## Model registry

`registry/models.json` lists the known checkpoints. Models are fetched with
`huggingface_hub.snapshot_download` into a shared cache
(`~/.decision-swarm/models`, override: `--models-dir` / `DECISION_SWARM_MODELS`),
content-addressed by repo + revision — one copy per checkpoint no matter how many cells
use it, and offline operation once warm.

| id | checkpoint | modality | params | weights license |
|---|---|---|---|---|
| `laya-english` | `convaiinnovations/laya` (root) | text | 421M | Apache-2.0 |
| `laya-multilingual` | `convaiinnovations/laya` (`multilingual/`) | text | 322M | Apache-2.0 |
| `laya-typed-decisions` | `convaiinnovations/laya` (`typed-decisions/`) | text | 421M | Apache-2.0 |
| `laya-vision` | `thaitea/laya-vision` | image+text | 201M | **CC BY-NC-SA 4.0** |
| `laya-vision-237m` | `thaitea/laya-vision-smolvlm-256m-score` | image+text | 237M | **CC BY-NC-SA 4.0** |
| `custom-hf-repo` | any Hugging Face repo id | user-supplied | — | user-supplied |

## Repository layout

```text
registry/models.json            # known checkpoints (sizes, loaders, licenses)
pyproject.toml                  # decision-swarm package
src/decision_swarm/
├── manifest.py                 # swarm.json schema + validation
├── registry.py                 # model registry access
├── cache.py                    # shared model cache (fetch/verify/dedup)
├── runtime/
│   ├── orchestrator.py         # spawns cell workers, collects results
│   ├── worker.py               # per-cell process: load model once, answer queries
│   └── protocol.py             # stdio JSON protocol
├── pack.py                     # pack builder (what Submit emits)
├── envs.py                     # per-package venv builder/installer
└── cli.py                      # decision-swarm entrypoint
templates/runtime/              # generated pack runtime + CLI stub
tests/                          # pytest suite (CI runs this)
```

A `swarm.json` pack manifest looks like this (full example lives in the plan; the schema
is validated in `manifest.py`):

```json
{
  "schema_version": 1,
  "name": "browser-swarm",
  "input_schema": {
    "views": {
      "screenshot": {"type": "image"},
      "dom": {"type": "text"},
      "classes": {"type": "text"}
    }
  },
  "cells": [
    {
      "id": "screenshot_sensor",
      "model": "thaitea/laya-vision",
      "modality": "image+text",
      "views": ["screenshot"],
      "default_questions": {
        "submit_visible": {"type": "noul", "instructions": "Do you see a submit button?"}
      },
      "runtime": {"package_source": "git+https://github.com/r33drichards/laya-vision",
                  "loader": "load_vlm"}
    },
    {
      "id": "dom_sensor",
      "model": "convaiinnovations/laya",
      "modality": "text",
      "views": ["dom", "classes"],
      "default_questions": {
        "submit_button": {"type": "choice", "instructions": "Which element is the submit button?",
                          "criteria": {"a": "the input with value Send", "b": "the button labeled Send"}}
      }
    }
  ]
}
```

Cell ids are user-named slugs (`a-z0-9_-`, unique). Questions are baked into the pack as
defaults so answers stay comparable across queries; each query may override them per cell.

## Runtime notes

- **One worker process per cell** — a small JSON protocol over stdio; each worker loads its
  model once and stays warm. Each worker receives only its bound views + resolved questions.
- **Env isolation** — one venv per unique `package_source`. Required in practice: the
  `laya-vision` fork installs its own `laya` package and cannot coexist with PyPI `laya`,
  and it needs `transformers >= 5.3` while PyPI `laya` pins its own stack.

## Roadmap

| Phase | Scope |
|---|---|
| **0 — Scaffold** | repo, package skeleton, CLI surface, registry, CI ✅ |
| **1 — Core** | manifest schema + validation, registry access, cache manager (fetch / sha256 / dedup), pack builder ✅ |
| **2 — Runtime** | worker process + stdio protocol, env isolation, parallel cell execution, result schema ✅ |
| **3 — CLI + exes** | `run` / `fetch` / `install` / `inspect`, offline mode, PyInstaller per-OS builds ✅ |
| **4 — GUI** | Tauri app: cell grid, cluster panel, Download, Submit → pack ✅ |
| Deferred | aggregation/consensus, npm adapter, Godot addon, swarm-as-HTTP-server |

## Licensing

- **This tool:** Apache-2.0.
- **Laya family weights** (`convaiinnovations/laya`): Apache-2.0 — commercial use fine.
- **`thaitea/laya-vision` weights: CC BY-NC-SA 4.0** (non-commercial) — code Apache-2.0.
  The registry and GUI surface this per model; it constrains commercial use of the weights,
  not this tool.

See `CONTRIBUTING.md` for development guidelines.
