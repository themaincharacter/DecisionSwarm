# Decision Swarm GUI

Tauri v2 desktop app: a thin Rust shell around a vanilla-JS web UI. **All pack
logic lives in the Python `decision-swarm` package** — the shell invokes it as a
CLI with JSON output (`models --json`, `build --json`, `fetch --manifest`), so
the GUI and the CLI produce byte-identical packs.

## Layout

```
gui/
├── src/              # web UI (no build step; Tauri serves it as frontendDist)
│   ├── index.html
│   ├── app.js
│   ├── bridge.js     # Tauri IPC vs mock bridge (browser dev mode)
│   └── style.css
└── src-tauri/        # thin Rust shell: list_models / write_manifest /
                      # fetch_models / build_pack commands
```

## Develop

```bash
# engine (from the repo root)
python3 -m venv .venv && .venv/bin/python -m pip install -e ".[dev]"

# UI in a plain browser (mock bridge, no Tauri needed)
cd gui/src && python3 -m http.server 8123   # open http://localhost:8123

# full desktop app
cd gui && npm install && npm run dev        # needs Rust + webkit2gtk (Linux)
```

Linux build dependencies: `sudo apt install pkgconf libwebkit2gtk-4.1-dev
build-essential libxdo-dev libssl-dev librsvg2-dev`.

The shell finds the engine automatically: `DECISION_SWARM_CLI` if set, a
launcher bundled beside the app, then a repo checkout's `.venv` (or `dist`)
launcher found by walking up from the executable/cwd, then `decision-swarm` on
PATH.

## Notes

- The web UI never assumes Tauri: `bridge.js` falls back to a deterministic
  mock so every screen is exercisable in a browser.
- The model dropdown, cluster metrics and Submit all read the same registry
  (`registry/models.json`) as the CLI — `decision-swarm models --json`.
