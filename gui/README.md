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
├── scripts/
│   └── sidecar.mjs   # copies the engine launcher into src-tauri/binaries/
└── src-tauri/        # thin Rust shell: list_models / write_manifest /
                      # fetch_models / build_pack commands
    └── bundle.release.json  # release-only config: bundle > externalBin
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

## Release packaging

`npm run build` bundles per-OS installers with the engine inside — the end user
needs no Python and no git checkout.

```bash
# 1. build the engine launcher (repo root)
python -m pip install pyinstaller
python -m pyinstaller --clean --noconfirm decision-swarm.spec

# 2. bundle (gui/)
npm install
npm run build        # copies dist/decision-swarm into src-tauri/binaries/ first
```

Installers land in `src-tauri/target/release/bundle/`: `.deb` / `.rpm` /
`.AppImage` (Linux), `.dmg` (macOS), `-setup.exe` / `.msi` (Windows).

How it fits together:

- `scripts/sidecar.mjs` copies the launcher to
  `src-tauri/binaries/decision-swarm-<target-triple>` (run automatically by
  `prebuild`; set `DECISION_SWARM_SIDECAR` to pick a different launcher).
- `src-tauri/bundle.release.json` adds `bundle > externalBin` for release
  builds only, so `cargo build` / `npm run dev` keep working without a built
  launcher — while `npm run build` always bundles one and fails without it.
- `.github/workflows/release-gui.yml` builds Windows (x64, arm64), macOS
  (arm64, Intel) and Linux (x86_64, arm64) installers on every `v*` tag and
  attaches them to the GitHub release, alongside the `release-cli.yml` bins.
- At runtime the shell looks for the launcher beside the app executable first
  (then `Contents/Resources` on macOS, then PATH), so `models`, `fetch` and
  `build` all work from the installed app.

Signing: unsigned bundles run, but trigger macOS Gatekeeper / Windows
SmartScreen warnings. For signed and notarized macOS builds set the repository
secrets `APPLE_CERTIFICATE` (base64 `.p12`), `APPLE_CERTIFICATE_PASSWORD`,
`APPLE_SIGNING_IDENTITY`, and for notarization `APPLE_ID`, `APPLE_PASSWORD`,
`APPLE_TEAM_ID`. Windows installers stay unsigned unless you configure
`bundle > windows > signCommand`.

## Notes

- The web UI never assumes Tauri: `bridge.js` falls back to a deterministic
  mock so every screen is exercisable in a browser.
- The model dropdown, cluster metrics and Submit all read the same registry
  (`registry/models.json`) as the CLI — `decision-swarm models --json`.
