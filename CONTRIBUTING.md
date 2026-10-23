# Contributing to Decision Swarm

Thanks for helping out. This project is early (see the roadmap in `README.md`); keeping the
skeleton green is the baseline.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
decision-swarm --help
```

## Project structure

- `src/decision_swarm/` — the Python package. One module per concern (see `README.md`).
- `registry/models.json` — known checkpoints. New entries need: `id`, `hf_repo`,
  `revision`, `modality`, `package_source`, `loader`, `params`, `download_bytes`,
  `license`, `notes`. **Always fill `license` accurately** — the GUI shows it to users, and
  some weights (e.g. `thaitea/laya-vision`) are non-commercial.
- `tests/` — pytest suite. CI (`.github/workflows/test.yml`) runs it on Linux, macOS and
  Windows, Python 3.10 and 3.12. Keep tests portable: no shell-out, use `pathlib`.

## Ground rules

- **Tests must pass on all CI platforms.** Run `pytest` before opening a PR.
- **Typed questions are the public contract.** Question `type` is one of `choice`, `score`,
  `noul`; don't invent new types without a plan update.
- **Manifest changes** bump `schema_version` in `manifest.py` and `swarm.json` examples.
- **No aggregation in v1.** Consensus/aggregation work is explicitly deferred; per-cell
  worker topology should stay intact so it can be added later without rearchitecture.
- **Model facts** (sizes, licenses, loader names) belong in `registry/models.json`, not
  hard-coded in the GUI or runtime.

## Commits and PRs

- Small, focused commits with imperative subjects ("Add cache fetch/verify", not "fixes").
- PR description should say which phase/section of the plan it advances and how the
  "done when" criteria were verified.
- New dependencies: justify them in the PR; runtime deps must not collide with the
  per-package env story (`envs.py`).

## Reporting issues

Include: OS, Python version, `decision-swarm --version`, the pack `swarm.json` (if
relevant), and whether the model cache was warm or cold.
