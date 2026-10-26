"""Pack builder — what the GUI's Submit emits.

A swarm pack directory contains ``swarm.json`` (manifest), ``models.lock.json``
(repo + revision + commit + sha256 per unique checkpoint), a generated
``runtime/`` package and a README with usage for the pack. Templates for the
generated runtime live in ``templates/runtime/``.

The lockfile *pins* the models a pack expects but never contains their weights:
a pack ships as a description, and weights resolve from the shared cache.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from decision_swarm import cache, registry
from decision_swarm.manifest import MANIFEST_NAME, SwarmManifest, save_manifest

PACK_LOCKFILE_NAME = "models.lock.json"
"""Model lockfile name inside a pack."""

PACK_RUNTIME_DIR = "runtime"
"""Generated Python package directory inside a pack."""

PACK_README_NAME = "README.md"
"""Generated usage README inside a pack."""

LOCKFILE_SCHEMA_VERSION = 1
"""Current ``models.lock.json`` schema version."""

TEMPLATES_RELATIVE_PATH = ("templates", "runtime")
"""Location of the runtime templates inside the repository."""

TEMPLATES_BUNDLED_PATH = ("data", "templates", "runtime")
"""Template copy bundled into wheels and frozen executables."""

CommitResolver = Callable[[registry.ModelEntry], "str | None"]
"""Resolves a registry entry's floating revision (e.g. ``main``) to a commit sha."""


class PackError(RuntimeError):
    """Raised when a pack cannot be built."""


@dataclass(frozen=True)
class PackModel:
    """One unique checkpoint recorded in ``models.lock.json``."""

    id: str
    hf_repo: str
    subfolder: str | None
    revision: str
    cached: bool
    bytes: int
    sha256: Mapping[str, str]
    params: str | None = None
    download_bytes: int | None = None
    commit: str | None = None


def hub_commit_resolver(entry: registry.ModelEntry) -> str | None:
    """Ask the Hub which commit ``entry.revision`` points at right now.

    Never fails a build: offline, missing repos and auth problems all resolve to
    ``None``, which the lockfile records as "unpinned".
    """
    if not entry.hf_repo:
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(entry.hf_repo, revision=entry.revision or "main")
    except Exception:  # noqa: BLE001 - a cold/offline build must still succeed
        return None
    sha = getattr(info, "sha", None)
    return sha if isinstance(sha, str) and sha else None



def pack_slug(name: str) -> str:
    """Kebab-case project/script name for a pack (from the manifest name)."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "swarm-pack"


def pack_module(name: str) -> str:
    """Snake_case module name for a pack's runtime package."""
    return pack_slug(name).replace("-", "_")


def default_templates_dir() -> Path:
    """Locate the runtime templates: bundled copy, then the repository checkout."""
    module_dir = Path(__file__).resolve().parent
    bundled = module_dir.joinpath(*TEMPLATES_BUNDLED_PATH)
    if bundled.is_dir():
        return bundled
    for parent in module_dir.parents:
        candidate = parent.joinpath(*TEMPLATES_RELATIVE_PATH)
        if candidate.is_dir():
            return candidate
    raise PackError(f"{Path(*TEMPLATES_RELATIVE_PATH)} not found near {__file__}")


def _render(text: str, mapping: Mapping[str, str]) -> str:
    for key, value in mapping.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def unique_entries(
    manifest: SwarmManifest,
    *,
    registry_entries: Mapping[str, registry.ModelEntry] | None = None,
) -> list[registry.ModelEntry]:
    """Unique model entries used by the manifest's cells, in cell order."""
    seen: set[tuple[str, str, str | None]] = set()
    entries: list[registry.ModelEntry] = []
    for cell in manifest.cells:
        entry = registry.resolve_model(cell.model, registry_entries)
        key = (entry.hf_repo or "", entry.revision or "", entry.subfolder)
        if key in seen:
            continue
        seen.add(key)
        entries.append(entry)
    return entries


def collect_models(
    manifest: SwarmManifest,
    *,
    models_root: str | Path | None = None,
    registry_entries: Mapping[str, registry.ModelEntry] | None = None,
    resolve_commit: CommitResolver | None = None,
) -> list[PackModel]:
    """One lock record per unique checkpoint (repo + revision + subfolder).

    Digests come from the shared cache (read-only: nothing is downloaded);
    uncached checkpoints are recorded with ``cached: False`` and no digests.
    With ``resolve_commit`` set, the floating revision is resolved to the commit
    it points at, so the lockfile pins something reproducible.
    """
    models: list[PackModel] = []
    for entry in unique_entries(manifest, registry_entries=registry_entries):
        cached = True
        try:
            result = cache.fetch(entry, root=models_root, offline=True)
            size, digests = result.bytes, dict(result.sha256)
        except cache.CacheError:
            cached, size, digests = False, 0, {}
        commit = resolve_commit(entry) if resolve_commit is not None else None
        models.append(
            PackModel(
                id=entry.id,
                hf_repo=entry.hf_repo or "",
                subfolder=entry.subfolder,
                revision=entry.revision or "",
                cached=cached,
                bytes=size,
                sha256=digests,
                params=entry.params,
                download_bytes=entry.download_bytes,
                commit=commit,
            )
        )
    return models


SERVE_EXAMPLE_READY = '{"type": "ready", "name": "<pack>", "cells": [...], "views": [...], "warnings": []}'
SERVE_EXAMPLE_QUERY = '{"type": "query", "id": 1, "views": {...}, "questions": {...}}'
SERVE_EXAMPLE_RESULT = '{"type": "result", "id": 1, "cells": {...}}'
SERVE_EXAMPLE_ERROR = '{"type": "error", "id": 1, "message": "..."}'
SERVE_EXAMPLE_SHUTDOWN = '{"type": "shutdown"}'


def _pack_readme(manifest: SwarmManifest, models: list[PackModel], slug: str) -> str:
    lines = [
        f"# {manifest.name} (swarm pack)",
        "",
        "Generated by Decision Swarm. This directory is a runnable artifact: the pack",
        "manifest, the model lockfile, and a generated Python runtime under `runtime/`.",
        "",
        "## Cells",
        "",
        "| id | model | modality | views | default questions |",
        "|---|---|---|---|---|",
    ]
    for cell in manifest.cells:
        questions = ", ".join(question.name for question in cell.default_questions) or "-"
        lines.append(
            f"| `{cell.id}` | `{cell.model}` | {cell.modality} | {', '.join(cell.views)} | {questions} |"
        )
    lines += ["", "## Models", "", "| checkpoint | revision | cached | bytes |", "|---|---|---|---|"]
    for model in models:
        checkpoint = model.hf_repo + (f"/{model.subfolder}" if model.subfolder else "")
        cached = "yes" if model.cached else "no"
        lines.append(f"| `{checkpoint}` | `{model.revision}` | {cached} | {model.bytes} |")
    lines += [
        "",
        "## Usage",
        "",
        "```bash",
        f"pip install ./runtime        # installs the {slug} runtime package",
        f"{slug} inspect              # list the pack's cells",
        f"{slug} run --input obs.json  # one-shot: answer one observation, then exit",
        f"{slug} serve                 # long-lived: warm decision service over stdio",
        "```",
        "",
        "### Long-lived decisions (`serve`)",
        "",
        "`serve` loads every model once, then answers query after query at model",
        "latency — no model reload per decision. One JSON object per line on stdio:",
        "",
        "```jsonl",
        SERVE_EXAMPLE_READY,
        SERVE_EXAMPLE_QUERY,
        SERVE_EXAMPLE_RESULT,
        SERVE_EXAMPLE_ERROR,
        SERVE_EXAMPLE_SHUTDOWN,
        "```",
        "",
        "`ready` arrives once every model is loaded. A `query` line is exactly the",
        "`run --input` payload plus `type` and `id` (`id` is echoed back and may be",
        "omitted), and `result` carries the same `cells` object as `run`'s JSON",
        "output. Per-query failures come back as `error` lines without ending the",
        "session; `shutdown` (or closing stdin) exits cleanly.",
        "",
        "Model weights resolve from the shared cache (`~/.decision-swarm/models` by",
        "default); populate it first with `decision-swarm fetch --pack .`.",
        "",
    ]
    return "\n".join(lines)


def build_pack(
    manifest: SwarmManifest,
    dest: str | Path,
    *,
    models_root: str | Path | None = None,
    registry_entries: Mapping[str, registry.ModelEntry] | None = None,
    templates_dir: str | Path | None = None,
    resolve_commit: CommitResolver | None = None,
    models: list[PackModel] | None = None,
) -> Path:
    """Build a swarm pack directory: swarm.json, models.lock.json, runtime/, README.md.

    Nothing is downloaded: the lockfile reads digests from the shared cache.
    ``resolve_commit`` opts into recording the commit each revision points at;
    pass ``models`` to reuse an already-collected lock record list.
    """
    pack_dir = Path(dest)
    pack_dir.mkdir(parents=True, exist_ok=True)

    save_manifest(manifest, pack_dir / MANIFEST_NAME)

    if models is None:
        models = collect_models(
            manifest,
            models_root=models_root,
            registry_entries=registry_entries,
            resolve_commit=resolve_commit,
        )
    lock = {
        "schema_version": LOCKFILE_SCHEMA_VERSION,
        "models": [
            {
                "id": model.id,
                "hf_repo": model.hf_repo,
                "subfolder": model.subfolder,
                "revision": model.revision,
                "commit": model.commit,
                "params": model.params,
                "download_bytes": model.download_bytes,
                "cached": model.cached,
                "bytes": model.bytes,
                "sha256": dict(model.sha256),
            }
            for model in models
        ],
    }
    (pack_dir / PACK_LOCKFILE_NAME).write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    slug = pack_slug(manifest.name)
    module = pack_module(manifest.name)
    mapping = {"PACK_NAME": manifest.name, "PACK_SLUG": slug, "PACK_MODULE": module}

    templates = Path(templates_dir) if templates_dir is not None else default_templates_dir()
    runtime_dir = pack_dir / PACK_RUNTIME_DIR
    for template_name, relative in (
        ("pyproject.toml.tmpl", "pyproject.toml"),
        ("__init__.py.tmpl", f"src/{module}/__init__.py"),
        ("cli.py.tmpl", f"src/{module}/cli.py"),
    ):
        template_path = templates / template_name
        if not template_path.is_file():
            raise PackError(f"missing template {template_path}")
        target = runtime_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_render(template_path.read_text(encoding="utf-8"), mapping), encoding="utf-8")

    # the runtime package embeds a copy of the manifest so it runs stand-alone
    save_manifest(manifest, runtime_dir / "src" / module / "swarm.json")

    (pack_dir / PACK_README_NAME).write_text(_pack_readme(manifest, models, slug), encoding="utf-8")
    return pack_dir
