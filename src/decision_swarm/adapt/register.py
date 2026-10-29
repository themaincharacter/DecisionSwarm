"""Register a trained adapter in the model registry — versioned, never overwritten.

Registration is local-first (an opt-in Hub push is a later flag).  It refuses to
register an adapter whose evaluate report did not pass the gates: a
specialization that regressed the must-not-regress set is not registered even
when X improved (Q5).  The manifest is never rewritten; ``--patch`` is the
explicit opt-in that swaps one cell onto the adapter.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from decision_swarm.adapt import AdaptError
from decision_swarm.registry import ModelEntry, merged_ref

_NAME_RE = re.compile(r"^[a-z0-9_-]+$")


def versioned_id(base_id: str, name: str | None, existing: set[str]) -> str:
    """A registry id that never overwrites: ``<base>-<name>``, then ``-2``, ``-3``..."""
    stem = f"{base_id}-{name}" if name else f"{base_id}-adapter"
    if not _NAME_RE.match(stem):
        raise AdaptError(f"adapter name {name!r} must be a slug (a-z0-9_-)")
    candidate = stem
    counter = 2
    while candidate in existing:
        candidate = f"{stem}-{counter}"
        counter += 1
    return candidate


def register_adapter(
    adapter_dir: str | Path,
    *,
    report: dict[str, Any],
    registry_path: str | Path,
    name: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Append a versioned ``kind: adapter`` entry; return ``(entry, cell snippet)``."""
    from decision_swarm import registry

    adapter_dir = Path(adapter_dir).resolve()
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise AdaptError(f"adapter_config.json not found in {adapter_dir}")
    if not (adapter_dir / "adapter.safetensors").is_file():
        raise AdaptError(f"adapter.safetensors not found in {adapter_dir}")
    if not isinstance(report, dict) or report.get("pass") is not True:
        raise AdaptError("the evaluate gate has not passed (or its report is missing); refusing to register")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = config.get("decision_swarm") or {}
    base_block = provenance.get("base") or {}
    base_id = base_block.get("registry_id")
    if not base_id:
        raise AdaptError("adapter_config.json carries no decision_swarm.base.registry_id")
    base_entry = registry.get_entry(base_id)
    digest = registry.adapter_digest(base_entry, adapter_dir)
    merged = {"hf_repo": f"{registry.MERGED_PREFIX}{name or (base_id + '-adapter')}", "revision": digest}

    registry_path = Path(registry_path)
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdaptError(f"{registry_path}: cannot read registry ({exc})") from exc
    models = raw.get("models")
    if not isinstance(models, list):
        raise AdaptError(f"{registry_path}: 'models' must be a list")
    existing = {item.get("id") for item in models if isinstance(item, dict)}
    entry_id = versioned_id(base_id, name, existing)
    merged["hf_repo"] = f"{registry.MERGED_PREFIX}{entry_id}"

    questions_file = provenance.get("questions")
    entry = {
        "id": entry_id,
        "kind": "adapter",
        "base": base_id,
        "path": str(adapter_dir),
        "merged": merged,
        "modality": "text",
        "package_source": base_entry.package_source,
        "loader": "load",
        "license": base_entry.license,
        "notes": (
            f"specialization: {report.get('metric_map') or 'unspecified'}; "
            f"trained {date.today().isoformat()}; evaluate gate passed"
        ),
    }
    models.append(entry)
    registry_path.write_text(json.dumps({"models": models}, indent=2) + "\n", encoding="utf-8")

    snippet = cell_snippet(entry, questions_file=questions_file, adapter_dir=adapter_dir)
    return entry, snippet


def cell_snippet(
    entry: dict[str, Any],
    *,
    questions_file: str | None = None,
    adapter_dir: str | Path,
) -> str:
    """A copy-paste manifest cell: model + adapter + the trained questions as defaults."""
    questions: dict[str, Any] = {}
    if questions_file:
        candidate = Path(adapter_dir) / questions_file
        if candidate.is_file():
            questions = json.loads(candidate.read_text(encoding="utf-8"))
    cell: dict[str, Any] = {
        "id": "<cell-id>",
        "model": entry["base"],
        "adapter": f"{entry['merged']['hf_repo']}@{entry['merged']['revision']}",
        "modality": "text",
        "views": ["<view>"],
    }
    if questions:
        cell["default_questions"] = questions
    return json.dumps(cell, indent=2)


def patch_manifest(manifest_path: str | Path, cell_id: str, entry: dict[str, Any]) -> None:
    """Opt-in rewrite: swap one cell onto the registered adapter (model + adapter)."""
    import dataclasses

    from decision_swarm import manifest

    swarm = manifest.load_manifest(manifest_path)
    cells = []
    found = False
    for cell in swarm.cells:
        if cell.id == cell_id:
            found = True
            cell = dataclasses.replace(
                cell,
                model=entry["base"],
                adapter=f"{entry['merged']['hf_repo']}@{entry['merged']['revision']}",
            )
        cells.append(cell)
    if not found:
        raise AdaptError(f"no cell {cell_id!r} in {manifest_path}")
    manifest.save_manifest(dataclasses.replace(swarm, cells=tuple(cells)), manifest_path)
