"""Model registry access.

``registry/models.json`` lists the known checkpoints (the Laya family, the
``thaitea/laya-vision`` checkpoints) plus a free-form entry for any Hugging
Face repo id. Entries carry the loader name, ``package_source`` for per-package
envs, size estimates and the weights license (shown to users before download).

Cells reference models by registry id (``laya-vision``), by Hugging Face repo
id (``thaitea/laya-vision`` resolves to the entry owning that repo root) or by
an arbitrary ``org/name`` repo id (resolved to a free-form custom entry).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from decision_swarm.manifest import CellSpec

REGISTRY_RELATIVE_PATH = ("registry", "models.json")
"""Location of the registry file inside the repository."""

REGISTRY_ENV = "DECISION_SWARM_REGISTRY"
"""Environment variable overriding the registry file location."""

_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENTRY_FIELDS = (
    "hf_repo",
    "subfolder",
    "revision",
    "modality",
    "package_source",
    "loader",
    "params",
    "license",
    "notes",
)


class RegistryError(ValueError):
    """Raised when the registry cannot be read or a model cannot be resolved."""


@dataclass(frozen=True)
class ModelEntry:
    """One registry entry: a checkpoint, or the free-form custom placeholder."""

    id: str
    hf_repo: str | None = None
    subfolder: str | None = None
    revision: str | None = None
    modality: str | None = None
    package_source: str | None = None
    loader: str | None = None
    params: str | None = None
    download_bytes: int | None = None
    license: str | None = None
    notes: str | None = None
    custom: bool = False

    @property
    def display_license(self) -> str:
        """License text for UI surfaces; custom entries have no known license."""
        return self.license or "unknown (user-supplied)"


def default_registry_path() -> Path:
    """Resolve the registry file: ``$DECISION_SWARM_REGISTRY``, then the repo checkout."""
    env_path = os.environ.get(REGISTRY_ENV)
    if env_path:
        return Path(env_path)
    for parent in Path(__file__).resolve().parents:
        candidate = parent.joinpath(*REGISTRY_RELATIVE_PATH)
        if candidate.is_file():
            return candidate
    raise RegistryError(
        f"{Path(*REGISTRY_RELATIVE_PATH)} not found near {__file__}; set {REGISTRY_ENV}"
    )


def _entry_from_dict(raw: Mapping[str, Any]) -> ModelEntry:
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise RegistryError(f"registry entry is missing an 'id': {raw!r}")
    fields = {name: raw.get(name) for name in _ENTRY_FIELDS}
    return ModelEntry(
        id=model_id,
        download_bytes=raw.get("download_bytes"),
        custom=bool(raw.get("custom", False)),
        **fields,
    )


def load_registry(path: str | Path | None = None) -> dict[str, ModelEntry]:
    """Load the model registry as a mapping of id -> entry."""
    registry_path = Path(path) if path is not None else default_registry_path()
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"{registry_path}: cannot read registry ({exc})") from exc
    raw_models = data.get("models") if isinstance(data, Mapping) else None
    if not isinstance(raw_models, list):
        raise RegistryError(f"{registry_path}: 'models' must be a list")

    registry: dict[str, ModelEntry] = {}
    for raw in raw_models:
        if not isinstance(raw, Mapping):
            raise RegistryError(f"{registry_path}: entry must be an object (got {raw!r})")
        entry = _entry_from_dict(raw)
        if entry.id in registry:
            raise RegistryError(f"{registry_path}: duplicate registry id {entry.id!r}")
        registry[entry.id] = entry
    return registry


def get_entry(model: str, registry: Mapping[str, ModelEntry] | None = None) -> ModelEntry:
    """Look up a registry entry strictly by id."""
    entries = registry if registry is not None else load_registry()
    if model in entries:
        return entries[model]
    raise RegistryError(f"unknown registry id {model!r} (known: {', '.join(sorted(entries))})")


def resolve_model(model: str, registry: Mapping[str, ModelEntry] | None = None) -> ModelEntry:
    """Resolve a cell's ``model`` field to a registry entry.

    Resolution order: registry id -> repo id owned by a registry entry (the
    entry with no subfolder) -> free-form custom entry for any other
    ``org/name`` repo id.
    """
    entries = registry if registry is not None else load_registry()
    if model in entries:
        return entries[model]
    roots = [
        entry
        for entry in entries.values()
        if not entry.custom and entry.hf_repo == model and entry.subfolder is None
    ]
    if len(roots) == 1:
        return roots[0]
    if _HF_REPO_RE.match(model):
        return ModelEntry(id=model, hf_repo=model, revision="main", custom=True)
    raise RegistryError(f"unknown model {model!r}: not a registry id or Hugging Face repo id")


def resolve_runtime(
    cell: "CellSpec", registry: Mapping[str, ModelEntry] | None = None
) -> tuple[str | None, str | None]:
    """Effective ``(package_source, loader)`` for a cell.

    Explicit ``cell.runtime`` pins win over the resolved registry entry.
    """
    entry = resolve_model(cell.model, registry)
    runtime = cell.runtime
    package_source = runtime.package_source if runtime is not None else entry.package_source
    loader = runtime.loader if runtime is not None else entry.loader
    return package_source, loader
