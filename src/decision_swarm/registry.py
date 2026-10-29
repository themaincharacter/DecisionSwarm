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

BUNDLED_RELATIVE_PATH = ("data", "registry", "models.json")
"""Registry copy bundled into wheels and frozen executables."""

REGISTRY_ENV = "DECISION_SWARM_REGISTRY"
"""Environment variable overriding the registry file location."""

_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")
MERGED_PREFIX = "local/"
"""Synthetic repo-id prefix for materialized (base, adapter) merged checkpoints."""

ENTRY_KINDS = ("checkpoint", "adapter")
"""Registry entry kinds: a checkpoint to serve, or a trained adapter to attach."""

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
    "kind",
    "base",
    "path",
    "merged",
)


class RegistryError(ValueError):
    """Raised when the registry cannot be read or a model cannot be resolved."""


@dataclass(frozen=True)
class ModelEntry:
    """One registry entry: a checkpoint, a trained adapter, or the custom placeholder."""

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
    kind: str = "checkpoint"
    base: str | None = None
    path: str | None = None
    merged: Mapping[str, Any] | None = None

    @property
    def display_license(self) -> str:
        """License text for UI surfaces; custom entries have no known license."""
        return self.license or "unknown (user-supplied)"


def default_registry_path() -> Path:
    """Resolve the registry file: ``$DECISION_SWARM_REGISTRY``, bundled copy, repo checkout."""
    env_path = os.environ.get(REGISTRY_ENV)
    if env_path:
        return Path(env_path)
    module_dir = Path(__file__).resolve().parent
    bundled = module_dir.joinpath(*BUNDLED_RELATIVE_PATH)
    if bundled.is_file():
        return bundled
    for parent in module_dir.parents:
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
    kind = fields.get("kind") or "checkpoint"
    if kind not in ENTRY_KINDS:
        raise RegistryError(f"registry entry {model_id!r}: kind must be one of {ENTRY_KINDS} (got {kind!r})")
    fields["kind"] = kind
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


def merged_ref(entry: ModelEntry) -> str:
    """The portable ref string for a merged checkpoint: ``local/<name>@<digest>``."""
    return f"{entry.hf_repo}@{entry.revision}"


def merged_entry(name: str, digest: str, base: ModelEntry) -> ModelEntry:
    """The synthetic cache entry for one (base, adapter) pair (Q8).

    ``local/<name>`` as the repo id and the content digest as the revision, so
    cache/export/bundle treat a materialized merged checkpoint like any other
    snapshot.
    """
    return ModelEntry(
        id=f"{MERGED_PREFIX}{name}@{digest}",
        hf_repo=f"{MERGED_PREFIX}{name}",
        revision=digest,
        modality=base.modality,
        package_source=base.package_source,
        loader=base.loader,
    )


def adapter_digest(base: ModelEntry, adapter_path: str | Path) -> str:
    """Content digest of one (base, adapter) pair — stable across machines."""
    import hashlib

    weights = Path(adapter_path) / "adapter.safetensors"
    if not weights.is_file():
        raise RegistryError(f"adapter weights not found: {weights}")
    payload = f"{base.hf_repo}@{base.revision or 'main'}#" + hashlib.sha256(weights.read_bytes()).hexdigest()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def resolve_adapter(
    ref: str,
    *,
    base: ModelEntry,
    registry: Mapping[str, ModelEntry] | None = None,
) -> ModelEntry:
    """Resolve a ``cell.adapter`` value to the merged checkpoint entry.

    Accepts the portable ``local/<name>@<digest>`` ref (self-contained) or a
    registered adapter id (``kind: adapter``; its ``merged`` block or its files
    supply the digest).
    """
    if ref.startswith(MERGED_PREFIX) and "@" in ref:
        repo, _, digest = ref.partition("@")
        if not digest:
            raise RegistryError(f"bad adapter ref {ref!r}")
        return merged_entry(repo[len(MERGED_PREFIX):], digest, base)
    entry = get_entry(ref, registry)
    if entry.kind != "adapter":
        raise RegistryError(f"registry entry {ref!r} is not an adapter (kind {entry.kind!r})")
    merged = entry.merged or {}
    repo, digest = merged.get("hf_repo"), merged.get("revision")
    if repo and digest:
        return merged_entry(str(repo)[len(MERGED_PREFIX):], str(digest), base)
    if not entry.path:
        raise RegistryError(f"adapter {ref!r} has neither a 'merged' block nor a 'path'")
    return merged_entry(entry.id, adapter_digest(base, entry.path), base)


def adapter_entry_for(ref: str, registry: Mapping[str, ModelEntry] | None = None) -> ModelEntry | None:
    """The registry adapter entry behind a ref, when there is one (for materialization)."""
    if ref.startswith(MERGED_PREFIX):
        for entry in (registry or load_registry()).values():
            merged = entry.merged or {}
            if entry.kind == "adapter" and merged_ref_from_parts(merged) == ref:
                return entry
        return None
    entry = get_entry(ref, registry)
    return entry if entry.kind == "adapter" else None


def merged_ref_from_parts(merged: Mapping[str, Any] | None) -> str | None:
    if not merged or not merged.get("hf_repo") or not merged.get("revision"):
        return None
    return f"{merged['hf_repo']}@{merged['revision']}"


def resolve_checkpoint(
    cell: "CellSpec", registry: Mapping[str, ModelEntry] | None = None
) -> ModelEntry:
    """The checkpoint a cell actually runs: merged (base + adapter) or the base.

    A cell with an adapter runs the materialized merged checkpoint (serving
    needs no loader changes); any other cell runs its model entry as resolved.
    """
    base = resolve_model(cell.model, registry)
    if not getattr(cell, "adapter", None):
        return base
    return resolve_adapter(cell.adapter, base=base, registry=registry)
