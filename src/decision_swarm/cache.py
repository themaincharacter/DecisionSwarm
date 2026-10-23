"""Shared model cache.

Checkpoints are downloaded once into a local cache (default
``~/.decision-swarm/models``; override with ``--models-dir`` or
``DECISION_SWARM_MODELS``) and content-addressed by repo + revision: one copy
per unique checkpoint no matter how many cells use it, verified against
``models.lock.json`` instead of re-fetched, and offline once warm. The CLI and
GUI read the same cache path.

A checkpoint's files live in ``<models-dir>/<hf_repo>/<revision>/``. Within
that snapshot, each downloaded content set (the whole repo, or one subfolder
like ``multilingual/``) is recorded with per-file sha256 digests in a marker
file, so fetches are idempotent: a second fetch of the same models does no
network I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_swarm.registry import ModelEntry

MODELS_DIR_ENV = "DECISION_SWARM_MODELS"
"""Environment variable overriding the cache location."""

DEFAULT_MODELS_DIR_NAME = ".decision-swarm/models"
"""Cache location under the user's home directory by default."""

CACHE_MARKER_NAME = ".decision-swarm-cache.json"
"""Per-snapshot marker recording fetched content sets and their digests."""

ROOT_SET = ""
"""Content-set key for the repository root (no subfolder)."""

Downloader = Callable[[str, str, Path, "str | None"], None]
"""Downloads ``repo_id`` at ``revision`` into ``target`` (optionally one subfolder)."""


class CacheError(RuntimeError):
    """Raised when a fetch cannot proceed (e.g. offline with a cold cache)."""


@dataclass(frozen=True)
class FetchResult:
    """Outcome of :func:`fetch` — ``downloaded`` is False on a cache hit."""

    entry: ModelEntry
    path: Path
    downloaded: bool
    bytes: int
    sha256: Mapping[str, str]  # relative path -> digest of each fetched file


@dataclass(frozen=True)
class CacheStatus:
    """Cache state of one entry, for ``inspect`` and GUI surfaces."""

    entry: ModelEntry
    cached: bool
    path: Path
    bytes: int


def models_dir(override: str | Path | None = None) -> Path:
    """Cache root: explicit override > ``$DECISION_SWARM_MODELS`` > ``~/.decision-swarm/models``."""
    if override is not None:
        return Path(override)
    env = os.environ.get(MODELS_DIR_ENV)
    if env:
        return Path(env)
    return Path.home() / DEFAULT_MODELS_DIR_NAME


def snapshot_dir(entry: ModelEntry, root: str | Path | None = None) -> Path:
    """Cache location for an entry's repo + revision (shared by subfolder variants)."""
    if not entry.hf_repo or not entry.revision:
        raise CacheError(f"model {entry.id!r} has no hf_repo/revision to cache")
    return models_dir(root) / entry.hf_repo / entry.revision


def content_set(entry: ModelEntry) -> str:
    """Content-set key within a snapshot: the repo root or the entry's subfolder."""
    return entry.subfolder or ROOT_SET


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _marker_path(directory: Path) -> Path:
    return directory / CACHE_MARKER_NAME


def _read_marker(directory: Path) -> dict[str, Any]:
    marker_path = _marker_path(directory)
    if not marker_path.is_file():
        return {"sets": {}}
    try:
        data = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"sets": {}}
    if not isinstance(data, dict) or not isinstance(data.get("sets"), dict):
        return {"sets": {}}
    return data


def _write_marker(directory: Path, entry: ModelEntry, sets: Mapping[str, Any]) -> None:
    marker = {"repo": entry.hf_repo, "revision": entry.revision, "sets": dict(sets)}
    _marker_path(directory).write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")


def _hash_files(directory: Path, set_key: str) -> tuple[int, dict[str, str]]:
    base = directory / set_key if set_key else directory
    total = 0
    hashes: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.name == CACHE_MARKER_NAME:
            continue
        relative = path.relative_to(directory).as_posix()
        hashes[relative] = sha256_file(path)
        total += path.stat().st_size
    return total, hashes


def _files_present(directory: Path, record: Mapping[str, Any]) -> bool:
    files = record.get("files")
    if not isinstance(files, dict):
        return False
    return all((directory / relative).is_file() for relative in files)


def _record(entry: ModelEntry, root: str | Path | None) -> Mapping[str, Any] | None:
    marker = _read_marker(snapshot_dir(entry, root))
    record = marker.get("sets", {}).get(content_set(entry))
    return record if isinstance(record, Mapping) else None


def is_cached(entry: ModelEntry, root: str | Path | None = None) -> bool:
    """True when this entry's content set is fully present in the cache."""
    record = _record(entry, root)
    return record is not None and _files_present(snapshot_dir(entry, root), record)


def status(entry: ModelEntry, root: str | Path | None = None) -> CacheStatus:
    """Cache state of one entry (used by ``decision-swarm inspect``)."""
    directory = snapshot_dir(entry, root)
    record = _record(entry, root)
    present = record is not None and _files_present(directory, record)
    return CacheStatus(
        entry=entry,
        cached=present,
        path=directory,
        bytes=int(record.get("bytes", 0)) if record else 0,
    )


def verify(entry: ModelEntry, root: str | Path | None = None) -> bool:
    """Recompute sha256 digests for a cached content set and compare to the marker."""
    directory = snapshot_dir(entry, root)
    record = _record(entry, root)
    if record is None:
        return False
    for relative, expected in dict(record.get("files", {})).items():
        path = directory / relative
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def _hub_download(repo_id: str, revision: str, target: Path, subfolder: str | None) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(target),
        allow_patterns=[f"{subfolder}/**"] if subfolder else None,
    )


def fetch(
    entry: ModelEntry,
    *,
    root: str | Path | None = None,
    offline: bool = False,
    downloader: Downloader | None = None,
) -> FetchResult:
    """Fetch an entry's content set into the cache (idempotent).

    Cache hits return without touching the network. With ``offline`` set, a
    cold cache raises :class:`CacheError` instead of downloading. ``downloader``
    is injectable for tests.
    """
    directory = snapshot_dir(entry, root)
    set_key = content_set(entry)
    record = _record(entry, root)
    if record is not None and _files_present(directory, record):
        return FetchResult(
            entry=entry,
            path=directory,
            downloaded=False,
            bytes=int(record.get("bytes", 0)),
            sha256=dict(record.get("files", {})),
        )
    if offline:
        raise CacheError(
            f"{entry.id!r} is not in the cache at {directory} and offline was requested; "
            "run 'decision-swarm fetch' while online first"
        )

    directory.mkdir(parents=True, exist_ok=True)
    download = downloader if downloader is not None else _hub_download
    download(entry.hf_repo or "", entry.revision or "", directory, entry.subfolder)

    total, hashes = _hash_files(directory, set_key)
    marker = _read_marker(directory)
    marker["sets"][set_key] = {"bytes": total, "files": hashes}
    _write_marker(directory, entry, marker["sets"])
    return FetchResult(entry=entry, path=directory, downloaded=True, bytes=total, sha256=hashes)
