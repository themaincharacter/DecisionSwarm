"""Agent-facing tools over a pack.

``runtime/mcp.py`` is a thin wire; this module holds the capabilities an agent
actually calls. Two families:

- **judging**: ``describe`` (the pack's schema, never loads a model) and
  ``judge`` (one observation -> per-cell answers), both routed through
  :func:`decision_swarm.runtime.serve.run_query` so envelope rules, validation
  and error text match ``serve`` exactly.
- **authoring**: ``list_models``, ``build``, ``fetch``, ``install``, ``export``
  and ``job_status``.

Model ids and package sources must resolve in the model registry unless the
server was started with ``allow_unvetted``. This is the trust boundary: a
manifest is LLM-authored data, and ``envs.build_env`` executes a cell's
``package_source`` with pip, so an arbitrary value there is arbitrary code.

Nothing in this module prints. stdout belongs to the transport.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decision_swarm import cache, envs, manifest as manifest_mod, pack, registry
from decision_swarm.manifest import MANIFEST_NAME, SwarmManifest
from decision_swarm.runtime import orchestrator, serve


class AgentError(RuntimeError):
    """A tool failure an agent can read and act on."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"message": str(self)}
        if self.path is not None:
            payload["path"] = self.path
        return payload


@dataclass(frozen=True)
class ToolSpec:
    """One callable tool: JSON Schema for its arguments plus the handler."""

    name: str
    description: str
    schema: Mapping[str, Any]
    handler: Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class AgentConfig:
    """How the server was started."""

    packs_dirs: tuple[Path, ...] = ()
    default_pack: Path | None = None
    allow_unvetted: bool = False
    max_warm: int = 1
    models_root: Path | None = None
    envs_root: Path | None = None


@dataclass
class _Job:
    """One in-flight background operation (only ``export`` for now)."""

    id: str
    kind: str
    state: str = "running"  # running | done | error
    result: dict[str, Any] | None = None
    message: str | None = None
    started: float = field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# pack resolution
# --------------------------------------------------------------------------- #


def discover_packs(config: AgentConfig) -> list[Path]:
    """Every pack directory reachable from ``--packs-dir`` (deduplicated, sorted)."""
    found: dict[Path, None] = {}
    for raw in config.packs_dirs:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        if (root / MANIFEST_NAME).is_file():
            found.setdefault(root.resolve(), None)
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / MANIFEST_NAME).is_file():
                found.setdefault(child.resolve(), None)
    return list(found)


def _not_found_message(spec: str, config: AgentConfig) -> str:
    names = [path.name for path in discover_packs(config)]
    if not names:
        return f"pack {spec!r} not found; no packs under {_dirs_label(config)}"
    seen = ", ".join(names)
    return f"pack {spec!r} not found; packs under {_dirs_label(config)}: {seen}"


def _dirs_label(config: AgentConfig) -> str:
    if not config.packs_dirs:
        return "(no --packs-dir)"
    return ", ".join(str(Path(raw).expanduser()) for raw in config.packs_dirs)


def resolve_pack(spec: str | Path | None, config: AgentConfig) -> Path:
    """Turn a pack argument into an absolute pack directory.

    Accepts a pack directory, a ``swarm.json`` path, or a pack name found under
    ``--packs-dir``. With no argument: the ``--pack`` default, else the single
    pack under ``--packs-dir``.
    """
    if spec is not None:
        text = str(spec)
        candidate = Path(text).expanduser()
        if candidate.is_dir() and (candidate / MANIFEST_NAME).is_file():
            return candidate.resolve()
        if candidate.is_file() and candidate.name == MANIFEST_NAME:
            return candidate.resolve().parent
        if candidate.exists():
            raise AgentError(f"{candidate} is not a pack directory or a {MANIFEST_NAME} file")
        matches = [path for path in discover_packs(config) if path.name == text]
        if len(matches) == 1:
            return matches[0]
        raise AgentError(_not_found_message(text, config))

    if config.default_pack is not None:
        return resolve_pack(config.default_pack, config)

    found = discover_packs(config)
    if len(found) == 1:
        return found[0]
    if not found:
        raise AgentError(
            f"no packs found under {_dirs_label(config)}; build one, or start the server "
            "with --pack / --packs-dir"
        )
    raise AgentError("several packs available; pass 'pack': " + ", ".join(p.name for p in found))


# --------------------------------------------------------------------------- #
# registry trust boundary
# --------------------------------------------------------------------------- #


def registry_problems(
    swarm: SwarmManifest,
    entries: Mapping[str, registry.ModelEntry],
) -> list[dict[str, str]]:
    """Models and package sources that are not in the registry.

    An empty list means the manifest only names vetted things. ``build`` reports
    all of them at once so an agent can fix a manifest in one pass.
    """
    known_sources = {entry.package_source for entry in entries.values() if entry.package_source}
    problems: list[dict[str, str]] = []
    for index, cell in enumerate(swarm.cells):
        try:
            entry = registry.resolve_model(cell.model, entries)
        except registry.RegistryError as exc:
            problems.append({"path": f"cells[{index}].model", "message": str(exc)})
            continue
        if entry.custom:
            problems.append(
                {
                    "path": f"cells[{index}].model",
                    "message": (
                        f"model {cell.model!r} is not in the model registry; "
                        "use a registry id or restart with --allow-unvetted"
                    ),
                }
            )
            continue
        if cell.runtime is not None and cell.runtime.package_source not in known_sources:
            problems.append(
                {
                    "path": f"cells[{index}].runtime.package_source",
                    "message": (
                        f"package_source {cell.runtime.package_source!r} is not in the model "
                        "registry; restart with --allow-unvetted to accept it"
                    ),
                }
            )
    return problems


def _raise_registry_problems(
    swarm: SwarmManifest,
    entries: Mapping[str, registry.ModelEntry],
) -> None:
    problems = registry_problems(swarm, entries)
    if problems:
        raise AgentError("; ".join(f"{item['path']}: {item['message']}" for item in problems))


def _split_manifest_error(exc: manifest_mod.ManifestError) -> dict[str, str]:
    path, separator, message = str(exc).partition(": ")
    if separator:
        return {"path": path, "message": message}
    return {"path": "swarm.json", "message": str(exc)}


# --------------------------------------------------------------------------- #
# warm swarms
# --------------------------------------------------------------------------- #


class WarmPool:
    """Lazily started swarms, at most ``max_warm`` alive, least-recently-used out.

    A judged pack is never unloaded between queries; switching to another pack
    when the pool is full closes the oldest one, because a swarm is gigabytes of
    model weights and worker processes.
    """

    def __init__(self, config: AgentConfig, entries: Mapping[str, registry.ModelEntry]) -> None:
        self._config = config
        self._entries = entries
        self._swarms: OrderedDict[Path, orchestrator.Swarm] = OrderedDict()
        self._lock = threading.Lock()

    def is_warm(self, pack_dir: Path) -> bool:
        return pack_dir in self._swarms

    def warm_packs(self) -> list[str]:
        return [path.name for path in self._swarms]

    def get(self, pack_dir: Path, swarm: SwarmManifest) -> orchestrator.Swarm:
        """Return this pack's warm swarm, starting it (and evicting) if needed."""
        with self._lock:
            if pack_dir in self._swarms:
                self._swarms.move_to_end(pack_dir)
                return self._swarms[pack_dir]
            plans = orchestrator.plan_cells(
                swarm,
                registry_entries=self._entries,
                envs_root=self._config.envs_root,
                models_root=self._config.models_root,
            )
            warm = orchestrator.Swarm(plans).start()
            self._swarms[pack_dir] = warm
            while len(self._swarms) > max(1, self._config.max_warm):
                _, evicted = self._swarms.popitem(last=False)
                evicted.close()
            return warm

    def close(self) -> None:
        with self._lock:
            for warm in self._swarms.values():
                warm.close()
            self._swarms.clear()


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


def describe_pack(
    path: str | Path,
    *,
    entries: Mapping[str, registry.ModelEntry],
    models_root: str | Path | None = None,
) -> dict[str, Any]:
    """The pack's views, cells, questions and model cache status. No model load."""
    pack_dir = resolve_pack(path, AgentConfig())
    swarm = manifest_mod.load_manifest(pack_dir)
    models = pack.collect_models(swarm, models_root=models_root, registry_entries=entries)
    return {
        "name": swarm.name,
        "path": str(pack_dir),
        "views": [{"name": view.name, "type": view.type} for view in swarm.views],
        "cells": [
            {
                "id": cell.id,
                "model": cell.model,
                "modality": cell.modality,
                "views": list(cell.views),
                "questions": [
                    {"name": question.name, **manifest_mod.question_to_dict(question)}
                    for question in cell.default_questions
                ],
            }
            for cell in swarm.cells
        ],
        "models": [
            {
                "id": model.id,
                "hf_repo": model.hf_repo,
                "subfolder": model.subfolder,
                "revision": model.revision,
                "cached": model.cached,
                "bytes": model.bytes,
            }
            for model in models
        ],
    }


def _pack_summary(path: Path) -> dict[str, Any]:
    swarm = manifest_mod.load_manifest(path)
    return {
        "name": swarm.name,
        "path": str(path),
        "cells": len(swarm.cells),
        "models": sorted({cell.model for cell in swarm.cells}),
    }


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #


_VIEWS_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "description": (
        "The observation: view name -> value. Text views carry the text inline; "
        "image views carry an absolute or relative file path on this machine."
    ),
    "additionalProperties": {"type": "string"},
    "minProperties": 1,
}

_PACK_SCHEMA: Mapping[str, Any] = {
    "type": "string",
    "description": "Pack name, pack directory, or path to its swarm.json. Omit when the server has a default pack.",
}

_QUESTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["choice", "score", "noul"]},
        "instructions": {"type": "string", "minLength": 1},
        "criteria": {
            "description": (
                "choice: object of option -> description (>=2) or a list of >=2 strings. "
                "score: list of >=2 rubric levels, ordered worst to best. "
                "noul: omit (yes/no probability)."
            ),
            "oneOf": [
                {"type": "object", "additionalProperties": {"type": "string"}, "minProperties": 2},
                {"type": "array", "items": {"type": "string"}, "minItems": 2},
            ],
        },
    },
    "required": ["type", "instructions"],
    "additionalProperties": False,
}

_MANIFEST_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "description": "A swarm.json document: 2-8 cells over one shared set of views.",
    "properties": {
        "schema_version": {"type": "integer", "enum": [1]},
        "name": {"type": "string", "minLength": 1},
        "input_schema": {
            "type": "object",
            "properties": {
                "views": {
                    "type": "object",
                    "description": "View name (slug a-z0-9_-) -> its type.",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"type": {"type": "string", "enum": ["image", "text"]}},
                        "required": ["type"],
                        "additionalProperties": False,
                    },
                    "minProperties": 1,
                }
            },
            "required": ["views"],
            "additionalProperties": False,
        },
        "cells": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "pattern": "^[a-z0-9_-]+$"},
                    "model": {
                        "type": "string",
                        "description": "A registry id from list_models (or org/name with --allow-unvetted).",
                    },
                    "modality": {"type": "string", "enum": ["text", "image+text"]},
                    "views": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "View names this cell reads; must exist in input_schema.views.",
                    },
                    "default_questions": {
                        "type": "object",
                        "description": "Question name (slug) -> question.",
                        "additionalProperties": _QUESTION_SCHEMA,
                    },
                    "runtime": {
                        "type": "object",
                        "description": "Only when overriding the registry's package source.",
                        "properties": {
                            "package_source": {"type": "string"},
                            "loader": {"type": "string"},
                        },
                        "required": ["package_source", "loader"],
                        "additionalProperties": False,
                    },
                },
                "required": ["id", "model", "modality", "views"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["schema_version", "name", "input_schema", "cells"],
    "additionalProperties": False,
}

_DESCRIPTIONS: Mapping[str, str] = {
    "describe": (
        "Describe a swarm pack: its views, cells, each cell's questions and criteria, "
        "and whether its models are downloaded. Costs nothing (no model load). Call "
        "this first. With no 'pack' it lists the available packs."
    ),
    "judge": (
        "Ask the swarm to judge one observation and return each cell's typed answers "
        "with confidences. Loads the models on the first call for a pack (seconds); "
        "later calls are warm."
    ),
    "list_models": (
        "List the models an authored pack may use: id, modality, size, license, loader "
        "and whether the weights are already cached."
    ),
    "build": (
        "Validate a swarm.json document and write a pack directory. Models and package "
        "sources must be in the registry. Use dry_run to check a manifest without "
        "writing anything."
    ),
    "fetch": "Download a pack's model weights into the shared cache. Safe to retry.",
    "install": "Build the per-package virtual environments a pack's cells need. Safe to retry.",
    "export": (
        "Bundle a pack with its weights and built envs into one .swarmpack.tar. Runs in "
        "the background because it can take minutes; poll job_status."
    ),
    "job_status": "Check a background job started by export.",
}


class AgentTools:
    """The tool table: one method per tool, all returning JSON-ready dicts."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        registry_entries: Mapping[str, registry.ModelEntry] | None = None,
        pool: WarmPool | None = None,
    ) -> None:
        self._config = config
        self._entries = (
            registry.load_registry() if registry_entries is None else registry_entries
        )
        self._pool = pool if pool is not None else WarmPool(config, self._entries)
        self._jobs: dict[str, _Job] = {}

    # -- judging ---------------------------------------------------------- #

    def describe(self, args: Mapping[str, Any]) -> dict[str, Any]:
        spec = args.get("pack")
        if spec is None and self._config.default_pack is None and not self._config.packs_dirs:
            raise AgentError(
                "no pack configured; start the server with --pack or --packs-dir, "
                "or pass 'pack' explicitly"
            )
        if spec is None and self._config.default_pack is None:
            found = discover_packs(self._config)
            if len(found) != 1:
                return {
                    "packs": [_pack_summary(path) for path in found],
                    "default": None,
                }
        pack_dir = resolve_pack(spec, self._config)
        payload = describe_pack(
            pack_dir, entries=self._entries, models_root=self._config.models_root
        )
        payload["warm"] = self._pool.is_warm(pack_dir)
        return payload

    def judge(self, args: Mapping[str, Any]) -> dict[str, Any]:
        views = args.get("views")
        if not isinstance(views, Mapping) or not views:
            raise AgentError("'views' must be a non-empty object of view name -> value")
        pack_dir = resolve_pack(args.get("pack"), self._config)
        swarm = manifest_mod.load_manifest(pack_dir)

        if not self._pool.is_warm(pack_dir):
            missing = [
                model.id
                for model in pack.collect_models(
                    swarm,
                    models_root=self._config.models_root,
                    registry_entries=self._entries,
                )
                if not model.cached
            ]
            if missing:
                raise AgentError(
                    "model(s) not in the cache: "
                    + ", ".join(missing)
                    + "; call the 'fetch' tool first (or run 'decision-swarm fetch')"
                )

        message: dict[str, Any] = {"type": "query", "views": dict(views)}
        if args.get("questions") is not None:
            message["questions"] = args["questions"]

        warm = self._pool.get(pack_dir, swarm)
        try:
            _message_id, result = serve.run_query(warm, swarm, message)
        except Exception as exc:  # noqa: BLE001 - surfaced to the agent verbatim
            raise AgentError(str(exc)) from exc

        detail = args.get("detail", "compact")
        projection = result.to_dict() if detail == "full" else result.to_compact()
        return {"pack": swarm.name, "path": str(pack_dir), **projection}

    # -- authoring -------------------------------------------------------- #

    def list_models(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "models": [
                {
                    "id": entry.id,
                    "hf_repo": entry.hf_repo,
                    "subfolder": entry.subfolder,
                    "revision": entry.revision,
                    "modality": entry.modality,
                    "params": entry.params,
                    "download_bytes": entry.download_bytes,
                    "license": entry.display_license,
                    "loader": entry.loader,
                    "package_source": entry.package_source,
                    "custom": entry.custom,
                    "cached": cache.is_cached(entry, self._config.models_root),
                }
                for entry in self._entries.values()
            ]
        }

    def build(self, args: Mapping[str, Any]) -> dict[str, Any]:
        raw = args.get("manifest")
        if not isinstance(raw, Mapping):
            raise AgentError("'manifest' must be the swarm.json document as a JSON object")
        try:
            swarm = manifest_mod.parse_manifest(raw)
        except manifest_mod.ManifestError as exc:
            return {"ok": False, "errors": [_split_manifest_error(exc)]}

        if not self._config.allow_unvetted:
            problems = registry_problems(swarm, self._entries)
            if problems:
                return {"ok": False, "errors": problems}

        dest = Path(str(args["dest"])).expanduser()
        pin = bool(args.get("pin", True))
        models = pack.collect_models(
            swarm,
            models_root=self._config.models_root,
            registry_entries=self._entries,
            resolve_commit=pack.hub_commit_resolver if pin else None,
        )
        summary: dict[str, Any] = {
            "ok": True,
            "pack": str(dest),
            "name": swarm.name,
            "cells": [cell.id for cell in swarm.cells],
            "models": [
                {"id": model.id, "cached": model.cached, "bytes": model.bytes}
                for model in models
            ],
            "mcp_config": pack.mcp_config(dest),
        }
        if args.get("dry_run"):
            summary["dry_run"] = True
            return summary

        built = pack.build_pack(
            swarm,
            dest,
            models_root=self._config.models_root,
            registry_entries=self._entries,
            models=models,
        )
        summary["pack"] = str(built)
        summary["mcp_config"] = pack.mcp_config(built)
        return summary

    def fetch(self, args: Mapping[str, Any]) -> dict[str, Any]:
        pack_dir = resolve_pack(args.get("pack"), self._config)
        swarm = manifest_mod.load_manifest(pack_dir)
        results: list[dict[str, Any]] = []
        failed = False
        for entry in pack.unique_entries(swarm, registry_entries=self._entries):
            try:
                fetched = cache.fetch(entry, root=self._config.models_root)
            except (cache.CacheError, OSError) as exc:
                failed = True
                results.append({"id": entry.id, "state": "error", "message": str(exc)})
                continue
            results.append(
                {
                    "id": entry.id,
                    "state": "downloaded" if fetched.downloaded else "cached",
                    "bytes": fetched.bytes,
                }
            )
        return {
            "pack": str(pack_dir),
            "models": results,
            "all_cached": not failed,
        }

    def install(self, args: Mapping[str, Any]) -> dict[str, Any]:
        pack_dir = resolve_pack(args.get("pack"), self._config)
        swarm = manifest_mod.load_manifest(pack_dir)
        if not self._config.allow_unvetted:
            _raise_registry_problems(swarm, self._entries)
        planned = envs.cell_envs(
            swarm, envs_root=self._config.envs_root, registry_entries=self._entries
        )
        results: list[dict[str, Any]] = []
        for cell_env in planned:
            existed = cell_env.python.exists()
            try:
                envs.build_env(cell_env, installer=_capturing_installer)
            except (envs.EnvError, OSError) as exc:
                results.append(
                    {
                        "package_source": cell_env.package_source,
                        "key": cell_env.key,
                        "state": "error",
                        "message": str(exc),
                    }
                )
                continue
            results.append(
                {
                    "package_source": cell_env.package_source,
                    "key": cell_env.key,
                    "state": "reused" if existed else "built",
                    "cells": list(cell_env.shared_by),
                }
            )
        return {"pack": str(pack_dir), "envs": results}

    def export(self, args: Mapping[str, Any]) -> dict[str, Any]:
        pack_dir = resolve_pack(args.get("pack"), self._config)
        if not (pack_dir / MANIFEST_NAME).is_file():
            raise AgentError(f"{pack_dir} is not a pack directory")
        dest = args.get("dest")
        job = _Job(id=f"export-{len(self._jobs) + 1}", kind="export")
        self._jobs[job.id] = job
        include_envs = not bool(args.get("no_envs", False))

        def run() -> None:
            try:
                bundle = pack.export_bundle(
                    pack_dir,
                    dest,
                    models_root=self._config.models_root,
                    envs_root=self._config.envs_root,
                    include_envs=include_envs,
                )
            except Exception as exc:  # noqa: BLE001 - reported through job_status
                job.message = str(exc)
                job.state = "error"
                return
            job.result = {"bundle": str(bundle), "bytes": bundle.stat().st_size}
            job.state = "done"

        # The receipt is the state at submission: read it before the thread can
        # run, or a fast failure (e.g. a cold cache) lands before the read and
        # the caller is told "error" for a job it has not polled yet.
        receipt = {"job_id": job.id, "kind": job.kind, "state": job.state, "pack": str(pack_dir)}
        threading.Thread(target=run, name=job.id, daemon=True).start()
        return receipt

    def job_status(self, args: Mapping[str, Any]) -> dict[str, Any]:
        job_id = str(args.get("job_id"))
        job = self._jobs.get(job_id)
        if job is None:
            raise AgentError(f"job {job_id!r} not found (server restarted?); re-run export")
        payload: dict[str, Any] = {
            "job_id": job.id,
            "kind": job.kind,
            "state": job.state,
            "elapsed_s": round(time.time() - job.started, 1),
        }
        if job.result is not None:
            payload["result"] = job.result
        if job.message is not None:
            payload["message"] = job.message
        return payload

    # -- table ------------------------------------------------------------ #

    def tools(self) -> dict[str, ToolSpec]:
        """The tool table keyed by name (what the MCP wire exposes)."""
        return {
            "describe": ToolSpec("describe", _DESCRIPTIONS["describe"], _DESCRIBE_ARGS, self.describe),
            "judge": ToolSpec("judge", _DESCRIPTIONS["judge"], _JUDGE_ARGS, self.judge),
            "list_models": ToolSpec(
                "list_models", _DESCRIPTIONS["list_models"], _NO_ARGS, self.list_models
            ),
            "build": ToolSpec("build", _DESCRIPTIONS["build"], _BUILD_ARGS, self.build),
            "fetch": ToolSpec("fetch", _DESCRIPTIONS["fetch"], _PACK_ONLY_ARGS, self.fetch),
            "install": ToolSpec("install", _DESCRIPTIONS["install"], _PACK_ONLY_ARGS, self.install),
            "export": ToolSpec("export", _DESCRIPTIONS["export"], _EXPORT_ARGS, self.export),
            "job_status": ToolSpec(
                "job_status", _DESCRIPTIONS["job_status"], _JOB_STATUS_ARGS, self.job_status
            ),
        }

    def close(self) -> None:
        self._pool.close()


_NO_ARGS: Mapping[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

_DESCRIBE_ARGS: Mapping[str, Any] = {
    "type": "object",
    "properties": {"pack": _PACK_SCHEMA},
    "additionalProperties": False,
}

_JUDGE_ARGS: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "views": _VIEWS_SCHEMA,
        "questions": {
            "type": "object",
            "description": (
                "Optional per-cell question overrides: cell id -> question name -> question. "
                "Omit to use the pack's default questions."
            ),
            "additionalProperties": {
                "type": "object",
                "additionalProperties": _QUESTION_SCHEMA,
            },
        },
        "pack": _PACK_SCHEMA,
        "detail": {
            "type": "string",
            "enum": ["compact", "full"],
            "default": "compact",
            "description": "compact: cell -> question -> {value, confidence}. full: adds model, latency, warnings.",
        },
    },
    "required": ["views"],
    "additionalProperties": False,
}

_BUILD_ARGS: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "manifest": _MANIFEST_SCHEMA,
        "dest": {"type": "string", "description": "Pack output directory (created if absent)."},
        "dry_run": {
            "type": "boolean",
            "default": False,
            "description": "Validate only; write nothing.",
        },
        "pin": {
            "type": "boolean",
            "default": True,
            "description": "Resolve floating model revisions to Hub commits (one metadata call per model).",
        },
    },
    "required": ["manifest", "dest"],
    "additionalProperties": False,
}

_PACK_ONLY_ARGS: Mapping[str, Any] = {
    "type": "object",
    "properties": {"pack": _PACK_SCHEMA},
    "additionalProperties": False,
}

_EXPORT_ARGS: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "pack": _PACK_SCHEMA,
        "dest": {"type": "string", "description": "Bundle path (default: <pack>.swarmpack.tar beside the pack)."},
        "no_envs": {
            "type": "boolean",
            "default": False,
            "description": "Bundle weights only; the recipient runs install.",
        },
    },
    "additionalProperties": False,
}

_JOB_STATUS_ARGS: Mapping[str, Any] = {
    "type": "object",
    "properties": {"job_id": {"type": "string"}},
    "required": ["job_id"],
    "additionalProperties": False,
}


def _capturing_installer(python: Path, requirements: Sequence[str]) -> None:
    """``pip install`` with captured output — stdout belongs to the transport."""
    try:
        completed = subprocess.run(
            [str(python), "-m", "pip", "install", *requirements],
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise envs.EnvError(f"pip install into {python} failed: {exc}") from exc
    if completed.returncode != 0:
        tail = " | ".join(completed.stderr.strip().splitlines()[-3:])
        raise envs.EnvError(
            f"pip install into {python} failed ({', '.join(requirements)}): {tail}"
        )
