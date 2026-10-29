"""``swarm.json`` schema and validation.

A pack manifest pins the shape of an observation (named views), the cells
(model + view bindings + default questions), and optional runtime pins.
``parse_manifest`` validates a decoded ``swarm.json`` document and returns the
typed :class:`SwarmManifest`; :func:`to_dict` writes it back faithfully.

Name rules — these names are used as JSON keys in query payloads and results,
so view names, cell ids and question names are slugs (``a-z0-9_-``) and unique
within their scope.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_NAME = "swarm.json"
"""Manifest filename inside a pack."""

SCHEMA_VERSION = 1
"""Current ``swarm.json`` schema version."""

SLUG_PATTERN = r"^[a-z0-9_-]+$"
"""Names used as JSON keys — view names, cell ids, question names."""

CELL_ID_PATTERN = SLUG_PATTERN
"""Cell ids are slugs, unique within a pack."""

MIN_CELLS, MAX_CELLS = 2, 8
"""Swarm size bounds."""

QUESTION_TYPES = ("choice", "score", "noul")
"""Typed questions a cell can answer: options, rubric level, yes/no probability."""

VIEW_TYPES = ("image", "text")
"""View types an observation can carry."""

MODALITIES = ("text", "image+text")
"""Cell modalities."""

_SLUG_RE = re.compile(SLUG_PATTERN)
_ROOT_FIELDS = {"schema_version", "name", "input_schema", "cells"}
_CELL_FIELDS = {"id", "model", "modality", "views", "default_questions", "runtime", "adapter"}
_QUESTION_FIELDS = {"type", "instructions", "criteria"}
_RUNTIME_FIELDS = {"package_source", "loader"}

Criteria = dict[str, str] | list[str] | None


class ManifestError(ValueError):
    """Raised when a ``swarm.json`` document fails validation."""


@dataclass(frozen=True)
class ViewSpec:
    """One named view of an observation."""

    name: str
    type: str  # VIEW_TYPES


@dataclass(frozen=True)
class QuestionSpec:
    """One typed question a cell answers."""

    name: str
    type: str  # QUESTION_TYPES
    instructions: str
    criteria: Criteria = None


@dataclass(frozen=True)
class RuntimeSpec:
    """Runtime pins for a cell: how to install and load its model."""

    package_source: str
    loader: str


@dataclass(frozen=True)
class CellSpec:
    """One sensor: a model bound to views with default questions."""

    id: str
    model: str
    modality: str  # MODALITIES
    views: tuple[str, ...]
    default_questions: tuple[QuestionSpec, ...] = ()
    runtime: RuntimeSpec | None = None
    adapter: str | None = None  # 0..1, never validated (Q9/Q10)


@dataclass(frozen=True)
class SwarmManifest:
    """A validated ``swarm.json`` document."""

    schema_version: int
    name: str
    views: tuple[ViewSpec, ...]
    cells: tuple[CellSpec, ...]


def _fail(path: str, message: str) -> ManifestError:
    return ManifestError(f"{path}: {message}")


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise _fail(path, f"unknown field(s): {', '.join(sorted(unknown))}")


def _require_slug(value: Any, path: str) -> str:
    if not isinstance(value, str) or not _SLUG_RE.match(value):
        raise _fail(path, f"must be a slug matching {SLUG_PATTERN} (got {value!r})")
    return value


def _require_str(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(path, f"must be a non-empty string (got {value!r})")
    return value


def _is_list(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _parse_criteria(qtype: str, raw: Any, path: str) -> Criteria:
    if qtype == "noul":
        if raw is not None:
            raise _fail(path, "noul questions take no criteria")
        return None
    if raw is None:
        raise _fail(path, f"'criteria' is required for {qtype} questions")
    if qtype == "score":
        if not _is_list(raw) or len(raw) < 2:
            raise _fail(path, "score criteria must be a list of at least 2 rubric levels")
        return [_require_str(item, f"{path}[{i}]") for i, item in enumerate(raw)]
    if isinstance(raw, Mapping):
        if len(raw) < 2:
            raise _fail(path, "choice criteria must offer at least 2 options")
        return {
            _require_str(key, f"{path} option name"): _require_str(value, f"{path}[{key!r}]")
            for key, value in raw.items()
        }
    if _is_list(raw) and len(raw) >= 2:
        return [_require_str(item, f"{path}[{i}]") for i, item in enumerate(raw)]
    raise _fail(
        path,
        "choice criteria must be an object of option -> description "
        "or a list of at least 2 strings",
    )


def parse_question(name: str, raw: Any, path: str) -> QuestionSpec:
    _require_slug(name, path)
    if not isinstance(raw, Mapping):
        raise _fail(path, "must be an object with 'type' and 'instructions'")
    _reject_unknown_keys(raw, _QUESTION_FIELDS, path)
    qtype = raw.get("type")
    if qtype not in QUESTION_TYPES:
        raise _fail(f"{path}.type", f"must be one of {', '.join(QUESTION_TYPES)} (got {qtype!r})")
    return QuestionSpec(
        name=name,
        type=qtype,
        instructions=_require_str(raw.get("instructions"), f"{path}.instructions"),
        criteria=_parse_criteria(qtype, raw.get("criteria"), f"{path}.criteria"),
    )


def _parse_views(data: Any) -> tuple[ViewSpec, ...]:
    if not isinstance(data, Mapping):
        raise _fail("input_schema", "must be an object with a 'views' mapping")
    _reject_unknown_keys(data, {"views"}, "input_schema")
    raw_views = data.get("views")
    if not isinstance(raw_views, Mapping) or not raw_views:
        raise _fail("input_schema.views", "must be a non-empty object of name -> {type}")
    views: list[ViewSpec] = []
    for name, spec in raw_views.items():
        path = f"input_schema.views.{name}"
        _require_slug(name, path)
        if not isinstance(spec, Mapping):
            raise _fail(path, "must be an object with a 'type' field")
        _reject_unknown_keys(spec, {"type"}, path)
        vtype = spec.get("type")
        if vtype not in VIEW_TYPES:
            raise _fail(f"{path}.type", f"must be one of {', '.join(VIEW_TYPES)} (got {vtype!r})")
        views.append(ViewSpec(name=name, type=vtype))
    return tuple(views)


def _parse_runtime(raw: Any, path: str) -> RuntimeSpec:
    if not isinstance(raw, Mapping):
        raise _fail(path, "must be an object with 'package_source' and 'loader'")
    _reject_unknown_keys(raw, _RUNTIME_FIELDS, path)
    return RuntimeSpec(
        package_source=_require_str(raw.get("package_source"), f"{path}.package_source"),
        loader=_require_str(raw.get("loader"), f"{path}.loader"),
    )


def _parse_cell(raw: Any, index: int, known_views: set[str], seen_ids: set[str]) -> CellSpec:
    path = f"cells[{index}]"
    if not isinstance(raw, Mapping):
        raise _fail(path, "must be an object")
    _reject_unknown_keys(raw, _CELL_FIELDS, path)
    cell_id = _require_slug(raw.get("id"), f"{path}.id")
    if cell_id in seen_ids:
        raise _fail(f"{path}.id", f"duplicate cell id {cell_id!r}")
    seen_ids.add(cell_id)
    modality = raw.get("modality")
    if modality not in MODALITIES:
        raise _fail(f"{path}.modality", f"must be one of {', '.join(MODALITIES)} (got {modality!r})")

    raw_views = raw.get("views")
    if not _is_list(raw_views) or not raw_views:
        raise _fail(f"{path}.views", "must be a non-empty list of view names")
    views: list[str] = []
    for i, view in enumerate(raw_views):
        if view not in known_views:
            raise _fail(f"{path}.views[{i}]", f"unknown view {view!r}")
        if view in views:
            raise _fail(f"{path}.views[{i}]", f"duplicate view {view!r}")
        views.append(view)

    raw_questions = raw.get("default_questions", {})
    if not isinstance(raw_questions, Mapping):
        raise _fail(f"{path}.default_questions", "must be an object of name -> question")
    questions = tuple(
        parse_question(name, question, f"{path}.default_questions.{name}")
        for name, question in raw_questions.items()
    )

    adapter = raw.get("adapter")
    if adapter is not None and (not isinstance(adapter, str) or not adapter.strip()):
        raise _fail(f"{path}.adapter", "must be a non-empty string (a registered adapter id or local/<name>@<digest>)")

    return CellSpec(
        id=cell_id,
        model=_require_str(raw.get("model"), f"{path}.model"),
        modality=modality,
        views=tuple(views),
        default_questions=questions,
        runtime=_parse_runtime(raw.get("runtime"), f"{path}.runtime") if "runtime" in raw else None,
        adapter=adapter,
    )


def parse_manifest(data: Any) -> SwarmManifest:
    """Validate a decoded ``swarm.json`` document.

    Raises :class:`ManifestError` (with a field path in the message) on the
    first problem found.
    """
    if not isinstance(data, Mapping):
        raise _fail("swarm.json", "must be a JSON object")
    _reject_unknown_keys(data, _ROOT_FIELDS, "swarm.json")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise _fail("schema_version", f"expected {SCHEMA_VERSION}, got {version!r}")

    views = _parse_views(data.get("input_schema"))
    raw_cells = data.get("cells")
    if not _is_list(raw_cells):
        raise _fail("cells", "must be a list")
    if not (MIN_CELLS <= len(raw_cells) <= MAX_CELLS):
        raise _fail("cells", f"must have between {MIN_CELLS} and {MAX_CELLS} cells (got {len(raw_cells)})")

    known_views = {view.name for view in views}
    seen_ids: set[str] = set()
    cells = tuple(_parse_cell(raw, i, known_views, seen_ids) for i, raw in enumerate(raw_cells))

    return SwarmManifest(
        schema_version=SCHEMA_VERSION,
        name=_require_str(data.get("name"), "name"),
        views=views,
        cells=cells,
    )


def question_to_dict(question: QuestionSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"type": question.type, "instructions": question.instructions}
    if question.criteria is not None:
        out["criteria"] = question.criteria
    return out


def to_dict(manifest: SwarmManifest) -> dict[str, Any]:
    """Serialize a manifest back to the ``swarm.json`` document shape."""
    cells: list[dict[str, Any]] = []
    for cell in manifest.cells:
        entry: dict[str, Any] = {
            "id": cell.id,
            "model": cell.model,
            "modality": cell.modality,
            "views": list(cell.views),
        }
        if cell.default_questions:
            entry["default_questions"] = {
                question.name: question_to_dict(question) for question in cell.default_questions
            }
        if cell.runtime is not None:
            entry["runtime"] = {
                "package_source": cell.runtime.package_source,
                "loader": cell.runtime.loader,
            }
        if cell.adapter is not None:
            entry["adapter"] = cell.adapter
        cells.append(entry)
    return {
        "schema_version": manifest.schema_version,
        "name": manifest.name,
        "input_schema": {"views": {view.name: {"type": view.type} for view in manifest.views}},
        "cells": cells,
    }


def load_manifest(path: str | Path) -> SwarmManifest:
    """Load and validate a manifest from a ``swarm.json`` file or a pack directory."""
    file_path = Path(path)
    if file_path.is_dir():
        file_path = file_path / MANIFEST_NAME
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"{file_path}: cannot read manifest ({exc})") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{file_path}: invalid JSON ({exc})") from exc
    return parse_manifest(data)


def save_manifest(manifest: SwarmManifest, path: str | Path) -> Path:
    """Write a manifest as ``swarm.json`` (to a file or into a pack directory)."""
    file_path = Path(path)
    if file_path.is_dir() or file_path.suffix != ".json":
        file_path = file_path / MANIFEST_NAME
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(to_dict(manifest), indent=2) + "\n", encoding="utf-8")
    return file_path
