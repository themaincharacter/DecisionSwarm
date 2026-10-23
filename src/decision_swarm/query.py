"""Query payloads and the result schema.

A query payload is what a run receives: named views of one observation plus
optional per-cell question overrides. Questions resolve per name — the query
override wins, pack defaults fill the rest — so answers stay comparable when
no override is given. The result schema is the v1 unaggregated shape: raw
per-cell typed answers (aggregation/consensus is deferred).

Image views carry a file path; text views carry inline text.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from decision_swarm.manifest import (
    CellSpec,
    ManifestError,
    QuestionSpec,
    SwarmManifest,
    parse_question,
)


class QueryError(ValueError):
    """Raised when a query payload fails validation."""


@dataclass(frozen=True)
class QueryPayload:
    """A validated query: observation views + per-cell question overrides."""

    views: Mapping[str, str]
    questions: Mapping[str, tuple[QuestionSpec, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Answer:
    """One typed answer: the value under its question-type key plus confidence."""

    name: str
    type: str  # choice | score | noul
    value: Any
    confidence: float
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {self.type: self.value, "confidence": self.confidence}
        out.update(self.extra)
        return out


@dataclass(frozen=True)
class CellResult:
    """Raw answers from one cell (v1: no aggregation)."""

    cell_id: str
    model: str
    answers: Mapping[str, Answer]
    latency_ms: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "answers": {name: answer.to_dict() for name, answer in self.answers.items()},
            "model": self.model,
            "latency_ms": self.latency_ms,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SwarmResult:
    """The v1 result schema: per-cell results keyed by cell id."""

    cells: Mapping[str, CellResult]

    def to_dict(self) -> dict[str, Any]:
        return {"cells": {cell_id: result.to_dict() for cell_id, result in self.cells.items()}}


def _fail(path: str, message: str) -> QueryError:
    return QueryError(f"{path}: {message}")


def _cell_for(swarm: SwarmManifest, cell_id: Any) -> CellSpec:
    if not isinstance(cell_id, str):
        raise _fail("questions", f"cell id must be a string (got {cell_id!r})")
    for cell in swarm.cells:
        if cell.id == cell_id:
            return cell
    raise _fail(f"questions.{cell_id}", "unknown cell id")


def parse_query(data: Any, swarm: SwarmManifest) -> QueryPayload:
    """Validate a decoded query payload against the pack manifest."""
    if not isinstance(data, Mapping):
        raise _fail("query", "must be a JSON object")
    unknown = set(data) - {"views", "questions"}
    if unknown:
        raise _fail("query", f"unknown field(s): {', '.join(sorted(unknown))}")

    raw_views = data.get("views")
    if not isinstance(raw_views, Mapping) or not raw_views:
        raise _fail("views", "must be a non-empty object of view name -> payload")
    known_views = {view.name for view in swarm.views}
    bound_views = {name for cell in swarm.cells for name in cell.views}
    views: dict[str, str] = {}
    for name, value in raw_views.items():
        if name not in known_views:
            raise _fail(f"views.{name}", "unknown view")
        if not isinstance(value, str) or not value:
            raise _fail(f"views.{name}", "must be a non-empty string")
        views[name] = value
    missing = sorted(bound_views - set(views))
    if missing:
        raise _fail("views", f"missing view(s) bound by cells: {', '.join(missing)}")

    raw_questions = data.get("questions", {})
    if not isinstance(raw_questions, Mapping):
        raise _fail("questions", "must be an object of cell id -> questions")
    questions: dict[str, tuple[QuestionSpec, ...]] = {}
    for cell_id, raw_cell_questions in raw_questions.items():
        _cell_for(swarm, cell_id)
        if not isinstance(raw_cell_questions, Mapping):
            raise _fail(f"questions.{cell_id}", "must be an object of name -> question")
        specs = []
        for name, raw_question in raw_cell_questions.items():
            try:
                specs.append(parse_question(name, raw_question, f"questions.{cell_id}.{name}"))
            except ManifestError as exc:
                raise QueryError(str(exc)) from exc
        questions[cell_id] = tuple(specs)
    return QueryPayload(views=views, questions=questions)


def load_query(path: str | Path, swarm: SwarmManifest) -> QueryPayload:
    """Load and validate a query payload from a JSON file."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QueryError(f"{file_path}: cannot read query ({exc})") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QueryError(f"{file_path}: invalid JSON ({exc})") from exc
    return parse_query(data, swarm)


def resolve_questions(cell: CellSpec, query: QueryPayload) -> tuple[QuestionSpec, ...]:
    """Merge pack defaults with the query's overrides (query wins per name).

    Order is deterministic: default questions first (overridden in place),
    then any new names introduced by the query.
    """
    merged: dict[str, QuestionSpec] = {question.name: question for question in cell.default_questions}
    for question in query.questions.get(cell.id, ()):
        merged[question.name] = question
    return tuple(merged.values())
