"""Tests for query payloads, question resolution, and the result schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from decision_swarm import manifest as manifest_mod, query
from test_manifest import VALID

QUERY: dict[str, Any] = {
    "views": {
        "screenshot": "shot.png",
        "dom": "<html><button>Send</button></html>",
        "classes": "btn btn-primary",
    },
    "questions": {
        "dom_sensor": {
            "submit_button": {
                "type": "choice",
                "instructions": "Which element submits the form?",
                "criteria": {"a": "the input with value Send", "b": "the button labeled Send"},
            },
            "dom_length": {
                "type": "score",
                "instructions": "How long is the DOM?",
                "criteria": ["short", "long"],
            },
        }
    },
}


@pytest.fixture()
def swarm() -> manifest_mod.SwarmManifest:
    return manifest_mod.parse_manifest(VALID)


def test_parses_valid_query(swarm: manifest_mod.SwarmManifest) -> None:
    parsed = query.parse_query(QUERY, swarm)
    assert dict(parsed.views) == QUERY["views"]
    assert [question.name for question in parsed.questions["dom_sensor"]] == [
        "submit_button",
        "dom_length",
    ]
    assert parsed.questions["dom_sensor"][0].instructions == "Which element submits the form?"


def test_defaults_apply_without_overrides(swarm: manifest_mod.SwarmManifest) -> None:
    parsed = query.parse_query({"views": QUERY["views"]}, swarm)
    for cell in swarm.cells:
        assert query.resolve_questions(cell, parsed) == cell.default_questions


def test_overrides_win_per_name_and_new_names_append(
    swarm: manifest_mod.SwarmManifest,
) -> None:
    parsed = query.parse_query(QUERY, swarm)
    dom_cell = next(cell for cell in swarm.cells if cell.id == "dom_sensor")
    merged = query.resolve_questions(dom_cell, parsed)

    assert [question.name for question in merged] == ["submit_button", "dom_length"]
    assert merged[0].instructions == "Which element submits the form?"  # override won
    assert merged[0].criteria == {
        "a": "the input with value Send",
        "b": "the button labeled Send",
    }

    # other cells are untouched by this cell's overrides
    other = next(cell for cell in swarm.cells if cell.id == "screenshot_sensor")
    assert query.resolve_questions(other, parsed) == other.default_questions


def test_load_query_from_file(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    path = tmp_path / "obs.json"
    path.write_text(json.dumps(QUERY), encoding="utf-8")
    parsed = query.load_query(path, swarm)
    assert dict(parsed.views) == QUERY["views"]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d.update(extra=1), "unknown field"),
        (lambda d: d.update(views={}), "non-empty object"),
        (lambda d: d["views"].update(nope="x"), "unknown view"),
        (lambda d: d["views"].update(screenshot=""), "non-empty string"),
        (lambda d: d["views"].pop("classes"), "missing view"),
        (lambda d: d["questions"].update(ghost_sensor={}), "unknown cell id"),
        (
            lambda d: d["questions"]["dom_sensor"]["submit_button"].pop("criteria"),
            "'criteria' is required",
        ),
    ],
)
def test_rejects_invalid_queries(
    mutate: Any, fragment: str, swarm: manifest_mod.SwarmManifest
) -> None:
    data = json.loads(json.dumps(QUERY))
    mutate(data)
    with pytest.raises(query.QueryError) as excinfo:
        query.parse_query(data, swarm)
    assert fragment in str(excinfo.value)


def test_rejects_non_object_queries(swarm: manifest_mod.SwarmManifest) -> None:
    with pytest.raises(query.QueryError, match="must be a JSON object"):
        query.parse_query(["not", "a", "query"], swarm)


def test_result_schema_matches_the_plan() -> None:
    result = query.SwarmResult(
        cells={
            "screenshot_sensor": query.CellResult(
                cell_id="screenshot_sensor",
                model="thaitea/laya-vision",
                answers={
                    "submit_visible": query.Answer(
                        name="submit_visible", type="noul", value=0.94, confidence=0.91
                    )
                },
                latency_ms=41,
            )
        }
    )
    assert result.to_dict() == {
        "cells": {
            "screenshot_sensor": {
                "answers": {"submit_visible": {"noul": 0.94, "confidence": 0.91}},
                "model": "thaitea/laya-vision",
                "latency_ms": 41,
                "warnings": [],
            }
        }
    }


def test_answers_pass_extra_fields_through() -> None:
    answer = query.Answer(
        name="submit_button",
        type="choice",
        value="b",
        confidence=0.8,
        extra={"options": {"a": 0.2, "b": 0.8}},
    )
    assert answer.to_dict() == {
        "choice": "b",
        "confidence": 0.8,
        "options": {"a": 0.2, "b": 0.8},
    }
