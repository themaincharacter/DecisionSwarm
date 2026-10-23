"""Tests for the ``swarm.json`` schema and validation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from decision_swarm import manifest

VALID: dict[str, Any] = {
    "schema_version": 1,
    "name": "browser-swarm",
    "input_schema": {
        "views": {
            "screenshot": {"type": "image"},
            "dom": {"type": "text"},
            "classes": {"type": "text"},
        }
    },
    "cells": [
        {
            "id": "screenshot_sensor",
            "model": "thaitea/laya-vision",
            "modality": "image+text",
            "views": ["screenshot"],
            "default_questions": {
                "submit_visible": {"type": "noul", "instructions": "Do you see a submit button?"}
            },
            "runtime": {
                "package_source": "git+https://github.com/r33drichards/laya-vision",
                "loader": "load_vlm",
            },
        },
        {
            "id": "dom_sensor",
            "model": "convaiinnovations/laya",
            "modality": "text",
            "views": ["dom", "classes"],
            "default_questions": {
                "submit_button": {
                    "type": "choice",
                    "instructions": "Which element is the submit button?",
                    "criteria": {
                        "a": "the input with value Send",
                        "b": "the button labeled Send",
                    },
                }
            },
        },
    ],
}


def _with(mutate: Any) -> dict[str, Any]:
    data = copy.deepcopy(VALID)
    mutate(data)
    return data


def test_parses_the_readme_example() -> None:
    parsed = manifest.parse_manifest(VALID)
    assert parsed.name == "browser-swarm"
    assert [view.name for view in parsed.views] == ["screenshot", "dom", "classes"]
    assert [cell.id for cell in parsed.cells] == ["screenshot_sensor", "dom_sensor"]

    first, second = parsed.cells
    assert first.runtime is not None and first.runtime.loader == "load_vlm"
    assert [q.name for q in first.default_questions] == ["submit_visible"]
    assert first.default_questions[0].type == "noul"
    assert first.default_questions[0].criteria is None

    assert second.runtime is None
    question = second.default_questions[0]
    assert question.type == "choice"
    assert question.criteria == {"a": "the input with value Send", "b": "the button labeled Send"}


def test_round_trips_to_dict() -> None:
    assert manifest.to_dict(manifest.parse_manifest(VALID)) == VALID


def test_load_and_save_manifest(tmp_path: Any) -> None:
    parsed = manifest.parse_manifest(VALID)
    written = manifest.save_manifest(parsed, tmp_path)
    assert written == tmp_path / "swarm.json"
    assert manifest.to_dict(manifest.load_manifest(tmp_path)) == VALID
    assert manifest.to_dict(manifest.load_manifest(written)) == VALID


def test_rejects_non_object_documents() -> None:
    with pytest.raises(manifest.ManifestError, match="must be a JSON object"):
        manifest.parse_manifest([1, 2])


def test_rejects_invalid_json(tmp_path: Any) -> None:
    bad = tmp_path / "swarm.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(manifest.ManifestError, match="invalid JSON"):
        manifest.load_manifest(bad)


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(name=""), "name"),
        (lambda d: d.update(extra=1), "unknown field"),
        (lambda d: d.update(cells=d["cells"][:1]), "between 2 and 8"),
        (lambda d: d.update(cells=d["cells"] * 5), "between 2 and 8"),
        (lambda d: d["cells"][1].update(id="screenshot_sensor"), "duplicate cell id"),
        (lambda d: d["cells"][0].update(id="Screenshot Sensor"), "slug"),
        (lambda d: d["cells"][0].update(views=["shot"]), "unknown view"),
        (lambda d: d["cells"][0].update(views=["screenshot", "screenshot"]), "duplicate view"),
        (lambda d: d["cells"][0].update(views=[]), "non-empty list"),
        (lambda d: d["cells"][0].update(model=""), "model"),
        (lambda d: d["cells"][0].update(modality="audio"), "modality"),
        (lambda d: d["cells"][0].update(extra=1), "unknown field"),
        (lambda d: d["input_schema"]["views"]["dom"].update(type="audio"), "must be one of image, text"),
        (
            lambda d: d["cells"][0]["default_questions"]["submit_visible"].update(type="maybe"),
            "must be one of choice, score, noul",
        ),
        (
            lambda d: d["cells"][0]["default_questions"]["submit_visible"].pop("instructions"),
            "instructions",
        ),
        (
            lambda d: d["cells"][0]["default_questions"]["submit_visible"].update(criteria=["x"]),
            "noul questions take no criteria",
        ),
        (
            lambda d: d["cells"][1]["default_questions"]["submit_button"].pop("criteria"),
            "'criteria' is required",
        ),
        (
            lambda d: d["cells"][1]["default_questions"]["submit_button"].update(criteria=["only"]),
            "at least 2",
        ),
        (
            lambda d: d["cells"][1]["default_questions"]["submit_button"].update(
                criteria={"a": "one", "b": ""}
            ),
            "non-empty",
        ),
        (
            lambda d: d["cells"][0]["default_questions"]["submit_visible"].update(
                type="score", criteria="high"
            ),
            "rubric levels",
        ),
        (lambda d: d["cells"][0].update(runtime={"loader": "load_vlm"}), "package_source"),
    ],
)
def test_rejects_invalid_manifests(mutate: Any, fragment: str) -> None:
    with pytest.raises(manifest.ManifestError) as excinfo:
        manifest.parse_manifest(_with(mutate))
    assert fragment in str(excinfo.value)
