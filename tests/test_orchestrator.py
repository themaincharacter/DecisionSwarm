"""Orchestrator tests — includes the browser-control swarm end to end."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from decision_swarm import manifest as manifest_mod, query
from decision_swarm.runtime import orchestrator

BROWSER_SWARM: dict[str, Any] = {
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
            "runtime": {"package_source": "stub-vision", "loader": "stub"},
        },
        {
            "id": "dom_sensor",
            "model": "convaiinnovations/laya",
            "modality": "text",
            "views": ["dom"],
            "default_questions": {
                "submit_button": {
                    "type": "choice",
                    "instructions": "Which element is the submit button?",
                    "criteria": {"a": "the input with value Send", "b": "the button labeled Send"},
                }
            },
            "runtime": {"package_source": "stub-text", "loader": "stub"},
        },
        {
            "id": "class_sensor",
            "model": "laya-typed-decisions",
            "modality": "text",
            "views": ["classes"],
            "default_questions": {
                "class_label": {
                    "type": "choice",
                    "instructions": "Which class label belongs to submit?",
                    "criteria": {"a": "btn btn-primary", "b": "card"},
                },
                "class_count": {
                    "type": "score",
                    "instructions": "How many class names are present?",
                    "criteria": ["few", "many"],
                },
            },
            "runtime": {"package_source": "stub-text", "loader": "stub"},
        },
    ],
}

OBS: dict[str, Any] = {
    "views": {
        "screenshot": "shot.png",
        "dom": "<html><button>Send</button></html>",
        "classes": "btn btn-primary",
    },
    "questions": {"dom_sensor": {"url": {"type": "noul", "instructions": "Is this a form page?"}}},
}


@pytest.fixture()
def swarm() -> manifest_mod.SwarmManifest:
    return manifest_mod.parse_manifest(BROWSER_SWARM)


def test_plan_cells_resolves_models_loaders_and_view_types(
    swarm: manifest_mod.SwarmManifest,
) -> None:
    plans = orchestrator.plan_cells(swarm, interpreter=sys.executable)
    assert [plan.model.id for plan in plans] == [
        "laya-vision",
        "laya-english",
        "laya-typed-decisions",
    ]
    assert [plan.loader for plan in plans] == ["stub", "stub", "stub"]
    assert [dict(plan.view_types) for plan in plans] == [
        {"screenshot": "image"},
        {"dom": "text"},
        {"classes": "text"},
    ]
    assert all(plan.warnings == () for plan in plans)


def test_plan_cells_warns_when_envs_are_not_built(swarm: manifest_mod.SwarmManifest) -> None:
    plans = orchestrator.plan_cells(swarm)
    for plan in plans:
        assert any("env is not built" in w and "decision-swarm install" in w for w in plan.warnings)


def test_browser_swarm_runs_end_to_end(swarm: manifest_mod.SwarmManifest) -> None:
    payload = query.parse_query(OBS, swarm)
    result = orchestrator.run(swarm, payload, interpreter=sys.executable)
    data = result.to_dict()

    assert set(data["cells"]) == {"screenshot_sensor", "dom_sensor", "class_sensor"}
    for cell_data in data["cells"].values():
        assert set(cell_data) == {"answers", "model", "latency_ms", "warnings"}
        assert cell_data["latency_ms"] >= 0.0

    # mixed questions answered with the right typed keys
    assert data["cells"]["screenshot_sensor"]["answers"]["submit_visible"] == {
        "noul": 0.5,
        "confidence": 1.0,
    }
    assert data["cells"]["screenshot_sensor"]["model"] == "thaitea/laya-vision"
    assert data["cells"]["dom_sensor"]["answers"]["submit_button"] == {"choice": "a", "confidence": 1.0}
    assert data["cells"]["class_sensor"]["answers"]["class_label"] == {"choice": "a", "confidence": 1.0}
    assert data["cells"]["class_sensor"]["answers"]["class_count"] == {"score": 0.0, "confidence": 1.0}

    # structural diversity: each cell saw ONLY its bound views (stub echoes them)
    assert data["cells"]["screenshot_sensor"]["warnings"] == ["views=screenshot"]
    assert data["cells"]["dom_sensor"]["warnings"] == ["views=dom"]
    assert data["cells"]["class_sensor"]["warnings"] == ["views=classes"]


def test_query_overrides_reach_workers(swarm: manifest_mod.SwarmManifest) -> None:
    payload = query.parse_query(OBS, swarm)
    result = orchestrator.run(swarm, payload, interpreter=sys.executable)
    answers = result.to_dict()["cells"]["dom_sensor"]["answers"]
    # the query-only question arrived; the default one still resolves
    assert set(answers) == {"submit_button", "url"}


def test_worker_failures_become_warnings(swarm: manifest_mod.SwarmManifest) -> None:
    broken = dict(BROWSER_SWARM)
    broken["cells"] = [dict(cell) for cell in BROWSER_SWARM["cells"]]
    broken["cells"][1]["runtime"] = {"package_source": "stub-text", "loader": "mystery"}
    payload = query.parse_query(OBS, swarm)

    result = orchestrator.run(manifest_mod.parse_manifest(broken), payload, interpreter=sys.executable)
    data = result.to_dict()["cells"]
    assert data["dom_sensor"]["answers"] == {}
    assert any("worker error" in warning for warning in data["dom_sensor"]["warnings"])
    assert data["screenshot_sensor"]["answers"]  # the rest of the swarm still answers


def test_swarm_stays_warm_across_queries(swarm: manifest_mod.SwarmManifest) -> None:
    payload = query.parse_query(OBS, swarm)
    plans = orchestrator.plan_cells(swarm, interpreter=sys.executable)
    with orchestrator.Swarm(plans) as warm:
        first = warm.query(payload).to_dict()
        second = warm.query(payload).to_dict()
    assert first["cells"].keys() == second["cells"].keys()
    assert second["cells"]["class_sensor"]["answers"]["class_count"] == {
        "score": 0.0,
        "confidence": 1.0,
    }
