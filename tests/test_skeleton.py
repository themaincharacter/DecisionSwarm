"""Smoke tests: the package skeleton imports and its constants match the plan."""

from __future__ import annotations

import importlib

import pytest

from decision_swarm import manifest

EXPECTED_MODULES = [
    "decision_swarm",
    "decision_swarm.manifest",
    "decision_swarm.registry",
    "decision_swarm.cache",
    "decision_swarm.pack",
    "decision_swarm.envs",
    "decision_swarm.cli",
    "decision_swarm.runtime",
    "decision_swarm.runtime.protocol",
    "decision_swarm.runtime.worker",
    "decision_swarm.runtime.orchestrator",
]


@pytest.mark.parametrize("module", EXPECTED_MODULES)
def test_modules_import(module: str) -> None:
    importlib.import_module(module)


def test_manifest_constants_match_the_plan() -> None:
    assert manifest.SCHEMA_VERSION == 1
    assert (manifest.MIN_CELLS, manifest.MAX_CELLS) == (2, 8)
    assert set(manifest.QUESTION_TYPES) == {"choice", "score", "noul"}
    assert set(manifest.VIEW_TYPES) == {"image", "text"}


def test_cell_id_pattern_accepts_slugs() -> None:
    import re

    assert re.match(manifest.CELL_ID_PATTERN, "screenshot_sensor")
    assert re.match(manifest.CELL_ID_PATTERN, "dom-2")
    assert not re.match(manifest.CELL_ID_PATTERN, "Dom Sensor")
