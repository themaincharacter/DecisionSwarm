"""Tests for per-package-source env planning and building."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from decision_swarm import envs, manifest as manifest_mod
from test_orchestrator import BROWSER_SWARM

NO_SOURCE: dict[str, Any] = {
    "schema_version": 1,
    "name": "custom-swarm",
    "input_schema": {"views": {"dom": {"type": "text"}}},
    "cells": [
        {
            "id": "a_cell",
            "model": "my-org/custom-a",
            "modality": "text",
            "views": ["dom"],
            "default_questions": {"q": {"type": "noul", "instructions": "Is it there?"}},
        },
        {
            "id": "b_cell",
            "model": "my-org/custom-b",
            "modality": "text",
            "views": ["dom"],
            "default_questions": {"q": {"type": "noul", "instructions": "Is it there?"}},
        },
    ],
}


def test_env_key_is_stable_and_filesystem_safe() -> None:
    source = "git+https://github.com/r33drichards/laya-vision"
    key = envs.env_key(source)
    assert key == envs.env_key(source)
    assert key != envs.env_key("laya>=0.3.20")
    assert "/" not in key and ":" not in key and "+" not in key


def test_cell_envs_dedup_by_package_source(tmp_path: Path) -> None:
    swarm = manifest_mod.parse_manifest(BROWSER_SWARM)
    planned = envs.cell_envs(swarm, envs_root=tmp_path)

    assert len(planned) == 2  # stub-vision + stub-text (shared by two cells)
    by_source = {env.package_source: env for env in planned}
    assert by_source["stub-vision"].shared_by == ("screenshot_sensor",)
    assert by_source["stub-text"].shared_by == ("dom_sensor", "class_sensor")

    for env in planned:
        assert env.path == tmp_path / env.key
        assert env.python.parent in (env.path / "bin", env.path / "Scripts")


def test_build_env_installs_the_package_source(tmp_path: Path) -> None:
    swarm = manifest_mod.parse_manifest(BROWSER_SWARM)
    planned = {env.package_source: env for env in envs.cell_envs(swarm, envs_root=tmp_path)}
    target = planned["stub-text"]

    calls: list[tuple[Path, list[str]]] = []

    def fake_installer(python: Path, requirements: Any) -> None:
        calls.append((python, list(requirements)))

    envs.build_env(target, installer=fake_installer, with_pip=False)

    assert target.python.exists()
    assert calls == [(target.python, ["stub-text"])]


def test_cells_without_a_package_source_are_rejected() -> None:
    swarm = manifest_mod.parse_manifest(NO_SOURCE)
    with pytest.raises(envs.EnvError, match="no package_source"):
        envs.cell_envs(swarm, envs_root=sys.prefix)  # root never used
