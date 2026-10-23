"""Tests for packaging: bundled data resolution and frozen-executable paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from decision_swarm import cache, envs, pack, registry
from decision_swarm.runtime import orchestrator


def _fake_module_dir(tmp_path: Path) -> Path:
    module_dir = tmp_path / "site-packages" / "decision_swarm"
    module_dir.mkdir(parents=True)
    (module_dir / "registry.py").write_text("", encoding="utf-8")
    return module_dir


def test_registry_resolves_bundled_copy_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _fake_module_dir(tmp_path)
    bundled = module_dir / "data" / "registry" / "models.json"
    bundled.parent.mkdir(parents=True)
    bundled.write_text('{"schema_version": 1, "models": []}', encoding="utf-8")
    monkeypatch.setattr(registry, "__file__", str(module_dir / "registry.py"))
    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)

    assert registry.default_registry_path() == bundled


def test_registry_falls_back_to_repo_walk_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _fake_module_dir(tmp_path)
    repo_registry = tmp_path / "registry" / "models.json"
    repo_registry.parent.mkdir(parents=True)
    repo_registry.write_text('{"schema_version": 1, "models": []}', encoding="utf-8")
    monkeypatch.setattr(registry, "__file__", str(module_dir / "registry.py"))
    monkeypatch.delenv(registry.REGISTRY_ENV, raising=False)

    assert registry.default_registry_path() == repo_registry


def test_registry_env_override_still_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _fake_module_dir(tmp_path)
    override = tmp_path / "elsewhere" / "models.json"
    override.parent.mkdir(parents=True)
    override.write_text('{"schema_version": 1, "models": []}', encoding="utf-8")
    monkeypatch.setattr(registry, "__file__", str(module_dir / "registry.py"))
    monkeypatch.setenv(registry.REGISTRY_ENV, str(override))

    assert registry.default_registry_path() == override


def test_templates_resolve_bundled_copy_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_dir = _fake_module_dir(tmp_path)
    bundled = module_dir / "data" / "templates" / "runtime"
    bundled.mkdir(parents=True)
    monkeypatch.setattr(pack, "__file__", str(module_dir / "pack.py"))

    assert pack.default_templates_dir() == bundled


def test_worker_source_root_is_importable_in_dev() -> None:
    root = Path(orchestrator._worker_source_root())
    assert (root / "decision_swarm" / "__init__.py").is_file()


def test_worker_source_root_uses_frozen_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert orchestrator._worker_source_root() == str(tmp_path / "decision_swarm_src")


def test_child_env_puts_source_root_on_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import os

    monkeypatch.setattr(orchestrator, "_worker_source_root", lambda: "/src/root")
    monkeypatch.setenv("PYTHONPATH", "/pre/existing")
    env = orchestrator._child_env()
    assert env["PYTHONPATH"].split(os.pathsep)[0] == "/src/root"
    assert "/pre/existing" in env["PYTHONPATH"]


def test_models_dir_and_envs_dir_defaults_are_under_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ``Path.home()`` consults ``HOME`` on POSIX but ``USERPROFILE`` (then
    # ``HOMEDRIVE``/``HOMEPATH``) on Windows, so patch the lookup itself rather
    # than a single environment variable.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv(cache.MODELS_DIR_ENV, raising=False)
    monkeypatch.delenv(envs.ENVS_DIR_ENV, raising=False)

    assert cache.models_dir() == tmp_path / cache.DEFAULT_MODELS_DIR_NAME
    assert envs.envs_dir() == tmp_path / envs.ENVS_DIR_NAME
