"""Tests for the install command: env building and run-with-envs wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from decision_swarm import cli, envs, manifest as manifest_mod, pack
from test_orchestrator import BROWSER_SWARM, OBS


@pytest.fixture()
def pack_dir(tmp_path: Path) -> Path:
    swarm = manifest_mod.parse_manifest(BROWSER_SWARM)
    return pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")


@pytest.fixture()
def obs_path(tmp_path: Path) -> Path:
    path = tmp_path / "obs.json"
    path.write_text(json.dumps(OBS), encoding="utf-8")
    return path


@pytest.fixture()
def fast_env_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Path, list[str]]]:
    """Skip real venv creation; record installer calls."""
    calls: list[tuple[Path, list[str]]] = []

    def fake_ensure_venv(env: envs.CellEnv, *, with_pip: bool = True) -> Path:
        env.python.parent.mkdir(parents=True, exist_ok=True)
        env.python.touch()
        return env.python

    def fake_installer(python: Path, requirements: Any) -> None:
        calls.append((python, list(requirements)))

    monkeypatch.setattr(envs, "ensure_venv", fake_ensure_venv)
    monkeypatch.setattr(envs, "default_installer", fake_installer)
    return calls


def test_install_builds_one_env_per_package_source(
    pack_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fast_env_builder: list[tuple[Path, list[str]]],
) -> None:
    assert (
        cli.main(
            ["install", "--pack", str(pack_dir), "--envs-dir", str(tmp_path / "envs")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "built  stub-vision" in out and "cells: screenshot_sensor" in out
    assert "built  stub-text" in out and "cells: dom_sensor, class_sensor" in out

    # one install per unique package source, not per cell
    assert sorted(requirements for _python, requirements in fast_env_builder) == [
        ["stub-text"],
        ["stub-vision"],
    ]


def test_install_marks_existing_envs_as_reused(
    pack_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    fast_env_builder: list[tuple[Path, list[str]]],
) -> None:
    argv = ["install", "--pack", str(pack_dir), "--envs-dir", str(tmp_path / "envs")]
    cli.main(argv)
    capsys.readouterr()
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("reused") == 2
    assert "built" not in out


def test_install_reports_installer_failures(
    pack_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    fast_env_builder: list[tuple[Path, list[str]]],
) -> None:
    def failing_installer(python: Path, requirements: Any) -> None:
        raise envs.EnvError("pip exploded")

    monkeypatch.setattr(envs, "default_installer", failing_installer)
    assert (
        cli.main(["install", "--pack", str(pack_dir), "--envs-dir", str(tmp_path / "envs")])
        == 1
    )
    assert "pip exploded" in capsys.readouterr().err


def test_install_then_run_uses_env_pythons(
    pack_dir: Path,
    obs_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration: real venvs (offline, fake pip install), then run on them."""
    installs: list[list[str]] = []

    def fake_installer(python: Path, requirements: Any) -> None:
        installs.append(list(requirements))

    monkeypatch.setattr(envs, "default_installer", fake_installer)
    envs_root = tmp_path / "envs"
    assert cli.main(["install", "--pack", str(pack_dir), "--envs-dir", str(envs_root)]) == 0
    capsys.readouterr()
    assert sorted(installs) == [["stub-text"], ["stub-vision"]]

    assert (
        cli.main(
            [
                "run",
                "--pack",
                str(pack_dir),
                "--input",
                str(obs_path),
                "--models-dir",
                str(tmp_path / "cache"),
                "--envs-dir",
                str(envs_root),
            ]
        )
        == 0
    )
    data = json.loads(capsys.readouterr().out)
    for cell_data in data["cells"].values():
        assert not any("env is not built" in warning for warning in cell_data["warnings"])
    assert "views=dom" in data["cells"]["dom_sensor"]["warnings"]
