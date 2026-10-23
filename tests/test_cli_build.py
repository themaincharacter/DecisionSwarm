"""Tests for the build and models commands (the GUI's engine plumbing)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_swarm import cache, cli, manifest as manifest_mod, pack
from test_cache import FakeDownloader
from test_orchestrator import BROWSER_SWARM, OBS


@pytest.fixture()
def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "swarm.json"
    path.write_text(json.dumps(BROWSER_SWARM), encoding="utf-8")
    return path


def test_build_creates_a_runnable_pack(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "my-swarm"
    assert (
        cli.main(
            [
                "build",
                "--manifest",
                str(manifest_file),
                "--dest",
                str(dest),
                "--models-dir",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    assert "built" in capsys.readouterr().out
    for relative in ("swarm.json", "models.lock.json", "README.md", "runtime/pyproject.toml"):
        assert (dest / relative).is_file()
    assert manifest_mod.load_manifest(dest).name == "browser-swarm"


def test_build_json_summary_matches_the_cells(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "my-swarm"
    assert (
        cli.main(
            [
                "build",
                "--manifest",
                str(manifest_file),
                "--dest",
                str(dest),
                "--models-dir",
                str(tmp_path / "cache"),
                "--json",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["name"] == "browser-swarm"
    assert summary["cells"] == ["screenshot_sensor", "dom_sensor", "class_sensor"]
    assert len(summary["models"]) == 3
    assert {model["id"] for model in summary["models"]} == {
        "laya-vision",
        "laya-english",
        "laya-typed-decisions",
    }
    assert all(model["cached"] is False for model in summary["models"])


def test_build_reports_invalid_manifests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "swarm.json"
    bad.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    assert cli.main(["build", "--manifest", str(bad), "--dest", str(tmp_path / "out")]) == 1
    assert "error" in capsys.readouterr().err


def test_models_lists_the_registry_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    by_id = {model["id"]: model for model in data["models"]}
    assert "laya-vision" in by_id
    assert "CC BY-NC-SA 4.0" in by_id["laya-vision"]["license"]
    assert by_id["laya-english"]["hf_repo"] == "convaiinnovations/laya"


def test_models_plain_output(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "laya-vision" in out and "laya-multilingual" in out
    assert "CC BY-NC-SA 4.0" in out


def test_fetch_accepts_a_manifest_without_a_pack(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GUI's Download path: fetch straight from a manifest."""
    downloader = FakeDownloader()
    monkeypatch.setattr(cache, "hub_download", downloader)
    argv = [
        "fetch",
        "--manifest",
        str(manifest_file),
        "--models-dir",
        str(tmp_path / "cache"),
    ]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.count("downloaded") == 3
    assert len(downloader.calls) == 3
