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


@pytest.fixture(autouse=True)
def _no_hub_lookups(monkeypatch: pytest.MonkeyPatch) -> None:
    """`build` pins revisions by asking the Hub; tests must stay offline."""
    monkeypatch.setattr(pack, "hub_commit_resolver", lambda entry: "0" * 40)


def _build(dest: Path, manifest_file: Path, *extra: str) -> int:
    return cli.main(
        [
            "build",
            "--manifest",
            str(manifest_file),
            "--dest",
            str(dest),
            "--models-dir",
            str(dest.parent / "cache"),
            *extra,
        ]
    )


def test_build_pins_each_revision_to_a_commit(
    manifest_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pack, "hub_commit_resolver", lambda entry: "abc123")
    dest = tmp_path / "my-swarm"

    assert _build(dest, manifest_file, "--json") == 0
    lock = json.loads((dest / "models.lock.json").read_text(encoding="utf-8"))

    assert [model["commit"] for model in lock["models"]] == ["abc123"] * 3
    assert {model["revision"] for model in lock["models"]} == {"main"}
    summary = json.loads(capsys.readouterr().out)
    assert {model["commit"] for model in summary["models"]} == {"abc123"}


def test_build_no_pin_skips_the_hub_lookup(
    manifest_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(entry: object) -> str:
        raise AssertionError("--no-pin must not resolve revisions")

    monkeypatch.setattr(pack, "hub_commit_resolver", boom)
    dest = tmp_path / "my-swarm"

    assert _build(dest, manifest_file, "--no-pin") == 0
    lock = json.loads((dest / "models.lock.json").read_text(encoding="utf-8"))
    assert all(model["commit"] is None for model in lock["models"])


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


def test_build_bundles_the_pack_and_weights_by_default(
    manifest_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    downloader = FakeDownloader()
    monkeypatch.setattr(cache, "hub_download", downloader)
    swarm = manifest_mod.parse_manifest(BROWSER_SWARM)
    for entry in pack.unique_entries(swarm):
        cache.fetch(entry, root=tmp_path / "cache", downloader=downloader)
    dest = tmp_path / "my-swarm"

    assert _build(dest, manifest_file, "--json") == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["bundle"] == str(tmp_path / "my-swarm.swarmpack.tar")
    assert (tmp_path / "my-swarm.swarmpack.tar").is_file()


def test_build_without_export_writes_only_the_pack(
    manifest_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dest = tmp_path / "my-swarm"

    assert _build(dest, manifest_file, "--no-export") == 0

    assert dest.is_dir()
    assert not (tmp_path / "my-swarm.swarmpack.tar").exists()


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
