"""End-to-end CLI tests for run: the browser-control swarm through the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_swarm import cache, cli, manifest as manifest_mod, pack
from test_cache import FakeDownloader
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


def _run_argv(pack_dir: Path, obs_path: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--pack",
        str(pack_dir),
        "--input",
        str(obs_path),
        "--models-dir",
        str(tmp_path / "cache"),
        *extra,
    ]


def test_run_returns_the_result_schema(
    pack_dir: Path, obs_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_run_argv(pack_dir, obs_path, tmp_path)) == 0
    data = json.loads(capsys.readouterr().out)

    assert set(data["cells"]) == {"screenshot_sensor", "dom_sensor", "class_sensor"}
    answers = data["cells"]["screenshot_sensor"]["answers"]
    assert answers["submit_visible"] == {"noul": 0.5, "confidence": 1.0}
    assert data["cells"]["dom_sensor"]["answers"]["submit_button"] == {"choice": "a", "confidence": 1.0}
    assert set(data["cells"]["dom_sensor"]["answers"]) == {"submit_button", "url"}

    # each cell's stub echo shows it saw only its bound views
    assert "views=screenshot" in data["cells"]["screenshot_sensor"]["warnings"]
    assert "views=dom" in data["cells"]["dom_sensor"]["warnings"]
    assert "views=classes" in data["cells"]["class_sensor"]["warnings"]


def test_run_pretty_output(
    pack_dir: Path, obs_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_run_argv(pack_dir, obs_path, tmp_path, "--pretty")) == 0
    out = capsys.readouterr().out
    assert "screenshot_sensor  (thaitea/laya-vision" in out
    assert "submit_visible: noul=0.5  confidence=1.00" in out
    assert "warning: views=dom" in out


def test_run_offline_requires_a_warm_cache(
    pack_dir: Path, obs_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_run_argv(pack_dir, obs_path, tmp_path, "--offline")) == 1
    err = capsys.readouterr().err
    assert "offline" in err and "not in the cache" in err


def test_run_offline_succeeds_when_cached(
    pack_dir: Path, obs_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    swarm = manifest_mod.parse_manifest(BROWSER_SWARM)
    models_root = tmp_path / "cache"
    downloader = FakeDownloader()
    for entry in pack.unique_entries(swarm):
        cache.fetch(entry, root=models_root, downloader=downloader)

    assert cli.main(_run_argv(pack_dir, obs_path, tmp_path, "--offline")) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["cells"]) == {"screenshot_sensor", "dom_sensor", "class_sensor"}


def test_run_reports_bad_payloads(
    pack_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"views": {"nope": "x"}}), encoding="utf-8")
    assert cli.main(_run_argv(pack_dir, bad, tmp_path)) == 1
    assert "error" in capsys.readouterr().err


def test_cli_worker_hidden_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """Frozen launchers spawn themselves as workers via the hidden route."""
    import io
    import sys

    from decision_swarm.runtime import protocol

    config = json.dumps({"loader": "stub", "model": {"model_id": "x"}, "view_types": {}})
    request = protocol.encode(protocol.QueryMessage(id=1, views={"dom": "<html>"}, questions={}))
    stdin = io.StringIO(request + protocol.encode(protocol.ShutdownMessage()))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)

    assert cli.main(["_worker", "--config", config]) == 0
    stdout.seek(0)
    ready = protocol.read(stdout)
    assert isinstance(ready, protocol.ReadyMessage)
    response = protocol.read(stdout)
    assert isinstance(response, protocol.ResultMessage)
    assert response.id == 1
