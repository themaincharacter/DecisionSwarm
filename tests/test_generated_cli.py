"""End-to-end tests for the CLI generated into a pack's runtime package."""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from decision_swarm import manifest as manifest_mod, pack
from test_orchestrator import BROWSER_SWARM, OBS

CELL_IDS = {"screenshot_sensor", "dom_sensor", "class_sensor"}
SHUTDOWN = '{"type": "shutdown"}\n'


@pytest.fixture()
def pack_dir(tmp_path: Path) -> Path:
    swarm = manifest_mod.parse_manifest(BROWSER_SWARM)
    return pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")


def _generated_cli(pack_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(pack_dir / "runtime" / "src"))
    for name in ("browser_swarm.cli", "browser_swarm"):
        sys.modules.pop(name, None)
    return importlib.import_module("browser_swarm.cli")


def test_generated_run_answers_one_observation(
    pack_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _generated_cli(pack_dir, monkeypatch)
    obs_path = tmp_path / "obs.json"
    obs_path.write_text(json.dumps(OBS), encoding="utf-8")

    code = mod.main(["run", "--input", str(obs_path), "--models-dir", str(tmp_path / "cache")])

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data["cells"]) == CELL_IDS
    assert data["cells"]["dom_sensor"]["answers"]["submit_button"] == {"choice": "a", "confidence": 1.0}
    assert set(data["cells"]["dom_sensor"]["answers"]) == {"submit_button", "url"}


def test_generated_serve_warms_up_and_answers(
    pack_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _generated_cli(pack_dir, monkeypatch)
    request = json.dumps({"type": "query", "id": 1, **OBS}) + "\n"
    monkeypatch.setattr(sys, "stdin", io.StringIO(request + SHUTDOWN))

    code = mod.main(["serve", "--models-dir", str(tmp_path / "cache")])

    assert code == 0
    ready, result = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert ready["type"] == "ready" and set(ready["cells"]) == CELL_IDS
    assert result["type"] == "result" and result["id"] == 1
    assert set(result["cells"]) == CELL_IDS
    assert result["cells"]["screenshot_sensor"]["answers"]["submit_visible"] == {
        "noul": 0.5,
        "confidence": 1.0,
    }


def test_generated_run_reports_bad_payloads(
    pack_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _generated_cli(pack_dir, monkeypatch)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"views": {"nope": "x"}}), encoding="utf-8")

    code = mod.main(["run", "--input", str(bad), "--models-dir", str(tmp_path / "cache")])

    assert code == 1
    assert "error" in capsys.readouterr().err
