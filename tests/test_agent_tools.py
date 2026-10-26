"""Tests for the agent tool layer: pack resolution, judging, authoring guards."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from decision_swarm import agent, manifest as manifest_mod, pack, query
from decision_swarm.registry import ModelEntry
from test_orchestrator import BROWSER_SWARM, OBS

ENTRIES: dict[str, ModelEntry] = {
    "laya-vision": ModelEntry(
        id="laya-vision",
        hf_repo="thaitea/laya-vision",
        revision="main",
        modality="image+text",
        package_source="stub-vision",
        loader="stub",
    ),
    "laya": ModelEntry(
        id="laya",
        hf_repo="convaiinnovations/laya",
        revision="main",
        modality="text",
        package_source="stub-text",
        loader="stub",
    ),
    "laya-typed-decisions": ModelEntry(
        id="laya-typed-decisions",
        hf_repo="convaiinnovations/laya",
        subfolder="typed-decisions",
        revision="main",
        modality="text",
        package_source="stub-text",
        loader="stub",
    ),
}


class FakeService:
    """Stands in for a warm ``orchestrator.Swarm``."""

    def __init__(self, swarm: manifest_mod.SwarmManifest) -> None:
        self.manifest = swarm
        self.queries: list[query.QueryPayload] = []

    def query(self, payload: query.QueryPayload) -> query.SwarmResult:
        self.queries.append(payload)
        cells = {
            cell.id: query.CellResult(
                cell_id=cell.id,
                model=cell.model,
                answers={
                    question.name: query.Answer(
                        name=question.name, type=question.type, value="a", confidence=0.75
                    )
                    for question in cell.default_questions
                },
                latency_ms=1.5,
            )
            for cell in self.manifest.cells
        }
        return query.SwarmResult(cells=cells)


class FakePool:
    """Stands in for ``WarmPool``; records what the tools asked for."""

    def __init__(self, service: FakeService, *, warm: bool = True) -> None:
        self.service = service
        self._warm = warm
        self.starts = 0
        self.closed = False

    def is_warm(self, pack_dir: Path) -> bool:
        return self._warm

    def get(self, pack_dir: Path, swarm: manifest_mod.SwarmManifest) -> FakeService:
        self.starts += 1
        return self.service

    def warm_packs(self) -> list[str]:
        return []

    def close(self) -> None:
        self.closed = True


def _config(**overrides: Any) -> agent.AgentConfig:
    defaults: dict[str, Any] = {
        "packs_dirs": (),
        "default_pack": None,
        "allow_unvetted": False,
        "max_warm": 1,
        "models_root": Path("/nonexistent-decision-swarm-cache"),
        "envs_root": None,
    }
    defaults.update(overrides)
    return agent.AgentConfig(**defaults)


def _write_pack(root: Path, name: str = "browser-swarm") -> Path:
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    manifest_mod.save_manifest(manifest_mod.parse_manifest(BROWSER_SWARM), dest)
    return dest


def _tools(config: agent.AgentConfig, *, warm: bool = True) -> tuple[agent.AgentTools, FakePool]:
    swarm = manifest_mod.parse_manifest(BROWSER_SWARM)
    pool = FakePool(FakeService(swarm), warm=warm)
    return agent.AgentTools(config, registry_entries=ENTRIES, pool=pool), pool


# --------------------------------------------------------------------------- #
# pack resolution
# --------------------------------------------------------------------------- #


def test_resolve_pack_accepts_dir_manifest_and_name(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "packs")
    config = _config(packs_dirs=(tmp_path / "packs",))

    assert agent.resolve_pack(pack_dir, config) == pack_dir.resolve()
    assert agent.resolve_pack(pack_dir / "swarm.json", config) == pack_dir.resolve()
    assert agent.resolve_pack("browser-swarm", config) == pack_dir.resolve()


def test_resolve_pack_names_the_available_packs_when_it_misses(tmp_path: Path) -> None:
    _write_pack(tmp_path / "packs", "browser-swarm")
    config = _config(packs_dirs=(tmp_path / "packs",))

    with pytest.raises(agent.AgentError) as excinfo:
        agent.resolve_pack("nope", config)

    message = str(excinfo.value)
    assert "pack 'nope' not found" in message
    assert "browser-swarm" in message


def test_resolve_pack_prefers_the_default_when_the_argument_is_omitted(tmp_path: Path) -> None:
    default = _write_pack(tmp_path / "packs", "default-swarm")
    other = _write_pack(tmp_path / "packs", "other-swarm")
    config = _config(packs_dirs=(tmp_path / "packs",), default_pack=default)

    assert agent.resolve_pack(None, config) == default.resolve()
    assert agent.resolve_pack(None, config) != other.resolve()


def test_resolve_pack_refuses_an_ambiguous_omission(tmp_path: Path) -> None:
    _write_pack(tmp_path / "packs", "a-swarm")
    _write_pack(tmp_path / "packs", "b-swarm")
    config = _config(packs_dirs=(tmp_path / "packs",))

    with pytest.raises(agent.AgentError) as excinfo:
        agent.resolve_pack(None, config)
    assert "several packs available" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# describe
# --------------------------------------------------------------------------- #


def test_describe_returns_the_schema_without_starting_a_swarm(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "packs")
    tools, pool = _tools(_config(packs_dirs=(tmp_path / "packs",)))

    payload = tools.describe({"pack": str(pack_dir)})

    assert pool.starts == 0
    assert payload["name"] == "browser-swarm"
    assert [view["name"] for view in payload["views"]] == ["screenshot", "dom", "classes"]
    assert [cell["id"] for cell in payload["cells"]] == [
        "screenshot_sensor",
        "dom_sensor",
        "class_sensor",
    ]
    choice = payload["cells"][1]["questions"][0]
    assert choice["name"] == "submit_button"
    assert choice["type"] == "choice"
    assert set(choice["criteria"]) == {"a", "b"}
    assert payload["models"][0]["cached"] is False


def test_describe_lists_packs_when_several_are_available(tmp_path: Path) -> None:
    _write_pack(tmp_path / "packs", "a-swarm")
    _write_pack(tmp_path / "packs", "b-swarm")
    tools, _pool = _tools(_config(packs_dirs=(tmp_path / "packs",)))

    payload = tools.describe({})

    assert sorted(Path(item["path"]).name for item in payload["packs"]) == [
        "a-swarm",
        "b-swarm",
    ]
    assert payload["default"] is None


def test_describe_needs_a_pack_configured() -> None:
    tools, _pool = _tools(_config())
    with pytest.raises(agent.AgentError) as excinfo:
        tools.describe({})
    assert "--packs-dir" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# judge
# --------------------------------------------------------------------------- #


def test_judge_defaults_to_the_compact_projection(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "packs")
    tools, pool = _tools(_config(default_pack=pack_dir))

    payload = tools.judge({"views": OBS["views"]})

    assert pool.starts == 1
    assert payload["pack"] == "browser-swarm"
    cell = payload["cells"]["dom_sensor"]
    assert cell["submit_button"] == {"value": "a", "confidence": 0.75}
    # compact drops the per-cell bookkeeping an agent does not need
    assert "latency_ms" not in cell
    assert "model" not in cell


def test_judge_full_keeps_model_and_latency(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "packs")
    tools, _pool = _tools(_config(default_pack=pack_dir))

    payload = tools.judge({"views": OBS["views"], "detail": "full"})

    cell = payload["cells"]["dom_sensor"]
    assert cell["model"] == "convaiinnovations/laya"
    assert cell["latency_ms"] == 1.5
    assert cell["answers"]["submit_button"]["confidence"] == 0.75


def test_judge_forwards_question_overrides(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "packs")
    tools, pool = _tools(_config(default_pack=pack_dir))

    tools.judge({"views": OBS["views"], "questions": OBS["questions"]})

    assert pool.service.queries[-1].questions["dom_sensor"][0].name == "url"


def test_judge_requires_a_non_empty_views_object(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "packs")
    tools, _pool = _tools(_config(default_pack=pack_dir))

    with pytest.raises(agent.AgentError):
        tools.judge({"views": {}})


def test_judge_points_at_fetch_when_models_are_missing(tmp_path: Path) -> None:
    pack_dir = _write_pack(tmp_path / "packs")
    tools, pool = _tools(_config(default_pack=pack_dir), warm=False)

    with pytest.raises(agent.AgentError) as excinfo:
        tools.judge({"views": OBS["views"]})

    assert "not in the cache" in str(excinfo.value)
    assert "'fetch'" in str(excinfo.value)
    assert pool.starts == 0


# --------------------------------------------------------------------------- #
# authoring: registry trust boundary
# --------------------------------------------------------------------------- #


def _manifest_with(overrides: Mapping[str, Any]) -> dict[str, Any]:
    document = json.loads(json.dumps(BROWSER_SWARM))
    document["cells"][0].update(overrides)
    return document


def test_build_reports_manifest_errors_with_a_path(tmp_path: Path) -> None:
    tools, _pool = _tools(_config())
    broken = json.loads(json.dumps(BROWSER_SWARM))
    broken["cells"][0]["id"] = "Not A Slug"

    payload = tools.build({"manifest": broken, "dest": str(tmp_path / "out")})

    assert payload["ok"] is False
    assert payload["errors"][0]["path"] == "cells[0].id"
    assert not (tmp_path / "out").exists()


def test_build_rejects_a_package_source_outside_the_registry(tmp_path: Path) -> None:
    tools, _pool = _tools(_config())
    manifest = _manifest_with(
        {"runtime": {"package_source": "git+https://example.invalid/evil", "loader": "stub"}}
    )

    payload = tools.build({"manifest": manifest, "dest": str(tmp_path / "out"), "pin": False})

    assert payload["ok"] is False
    assert payload["errors"][0]["path"] == "cells[0].runtime.package_source"
    assert "--allow-unvetted" in payload["errors"][0]["message"]
    assert not (tmp_path / "out").exists()


def test_build_rejects_a_model_outside_the_registry(tmp_path: Path) -> None:
    tools, _pool = _tools(_config())
    manifest = _manifest_with({"model": "someone/else-checkpoint"})

    payload = tools.build({"manifest": manifest, "dest": str(tmp_path / "out"), "pin": False})

    assert payload["ok"] is False
    assert payload["errors"][0]["path"] == "cells[0].model"


def test_allow_unvetted_lets_an_outside_source_through(tmp_path: Path) -> None:
    tools, _pool = _tools(_config(allow_unvetted=True))
    manifest = _manifest_with(
        {"runtime": {"package_source": "git+https://example.invalid/evil", "loader": "stub"}}
    )

    payload = tools.build({"manifest": manifest, "dest": str(tmp_path / "out"), "pin": False})

    assert payload["ok"] is True
    assert (tmp_path / "out" / "mcp.json").is_file()


def test_build_dry_run_writes_nothing(tmp_path: Path) -> None:
    tools, _pool = _tools(_config())

    payload = tools.build(
        {"manifest": BROWSER_SWARM, "dest": str(tmp_path / "out"), "dry_run": True, "pin": False}
    )

    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert payload["cells"] == ["screenshot_sensor", "dom_sensor", "class_sensor"]
    assert not (tmp_path / "out").exists()


def test_build_writes_a_pack_with_a_paste_ready_mcp_config(tmp_path: Path) -> None:
    tools, _pool = _tools(_config())

    payload = tools.build(
        {"manifest": BROWSER_SWARM, "dest": str(tmp_path / "out"), "pin": False}
    )

    assert payload["ok"] is True
    written = json.loads((tmp_path / "out" / "mcp.json").read_text(encoding="utf-8"))
    assert written == payload["mcp_config"]
    server = written["mcpServers"]["decision-swarm"]
    assert server["args"][:2] == ["mcp", "--packs-dir"]
    assert server["args"][-1] == str((tmp_path / "out").resolve())


def test_install_revalidates_the_pack_against_the_registry(tmp_path: Path) -> None:
    """A pack dir can be hand-edited after build; install must re-check it."""
    pack_dir = _write_pack(tmp_path / "packs")
    document = json.loads((pack_dir / "swarm.json").read_text(encoding="utf-8"))
    document["cells"][0]["runtime"]["package_source"] = "git+https://example.invalid/evil"
    (pack_dir / "swarm.json").write_text(json.dumps(document), encoding="utf-8")

    tools, _pool = _tools(_config(default_pack=pack_dir))

    with pytest.raises(agent.AgentError) as excinfo:
        tools.install({})
    assert "package_source" in str(excinfo.value)


def test_list_models_exposes_the_registry_and_cache_state() -> None:
    tools, _pool = _tools(_config())

    payload = tools.list_models({})
    by_id = {model["id"]: model for model in payload["models"]}

    assert by_id["laya"]["hf_repo"] == "convaiinnovations/laya"
    assert by_id["laya"]["package_source"] == "stub-text"
    assert by_id["laya"]["cached"] is False
    assert by_id["laya"]["custom"] is False


# --------------------------------------------------------------------------- #
# background jobs
# --------------------------------------------------------------------------- #


def test_job_status_explains_an_unknown_id() -> None:
    tools, _pool = _tools(_config())

    with pytest.raises(agent.AgentError) as excinfo:
        tools.job_status({"job_id": "export-9"})

    assert "not found" in str(excinfo.value)
    assert "re-run export" in str(excinfo.value)


def test_export_runs_in_the_background_and_reports_its_failure(tmp_path: Path) -> None:
    built = pack.build_pack(
        manifest_mod.parse_manifest(BROWSER_SWARM),
        tmp_path / "packs" / "browser-swarm",
        models_root=tmp_path / "cache",
    )
    tools, _pool = _tools(_config(default_pack=built))

    started = tools.export({})
    assert started["state"] == "running"

    deadline = time.time() + 5
    status = started
    while status["state"] == "running" and time.time() < deadline:
        time.sleep(0.02)
        status = tools.job_status({"job_id": started["job_id"]})

    assert status["state"] == "error"
    assert "cache" in status["message"]


def test_tools_expose_every_documented_tool() -> None:
    tools, _pool = _tools(_config())
    table = tools.tools()

    assert set(table) == {
        "describe",
        "judge",
        "list_models",
        "build",
        "fetch",
        "install",
        "export",
        "job_status",
    }
    assert table["judge"].schema["required"] == ["views"]
    assert table["build"].schema["required"] == ["manifest", "dest"]
