"""Tests for the long-lived serve loop: JSONL decisions over stdio."""

from __future__ import annotations

import io
import json
import sys
from typing import Any

import pytest

from decision_swarm import cli, manifest as manifest_mod, pack, query
from decision_swarm.runtime import serve
from test_orchestrator import BROWSER_SWARM, OBS

CELL_IDS = {"screenshot_sensor", "dom_sensor", "class_sensor"}


class FakeSwarm:
    """Stands in for a warm ``orchestrator.Swarm`` inside the serve loop."""

    def __init__(
        self,
        swarm: manifest_mod.SwarmManifest,
        *,
        startup_warnings: tuple[str, ...] = (),
    ) -> None:
        self.manifest = swarm
        self.queries: list[query.QueryPayload] = []
        self._startup_warnings = startup_warnings

    def query(self, payload: query.QueryPayload) -> query.SwarmResult:
        self.queries.append(payload)
        return query.SwarmResult(
            cells={
                cell.id: query.CellResult(
                    cell_id=cell.id, model=cell.model, answers={}, latency_ms=0.5
                )
                for cell in self.manifest.cells
            }
        )

    def startup_warnings(self) -> tuple[str, ...]:
        return self._startup_warnings


@pytest.fixture()
def swarm() -> manifest_mod.SwarmManifest:
    return manifest_mod.parse_manifest(BROWSER_SWARM)


def _run(swarm: manifest_mod.SwarmManifest, text: str, **kwargs: Any) -> tuple[int, list[dict[str, Any]]]:
    service = FakeSwarm(swarm, **kwargs)
    stdout = io.StringIO()
    code = serve.serve_loop(service, swarm, stdin=io.StringIO(text), stdout=stdout)
    return code, [json.loads(line) for line in stdout.getvalue().splitlines()]


def _query(payload: dict[str, Any], message_id: Any = 1) -> str:
    return json.dumps({"type": "query", "id": message_id, **payload}) + "\n"


SHUTDOWN = '{"type": "shutdown"}\n'


def test_serve_emits_ready_then_answers_a_query(swarm: manifest_mod.SwarmManifest) -> None:
    code, lines = _run(swarm, _query(OBS) + SHUTDOWN)

    assert code == 0
    ready, result = lines
    assert ready == {
        "type": "ready",
        "name": "browser-swarm",
        "cells": [cell.id for cell in swarm.cells],
        "views": ["screenshot", "dom", "classes"],
        "warnings": [],
    }
    assert result["type"] == "result" and result["id"] == 1
    assert set(result["cells"]) == CELL_IDS


def test_serve_payload_is_the_run_input_plus_type_and_id(
    swarm: manifest_mod.SwarmManifest,
) -> None:
    """OBS is accepted verbatim: same views/question-override shape as run."""
    service = FakeSwarm(swarm)
    stdout = io.StringIO()
    serve.serve_loop(
        service, swarm, stdin=io.StringIO(_query(OBS) + SHUTDOWN), stdout=stdout
    )

    assert len(service.queries) == 1
    assert dict(service.queries[0].views) == OBS["views"]
    override = service.queries[0].questions["dom_sensor"]
    assert [question.name for question in override] == ["url"]


def test_serve_stays_warm_across_queries(swarm: manifest_mod.SwarmManifest) -> None:
    code, lines = _run(swarm, _query(OBS) + _query(OBS, message_id=2))

    assert code == 0  # end-of-input shuts down cleanly too
    ready, first, second = lines
    assert ready["type"] == "ready"
    assert (first["id"], second["id"]) == (1, 2)
    assert all(line["type"] == "result" for line in (first, second))


def test_serve_survives_bad_lines(swarm: manifest_mod.SwarmManifest) -> None:
    code, lines = _run(
        swarm,
        "{broken\n"
        + "[1, 2]\n"
        + '{"type": "mystery", "id": 4}\n'
        + json.dumps({"type": "query", "id": 5, "views": {"nope": "x"}})
        + "\n"
        + _query(OBS)
        + SHUTDOWN,
    )

    assert code == 0
    kinds = [line["type"] for line in lines]
    assert kinds == ["ready", "error", "error", "error", "error", "result"]
    assert "invalid JSON line" in lines[1]["message"]
    assert "must be a JSON object" in lines[2]["message"]
    assert "unknown message type" in lines[3]["message"]
    assert lines[3]["id"] == 4 and lines[4]["id"] == 5
    assert "unknown view" in lines[4]["message"]


def test_serve_rejects_non_integer_ids(swarm: manifest_mod.SwarmManifest) -> None:
    code, lines = _run(swarm, _query(OBS, message_id="x") + SHUTDOWN)

    assert code == 0
    assert lines[1]["type"] == "error"
    assert "'id' must be an integer" in lines[1]["message"]


def test_serve_accepts_queries_without_an_id(swarm: manifest_mod.SwarmManifest) -> None:
    code, lines = _run(swarm, _query(OBS, message_id=None) + SHUTDOWN)

    assert code == 0
    assert lines[1]["type"] == "result"
    assert lines[1]["id"] is None


def test_ready_carries_startup_warnings(swarm: manifest_mod.SwarmManifest) -> None:
    code, lines = _run(
        swarm,
        SHUTDOWN,
        startup_warnings=("cell 'dom_sensor': worker error: boom",),
    )

    assert code == 0
    assert lines[0]["warnings"] == ["cell 'dom_sensor': worker error: boom"]


def test_cli_serve_runs_a_warm_swarm_over_stdio(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: real workers load once, ready arrives warm, queries answer."""
    pack_dir = pack.build_pack(
        manifest_mod.parse_manifest(BROWSER_SWARM), tmp_path / "my-swarm", models_root=tmp_path / "cache"
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(_query(OBS) + SHUTDOWN),
    )

    code = cli.main(
        ["serve", "--pack", str(pack_dir), "--models-dir", str(tmp_path / "cache")]
    )

    assert code == 0
    ready, result = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert ready["type"] == "ready" and set(ready["cells"]) == CELL_IDS
    assert result["type"] == "result" and result["id"] == 1
    answers = result["cells"]["dom_sensor"]["answers"]
    assert answers["submit_button"] == {"choice": "a", "confidence": 1.0}
    assert set(answers) == {"submit_button", "url"}
    assert "views=screenshot" in result["cells"]["screenshot_sensor"]["warnings"]


def test_answer_returns_the_wire_response(swarm: manifest_mod.SwarmManifest) -> None:
    service = FakeSwarm(swarm)

    response = serve.answer(service, swarm, {"type": "query", "id": 9, **OBS})

    assert response["type"] == "result" and response["id"] == 9
    assert set(response["cells"]) == CELL_IDS


def test_answer_maps_errors_without_raising(swarm: manifest_mod.SwarmManifest) -> None:
    service = FakeSwarm(swarm)

    response = serve.answer(service, swarm, {"type": "query", "id": 3, "views": {"nope": "x"}})

    assert response["type"] == "error" and response["id"] == 3
    assert "unknown view" in response["message"]
    assert service.queries == []  # never reached the swarm


def test_answer_treats_shutdown_as_a_transport_concern(
    swarm: manifest_mod.SwarmManifest,
) -> None:
    response = serve.answer(FakeSwarm(swarm), swarm, {"type": "shutdown"})

    assert response["type"] == "error"
    assert "unknown message type" in response["message"]


def test_run_query_is_the_shared_validation_path(swarm: manifest_mod.SwarmManifest) -> None:
    message_id, result = serve.run_query(
        FakeSwarm(swarm), swarm, {"type": "query", "id": 11, **OBS}
    )

    assert message_id == 11
    assert set(result.cells) == CELL_IDS


def test_run_query_raises_on_a_bad_envelope(swarm: manifest_mod.SwarmManifest) -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        serve.run_query(FakeSwarm(swarm), swarm, [1, 2])
