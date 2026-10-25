"""Tests for the per-cell worker: config, loaders, and a live stdio session."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from decision_swarm import manifest as manifest_mod
from decision_swarm.runtime import loaders, protocol, worker


def _config(loader: str = "stub") -> str:
    return json.dumps(
        {
            "loader": loader,
            "model": {"model_id": "stub-model", "hf_repo": "example/stub", "revision": "main"},
            "view_types": {"screenshot": "image", "dom": "text"},
        }
    )


def _questions() -> dict[str, manifest_mod.QuestionSpec]:
    raw = {
        "submit_visible": {"type": "noul", "instructions": "Do you see a submit button?"},
        "submit_button": {
            "type": "choice",
            "instructions": "Which element submits?",
            "criteria": {"a": "the input", "b": "the button"},
        },
        "dom_size": {"type": "score", "instructions": "How big is the DOM?", "criteria": ["small", "large"]},
    }
    return {
        name: manifest_mod.parse_question(name, spec, f"questions.{name}") for name, spec in raw.items()
    }


def _spawn(config: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "decision_swarm.runtime.worker", "--config", config],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_config_parses_and_round_trips() -> None:
    config = worker.WorkerConfig.from_json(_config())
    assert config.loader == "stub"
    assert config.model.model_id == "stub-model" and config.model.revision == "main"
    assert dict(config.view_types) == {"screenshot": "image", "dom": "text"}


def test_config_rejects_bad_shapes() -> None:
    with pytest.raises(loaders.LoaderError, match="not valid JSON"):
        worker.WorkerConfig.from_json("{nope")
    with pytest.raises(loaders.LoaderError, match="loader"):
        worker.WorkerConfig.from_json(json.dumps({"model": {}}))


def test_unknown_loader_is_rejected() -> None:
    with pytest.raises(loaders.LoaderError, match="unknown loader"):
        loaders.build_predictor("mystery", loaders.ModelSpec(model_id="x"))


def test_stub_predictor_is_deterministic_and_echoes_views() -> None:
    predictor = loaders.build_predictor("stub", loaders.ModelSpec(model_id="x"))
    answers, warnings = predictor.predict(
        {"screenshot": "shot.png", "dom": "<html>"}, _questions()
    )
    assert answers["submit_visible"] == {"noul": 0.5, "confidence": 1.0}
    assert answers["submit_button"] == {"choice": "a", "confidence": 1.0}
    assert answers["dom_size"] == {"score": 0.0, "confidence": 1.0}
    assert warnings == ("views=dom,screenshot",)


def test_worker_session_end_to_end() -> None:
    process = _spawn(_config())
    assert process.stdin is not None and process.stdout is not None

    ready = protocol.read(process.stdout)
    assert isinstance(ready, protocol.ReadyMessage)

    protocol.write(
        process.stdin,
        protocol.QueryMessage(
            id=1,
            views={"screenshot": "shot.png", "dom": "<html>"},
            questions=_questions(),
        ),
    )
    response = protocol.read(process.stdout)
    assert isinstance(response, protocol.ResultMessage)
    assert response.id == 1
    assert set(response.answers) == {"submit_visible", "submit_button", "dom_size"}
    assert response.answers["submit_button"] == {"choice": "a", "confidence": 1.0}
    # the stub echoes the views it received: exactly the ones sent
    assert response.warnings == ("views=dom,screenshot",)
    assert response.latency_ms >= 0.0

    protocol.write(process.stdin, protocol.ShutdownMessage())
    assert process.wait(timeout=10) == 0


def test_worker_survives_bad_lines() -> None:
    process = _spawn(_config())
    assert process.stdin is not None and process.stdout is not None

    assert isinstance(protocol.read(process.stdout), protocol.ReadyMessage)

    process.stdin.write("{broken\n")
    process.stdin.flush()
    first = protocol.read(process.stdout)
    assert isinstance(first, protocol.ErrorMessage) and first.id is None

    protocol.write(
        process.stdin,
        protocol.QueryMessage(id=2, views={"dom": "<html>"}, questions=_questions()),
    )
    second = protocol.read(process.stdout)
    assert isinstance(second, protocol.ResultMessage) and second.id == 2

    protocol.write(process.stdin, protocol.ShutdownMessage())
    assert process.wait(timeout=10) == 0


def test_worker_reports_config_errors_and_exits_nonzero() -> None:
    process = _spawn(json.dumps({"loader": "mystery", "model": {"model_id": "x"}}))
    assert process.stdout is not None
    message = protocol.read(process.stdout)
    assert isinstance(message, protocol.ErrorMessage)
    assert "unknown loader" in message.message
    assert process.wait(timeout=10) == 1
