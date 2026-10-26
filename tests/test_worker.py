"""Tests for the per-cell worker: config, loaders, and a live stdio session."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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


class _FakeLaya:
    """Stands in for the ``laya`` package so loaders can be tested without torch."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def load(self, target: str, **kwargs: object) -> object:
        self.calls.append((target, kwargs))
        return object()

    def load_vlm(self, target: str, **kwargs: object) -> object:
        self.calls.append((target, kwargs))
        return object()


def _cached_snapshot(root: Path, subfolder: str | None = None) -> Path:
    """A cache snapshot with a root checkpoint and, optionally, one subfolder."""
    snapshot = root / "convaiinnovations" / "laya" / "main"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    if subfolder:
        target = snapshot / subfolder
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
    return snapshot


def test_model_spec_resolves_the_cached_snapshot(tmp_path: Path) -> None:
    snapshot = _cached_snapshot(tmp_path, "typed-decisions")

    root = loaders.ModelSpec(
        model_id="laya-english", hf_repo="convaiinnovations/laya", models_dir=str(tmp_path)
    )
    sub = loaders.ModelSpec(
        model_id="laya-typed-decisions",
        hf_repo="convaiinnovations/laya",
        subfolder="typed-decisions",
        models_dir=str(tmp_path),
    )

    assert root.local_path == snapshot
    assert sub.local_path == snapshot / "typed-decisions"


def test_model_spec_ignores_an_incomplete_snapshot(tmp_path: Path) -> None:
    (tmp_path / "convaiinnovations" / "laya" / "main").mkdir(parents=True)
    spec = loaders.ModelSpec(
        model_id="laya-english", hf_repo="convaiinnovations/laya", models_dir=str(tmp_path)
    )
    assert spec.local_path is None  # no config.json -> not a usable checkpoint


def test_model_spec_has_no_local_path_without_a_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = loaders.ModelSpec(model_id="convaiinnovations/laya")
    assert spec.local_path is None


def test_loaders_prefer_the_cached_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _cached_snapshot(tmp_path)
    fake = _FakeLaya()
    monkeypatch.setitem(sys.modules, "laya", fake)
    spec = loaders.ModelSpec(
        model_id="laya-english", hf_repo="convaiinnovations/laya", models_dir=str(tmp_path)
    )

    loaders.load_laya(spec, {})
    loaders.load_laya_vlm(spec, {})

    assert fake.calls == [(str(snapshot), {}), (str(snapshot), {})]


def test_loaders_fall_back_to_the_hub_when_uncached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeLaya()
    monkeypatch.setitem(sys.modules, "laya", fake)
    spec = loaders.ModelSpec(
        model_id="convaiinnovations/laya", subfolder="typed-decisions", models_dir=str(tmp_path)
    )

    loaders.load_laya(spec, {})
    loaders.load_laya_vlm(spec, {})

    assert fake.calls == [
        ("convaiinnovations/laya", {"subfolder": "typed-decisions"}),
        ("convaiinnovations/laya", {"revision": "main"}),
    ]
