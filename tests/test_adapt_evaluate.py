"""Tests for the adapt evaluate gate (pure) and its CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_swarm import cli
from decision_swarm.adapt import evaluate as adapt_evaluate


def _slice(accuracy: float, baseline: float | None, items: int = 100) -> dict:
    return {"items": items, "accuracy": accuracy, "baseline_accuracy": baseline}


def test_target_gate_passes_on_min_delta() -> None:
    result = adapt_evaluate.check_gates(
        {"all": _slice(0.80, 0.70), "x": _slice(0.90, 0.80)},
        target_slice="x",
        guard={"items": 30, "accuracy": 0.70, "baseline_accuracy": 0.72},
    )
    assert result["pass"] is True
    names = [gate["name"] for gate in result["gates"]]
    assert names == ["target", "no-regression", "must-not-regress"]


def test_target_gate_fails_below_min_delta() -> None:
    result = adapt_evaluate.check_gates(
        {"x": _slice(0.82, 0.80)},
        target_slice="x",
        guard={"items": 30, "accuracy": 0.72, "baseline_accuracy": 0.72},
    )
    assert result["pass"] is False
    assert result["gates"][0]["passed"] is False
    assert "delta +0.0200" in result["gates"][0]["detail"]


def test_no_regression_gate_fails_on_a_big_drop() -> None:
    result = adapt_evaluate.check_gates(
        {"x": _slice(0.90, 0.80), "other": _slice(0.60, 0.70)},
        target_slice="x",
        guard={"items": 30, "accuracy": 0.72, "baseline_accuracy": 0.72},
    )
    assert result["pass"] is False
    regression = next(g for g in result["gates"] if g["name"] == "no-regression")
    assert regression["passed"] is False
    assert "other -0.1000" in regression["detail"]


def test_guard_gate_fails_closed_without_a_baseline() -> None:
    result = adapt_evaluate.check_gates(
        {"x": _slice(0.90, 0.80)},
        target_slice="x",
        guard={"items": 30, "accuracy": 0.50, "baseline_accuracy": None},
    )
    guard = next(g for g in result["gates"] if g["name"] == "must-not-regress")
    assert guard["passed"] is False
    assert result["pass"] is False


def test_missing_target_slice_fails() -> None:
    result = adapt_evaluate.check_gates(
        {"all": _slice(0.90, 0.80)},
        target_slice="x",
        guard={"items": 30, "accuracy": 0.72, "baseline_accuracy": 0.72},
    )
    assert result["gates"][0]["passed"] is False
    assert "has no eval items" in result["gates"][0]["detail"]


def test_default_guard_path_ships_with_the_repo() -> None:
    path = adapt_evaluate.default_guard_path()
    assert path.is_file()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 30
    assert all(row["tags"] == ["guard"] for row in rows)


# ----------------------------------------------------------------- CLI


def test_evaluate_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["adapt", "evaluate", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for option in ("--merged", "--data", "--slice", "--baseline", "--min-delta", "--max-regression", "--guard"):
        assert option in out


def test_evaluate_requires_a_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from decision_swarm.registry import ModelEntry

    monkeypatch.setattr(
        cli.registry,
        "get_entry",
        lambda model, registry=None: ModelEntry(id=model, hf_repo="x/y", package_source="laya", loader="load"),
    )
    monkeypatch.setattr(cli.cache, "models_dir", lambda override=None: tmp_path / "models")
    code = cli.main(
        ["adapt", "evaluate", "--base", "laya-english", "--merged", str(tmp_path / "m"), "--data", "eval.jsonl"]
    )
    assert code == 1


def test_evaluate_runs_the_env_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from decision_swarm.registry import ModelEntry

    report_path = tmp_path / "evaluate_report.json"
    captured: dict = {}

    def fake_run(python, module, config):
        captured.update(config=config, module=module)
        report_path.write_text(json.dumps({"pass": True, "gates": []}), encoding="utf-8")
        return 0

    monkeypatch.setattr(
        cli.registry,
        "get_entry",
        lambda model, registry=None: ModelEntry(id=model, hf_repo="x/y", package_source="laya", loader="load"),
    )
    monkeypatch.setattr(cli, "_adapt_prepare_env", lambda entry, envs_root=None: type("E", (), {"python": Path("/p")})())
    monkeypatch.setattr(cli, "_run_adapt_module", fake_run)

    code = cli.main(
        [
            "adapt", "evaluate",
            "--base", "laya-english",
            "--merged", str(tmp_path / "merged"),
            "--data", "eval.jsonl",
            "--slice", "submit",
            "--base-dir", str(tmp_path / "base"),
            "--out", str(report_path),
            "--json",
        ]
    )
    assert code == 0
    assert captured["module"] == "decision_swarm.adapt.evaluate"
    assert captured["config"]["slice"] == "submit"
    assert captured["config"]["baseline_dir"] == str(tmp_path / "base")
    assert captured["config"]["guard"].endswith("must-not-regress.jsonl")
    assert json.loads(capsys.readouterr().out)["pass"] is True
