"""Tests for the ``decision-swarm adapt`` command surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_swarm import cli
from decision_swarm.registry import ModelEntry

CHOICE = {"type": "choice", "instructions": "Which department?", "criteria": {"billing": "Billing", "tech": "Tech"}}


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_adapt_help_lists_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["adapt", "--help"])
    assert excinfo.value.code == 0
    assert "validate" in capsys.readouterr().out


def test_adapt_validate_help_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["adapt", "validate", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for option in ("--data", "--eval", "--base", "--json", "--text-column", "--label-column"):
        assert option in out


def test_validate_reports_stats_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = _write_jsonl(
        tmp_path / "train.jsonl",
        [
            {"state": "a ticket", "questions": {"q": CHOICE}, "expected": {"q": "billing"}},
            {"state": "another ticket", "questions": {"q": CHOICE}, "expected": {"q": "tech"}},
        ],
    )
    assert cli.main(["adapt", "validate", "--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "items                2" in out
    assert "optimizer budget: 2 train item(s)" in out  # int(2 * 0.1) == 0 calibration holdout
    assert "billing 1 · tech 1" in out


def test_validate_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"state": "a ticket", "questions": {"q": CHOICE}, "expected": {"q": "billing"}}],
    )
    assert cli.main(["adapt", "validate", "--data", str(data), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["train"]["items"] == 1
    assert payload["leaked_eval_items"] == 0
    assert payload["budget"]


def test_validate_reports_leakage(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    shared = {"state": "a ticket", "questions": {"q": CHOICE}, "expected": {"q": "billing"}}
    train = _write_jsonl(tmp_path / "train.jsonl", [shared])
    eval_file = _write_jsonl(
        tmp_path / "eval.jsonl",
        [shared, {"state": "fresh", "questions": {"q": CHOICE}, "expected": {"q": "tech"}}],
    )
    assert cli.main(["adapt", "validate", "--data", str(train), "--eval", str(eval_file)]) == 0
    out = capsys.readouterr().out
    assert "leakage              1 eval item(s) share a (state, question) signature with train" in out


def test_validate_accepts_csv_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data.csv"
    data.write_text("text,label\nhello,billing\nworld,tech\n", encoding="utf-8")
    assert cli.main(["adapt", "validate", "--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "label  choice  2 items" in out
    assert "billing 1 · tech 1" in out


def test_validate_without_usable_items_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"state": "", "questions": {"q": CHOICE}, "expected": {"q": "billing"}}],
    )
    assert cli.main(["adapt", "validate", "--data", str(data)]) == 1
    captured = capsys.readouterr()
    assert "no usable items" in captured.err


def test_validate_missing_file_is_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["adapt", "validate", "--data", str(tmp_path / "nope.jsonl")]) == 1
    assert "decision-swarm: error:" in capsys.readouterr().err


def test_validate_dry_run_skips_when_base_not_cached(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _write_jsonl(
        tmp_path / "train.jsonl",
        [{"state": "a ticket", "questions": {"q": CHOICE}, "expected": {"q": "billing"}}],
    )
    monkeypatch.setattr(
        cli.registry,
        "get_entry",
        lambda model, registry=None: ModelEntry(id=model, hf_repo=None, package_source="laya>=0.3.20"),
    )
    assert cli.main(["adapt", "validate", "--data", str(data), "--base", "any-base"]) == 0
    out = capsys.readouterr().out
    assert "dry-run skipped: base 'any-base' is not in the shared model cache" in out
