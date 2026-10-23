"""Smoke tests for the CLI scaffold: ``decision-swarm --help`` must work."""

from __future__ import annotations

import pytest

from decision_swarm import __version__, cli


def test_help_lists_all_commands_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in ("inspect", "fetch", "install", "run"):
        assert command in out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_subcommand_help_works(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["run", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for option in ("--pack", "--input", "--offline", "--json", "--pretty"):
        assert option in out


def test_missing_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])
    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["install", "--pack", "my-swarm"],
        ["run", "--pack", "my-swarm", "--input", "obs.json"],
    ],
)
def test_subcommands_report_not_implemented(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(argv) == 2
    assert "not implemented yet" in capsys.readouterr().err
