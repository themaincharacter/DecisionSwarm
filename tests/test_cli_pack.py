"""CLI tests for inspect and fetch against a built pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_swarm import cache, cli, manifest as manifest_mod, pack
from test_cache import FakeDownloader
from test_pack import MANIFEST_DICT


@pytest.fixture()
def pack_dir(tmp_path: Path) -> Path:
    swarm = manifest_mod.parse_manifest(MANIFEST_DICT)
    return pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")


def _run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_inspect_lists_cells_sizes_and_cache_status(
    pack_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _run(
        ["inspect", "--pack", str(pack_dir), "--models-dir", str(tmp_path / "cache")], capsys
    )
    assert code == 0
    assert "pack: browser-swarm" in out
    assert "screenshot_sensor" in out and "dom_sensor" in out
    assert "laya-vision" in out and "laya-english" in out
    assert "421M" in out  # registry sizes surface in inspect
    assert "cached: no" in out
    assert str(tmp_path / "cache") in out


def test_fetch_populates_cache_and_second_fetch_does_no_network(
    pack_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = FakeDownloader()
    monkeypatch.setattr(cache, "hub_download", downloader)
    argv = ["fetch", "--pack", str(pack_dir), "--models-dir", str(tmp_path / "cache")]

    code, out, _ = _run(argv, capsys)
    assert code == 0
    assert out.count("downloaded") == 2
    assert "already cached" not in out
    assert len(downloader.calls) == 2  # one per unique checkpoint

    code, out, _ = _run(argv, capsys)
    assert code == 0
    assert out.count("already cached") == 2
    assert len(downloader.calls) == 2  # no network I/O on the second fetch


def test_inspect_shows_cached_after_fetch(
    pack_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    downloader = FakeDownloader()
    monkeypatch.setattr(cache, "hub_download", downloader)
    _run(["fetch", "--pack", str(pack_dir), "--models-dir", str(tmp_path / "cache")], capsys)

    code, out, _ = _run(
        ["inspect", "--pack", str(pack_dir), "--models-dir", str(tmp_path / "cache")], capsys
    )
    assert code == 0
    assert out.count("cached: yes") == 2


def test_missing_pack_reports_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = _run(["inspect", "--pack", str(tmp_path / "nope")], capsys)
    assert code == 1
    assert "error" in err
