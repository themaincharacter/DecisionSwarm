"""Tests for the shared model cache: fetch, dedup, verify, offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from decision_swarm import cache
from decision_swarm.registry import ModelEntry

ENTRY = ModelEntry(
    id="laya-english",
    hf_repo="convaiinnovations/laya",
    revision="main",
    modality="text",
    package_source="laya>=0.3.20",
    loader="load",
)

SUB_ENTRY = ModelEntry(
    id="laya-multilingual",
    hf_repo="convaiinnovations/laya",
    subfolder="multilingual",
    revision="main",
    modality="text",
    package_source="laya>=0.3.20",
    loader="load",
)


class FakeDownloader:
    """Records calls and plants fake weight files where the hub would."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def __call__(self, repo_id: str, revision: str, target: Path, subfolder: str | None) -> None:
        self.calls.append((repo_id, revision, subfolder))
        base = target / subfolder if subfolder else target
        base.mkdir(parents=True, exist_ok=True)
        label = subfolder.encode() if subfolder else b"root"
        (base / "model.safetensors").write_bytes(b"weights:" + repo_id.encode() + b":" + label)


def test_models_dir_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(cache.MODELS_DIR_ENV, raising=False)
    assert cache.models_dir(tmp_path / "explicit") == tmp_path / "explicit"

    monkeypatch.setenv(cache.MODELS_DIR_ENV, str(tmp_path / "from-env"))
    assert cache.models_dir() == tmp_path / "from-env"
    assert cache.models_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_fetch_downloads_once_then_hits_cache(tmp_path: Path) -> None:
    downloader = FakeDownloader()

    first = cache.fetch(ENTRY, root=tmp_path, downloader=downloader)
    assert first.downloaded is True
    assert downloader.calls == [("convaiinnovations/laya", "main", None)]
    assert (first.path / "model.safetensors").is_file()
    assert first.sha256 and first.bytes > 0

    # the Phase 1 guarantee: a second fetch does no network I/O
    second = cache.fetch(ENTRY, root=tmp_path, downloader=downloader)
    assert second.downloaded is False
    assert downloader.calls == [("convaiinnovations/laya", "main", None)]
    assert second.sha256 == first.sha256
    assert cache.is_cached(ENTRY, root=tmp_path)


def test_offline_with_cold_cache_raises(tmp_path: Path) -> None:
    downloader = FakeDownloader()
    with pytest.raises(cache.CacheError, match="offline"):
        cache.fetch(ENTRY, root=tmp_path, offline=True, downloader=downloader)
    assert downloader.calls == []


def test_offline_with_warm_cache_works(tmp_path: Path) -> None:
    downloader = FakeDownloader()
    cache.fetch(ENTRY, root=tmp_path, downloader=downloader)
    warm = cache.fetch(ENTRY, root=tmp_path, offline=True, downloader=downloader)
    assert warm.downloaded is False
    assert len(downloader.calls) == 1


def test_subfolder_entries_share_one_snapshot_dir(tmp_path: Path) -> None:
    downloader = FakeDownloader()
    cache.fetch(ENTRY, root=tmp_path, downloader=downloader)
    cache.fetch(SUB_ENTRY, root=tmp_path, downloader=downloader)

    assert downloader.calls == [
        ("convaiinnovations/laya", "main", None),
        ("convaiinnovations/laya", "main", "multilingual"),
    ]
    assert cache.snapshot_dir(ENTRY, root=tmp_path) == cache.snapshot_dir(SUB_ENTRY, root=tmp_path)
    assert cache.is_cached(ENTRY, root=tmp_path) and cache.is_cached(SUB_ENTRY, root=tmp_path)


def test_verify_detects_tampering(tmp_path: Path) -> None:
    cache.fetch(ENTRY, root=tmp_path, downloader=FakeDownloader())
    assert cache.verify(ENTRY, root=tmp_path) is True

    (cache.snapshot_dir(ENTRY, root=tmp_path) / "model.safetensors").write_bytes(b"tampered")
    assert cache.verify(ENTRY, root=tmp_path) is False


def test_status_reports_cache_state_and_bytes(tmp_path: Path) -> None:
    cold = cache.status(ENTRY, root=tmp_path)
    assert cold.cached is False and cold.bytes == 0

    result = cache.fetch(ENTRY, root=tmp_path, downloader=FakeDownloader())
    warm = cache.status(ENTRY, root=tmp_path)
    assert warm.cached is True
    assert warm.bytes == result.bytes
    assert warm.path == cache.snapshot_dir(ENTRY, root=tmp_path)


def test_entries_without_repo_or_revision_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(cache.CacheError, match="hf_repo/revision"):
        cache.fetch(ModelEntry(id="orphan"), root=tmp_path, downloader=FakeDownloader())
