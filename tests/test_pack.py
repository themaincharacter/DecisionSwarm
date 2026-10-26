"""Tests for the pack builder: artifacts, lockfile, rendered runtime."""

from __future__ import annotations

import ast
import json
import tarfile
from pathlib import Path

import pytest

from decision_swarm import cache, envs, manifest as manifest_mod, pack
from decision_swarm.registry import ModelEntry

MANIFEST_DICT = {
    "schema_version": 1,
    "name": "browser-swarm",
    "input_schema": {
        "views": {
            "screenshot": {"type": "image"},
            "dom": {"type": "text"},
            "classes": {"type": "text"},
        }
    },
    "cells": [
        {
            "id": "screenshot_sensor",
            "model": "thaitea/laya-vision",
            "modality": "image+text",
            "views": ["screenshot"],
            "default_questions": {
                "submit_visible": {"type": "noul", "instructions": "Do you see a submit button?"}
            },
            "runtime": {
                "package_source": "git+https://github.com/r33drichards/laya-vision",
                "loader": "load_vlm",
            },
        },
        {
            "id": "dom_sensor",
            "model": "convaiinnovations/laya",
            "modality": "text",
            "views": ["dom", "classes"],
            "default_questions": {
                "submit_button": {
                    "type": "choice",
                    "instructions": "Which element is the submit button?",
                    "criteria": {"a": "the input with value Send", "b": "the button labeled Send"},
                }
            },
        },
    ],
}


class FakeDownloader:
    def __call__(self, repo_id: str, revision: str, target: Path, subfolder: str | None) -> None:
        base = target / subfolder if subfolder else target
        base.mkdir(parents=True, exist_ok=True)
        (base / "model.safetensors").write_bytes(b"weights:" + repo_id.encode())


@pytest.fixture()
def swarm() -> manifest_mod.SwarmManifest:
    return manifest_mod.parse_manifest(MANIFEST_DICT)


def test_build_pack_writes_all_artifacts(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = tmp_path / "my-swarm"
    built = pack.build_pack(swarm, dest, models_root=tmp_path / "cache")
    assert built == dest

    for relative in (
        "swarm.json",
        "models.lock.json",
        "README.md",
        "mcp.json",
        "runtime/pyproject.toml",
        "runtime/src/browser_swarm/__init__.py",
        "runtime/src/browser_swarm/cli.py",
        "runtime/src/browser_swarm/swarm.json",
    ):
        assert (dest / relative).is_file(), f"missing {relative}"


def test_mcp_config_points_at_this_pack(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")
    written = json.loads((dest / "mcp.json").read_text(encoding="utf-8"))

    assert written == pack.mcp_config(dest)
    server = written["mcpServers"]["decision-swarm"]
    assert server["command"] == "decision-swarm"
    assert server["args"] == [
        "mcp",
        "--packs-dir",
        str(dest.resolve().parent),
        "--pack",
        str(dest.resolve()),
    ]


def test_readme_documents_the_agent_tools(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")
    readme = (dest / "README.md").read_text(encoding="utf-8")

    assert "## Agents (MCP)" in readme
    assert '"mcpServers"' in readme
    assert "regenerate" in readme


def test_generated_runtime_exposes_the_mcp_command(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")
    rendered = (dest / "runtime" / "src" / "browser_swarm" / "cli.py").read_text(
        encoding="utf-8"
    )

    assert '"mcp"' in rendered


def test_pack_manifest_round_trips(tmp_path: Path, swarm: manifest_mod.SwarmManifest) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")
    loaded = manifest_mod.load_manifest(dest)
    assert manifest_mod.to_dict(loaded) == manifest_mod.to_dict(swarm)
    embedded = manifest_mod.load_manifest(dest / "runtime" / "src" / "browser_swarm" / "swarm.json")
    assert manifest_mod.to_dict(embedded) == manifest_mod.to_dict(swarm)


def test_lockfile_records_unique_checkpoints(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")
    lock = json.loads((dest / "models.lock.json").read_text(encoding="utf-8"))

    assert lock["schema_version"] == 1
    assert [(m["id"], m["hf_repo"], m["subfolder"], m["revision"]) for m in lock["models"]] == [
        ("laya-vision", "thaitea/laya-vision", None, "main"),
        ("laya-english", "convaiinnovations/laya", None, "main"),
    ]
    for model in lock["models"]:
        assert model["cached"] is False
        assert model["sha256"] == {}


def test_lockfile_fills_sha256_when_cached(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    models_root = tmp_path / "cache"
    downloader = FakeDownloader()
    for model_id in ("laya-vision", "laya-english"):
        entry = ModelEntry(
            id=model_id,
            hf_repo="thaitea/laya-vision" if model_id == "laya-vision" else "convaiinnovations/laya",
            revision="main",
        )
        cache.fetch(entry, root=models_root, downloader=downloader)

    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=models_root)
    lock = json.loads((dest / "models.lock.json").read_text(encoding="utf-8"))
    for model in lock["models"]:
        assert model["cached"] is True
        assert model["bytes"] > 0
        assert model["sha256"] == {"model.safetensors": cache.sha256_file(
            models_root / model["hf_repo"] / model["revision"] / "model.safetensors"
        )}


def test_generated_runtime_renders_templates(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")

    for relative in (
        "runtime/pyproject.toml",
        "runtime/src/browser_swarm/__init__.py",
        "runtime/src/browser_swarm/cli.py",
    ):
        text = (dest / relative).read_text(encoding="utf-8")
        assert "{{" not in text, f"unrendered placeholder in {relative}"
        if relative.endswith(".py"):
            ast.parse(text)

    pyproject = (dest / "runtime" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "browser-swarm"' in pyproject
    assert 'browser-swarm = "browser_swarm.cli:main"' in pyproject


def test_readme_mentions_cells_and_usage(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "# browser-swarm (swarm pack)" in readme
    assert "`screenshot_sensor`" in readme and "`dom_sensor`" in readme
    assert "pip install ./runtime" in readme
    assert "decision-swarm fetch --pack ." in readme
    assert "long-lived" in readme and "serve" in readme


def test_build_is_idempotent(tmp_path: Path, swarm: manifest_mod.SwarmManifest) -> None:
    dest = tmp_path / "my-swarm"
    first = pack.build_pack(swarm, dest, models_root=tmp_path / "cache")
    second = pack.build_pack(swarm, dest, models_root=tmp_path / "cache")
    assert first == second == dest
    assert (dest / "swarm.json").read_text(encoding="utf-8") == (
        dest / "runtime" / "src" / "browser_swarm" / "swarm.json"
    ).read_text(encoding="utf-8")


def test_collect_models_records_a_resolved_commit(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    models = pack.collect_models(
        swarm,
        models_root=tmp_path / "cache",
        resolve_commit=lambda entry: f"sha-{entry.id}",
    )

    assert [model.commit for model in models] == [
        "sha-laya-vision",
        "sha-laya-english",
    ]
    # the floating ref survives, so a reader can see what was pinned
    assert {model.revision for model in models} == {"main"}


def test_collect_models_leaves_commits_unset_without_a_resolver(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    models = pack.collect_models(swarm, models_root=tmp_path / "cache")
    assert all(model.commit is None for model in models)


def test_lockfile_records_the_commit(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(
        swarm,
        tmp_path / "my-swarm",
        models_root=tmp_path / "cache",
        resolve_commit=lambda entry: "deadbeef",
    )

    lock = json.loads((dest / "models.lock.json").read_text(encoding="utf-8"))
    assert {model["commit"] for model in lock["models"]} == {"deadbeef"}


def _seed_cache(root: Path, swarm: manifest_mod.SwarmManifest) -> None:
    """Plant checkpoint files where ``cache.fetch`` would leave them."""
    for entry in pack.unique_entries(swarm):
        snapshot = root / (entry.hf_repo or "") / (entry.revision or "")
        target = snapshot / entry.subfolder if entry.subfolder else snapshot
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")


def test_export_bundle_carries_the_pack_and_its_weights(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root, swarm)
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=cache_root)

    bundle = pack.export_bundle(dest, models_root=cache_root)

    assert bundle == tmp_path / "my-swarm.swarmpack.tar"
    with tarfile.open(bundle) as archive:
        names = set(archive.getnames())
    assert "my-swarm/swarm.json" in names
    assert "my-swarm/models.lock.json" in names
    assert "my-swarm/models/convaiinnovations/laya/main/config.json" in names
    assert "my-swarm/models/thaitea/laya-vision/main/config.json" in names


def test_export_bundle_refuses_a_cold_cache(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")

    with pytest.raises(pack.PackError, match="not in the cache"):
        pack.export_bundle(dest, models_root=tmp_path / "cache")
    assert not (tmp_path / "my-swarm.swarmpack.tar").exists()


def test_export_bundle_accepts_an_explicit_destination(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    cache_root = tmp_path / "cache"
    _seed_cache(cache_root, swarm)
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=cache_root)

    bundle = pack.export_bundle(dest, tmp_path / "out" / "ship.tar", models_root=cache_root)

    assert bundle == tmp_path / "out" / "ship.tar" and bundle.is_file()


def _seed_envs(root: Path, swarm: manifest_mod.SwarmManifest) -> list[str]:
    """Plant built-looking envs (one python per unique package source)."""
    keys: list[str] = []
    for planned in envs.cell_envs(swarm, envs_root=root):
        (planned.path / "bin").mkdir(parents=True, exist_ok=True)
        (planned.path / "bin" / "python").write_text("", encoding="utf-8")
        keys.append(planned.key)
    return keys


def test_export_bundle_carries_the_cell_envs(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    cache_root, envs_root = tmp_path / "cache", tmp_path / "envs"
    _seed_cache(cache_root, swarm)
    keys = _seed_envs(envs_root, swarm)
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=cache_root)

    bundle = pack.export_bundle(dest, models_root=cache_root, envs_root=envs_root)

    with tarfile.open(bundle) as archive:
        names = set(archive.getnames())
    assert len(keys) == 2  # one per unique package source
    for key in keys:
        assert f"my-swarm/envs/{key}/bin/python" in names


def test_export_bundle_can_leave_envs_out(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    cache_root, envs_root = tmp_path / "cache", tmp_path / "envs"
    _seed_cache(cache_root, swarm)
    _seed_envs(envs_root, swarm)
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=cache_root)

    bundle = pack.export_bundle(
        dest, models_root=cache_root, envs_root=envs_root, include_envs=False
    )

    with tarfile.open(bundle) as archive:
        assert not any("/envs/" in name for name in archive.getnames())


def test_missing_envs_reports_unbuilt_sources(
    tmp_path: Path, swarm: manifest_mod.SwarmManifest
) -> None:
    dest = pack.build_pack(swarm, tmp_path / "my-swarm", models_root=tmp_path / "cache")

    assert set(pack.missing_envs(dest, envs_root=tmp_path / "envs")) == {
        "laya>=0.3.20",
        "git+https://github.com/r33drichards/laya-vision",
    }


def test_helpers_derive_names() -> None:
    assert pack.pack_slug("Browser Swarm!") == "browser-swarm"
    assert pack.pack_module("Browser Swarm!") == "browser_swarm"
