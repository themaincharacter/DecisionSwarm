"""Tests for the pack builder: artifacts, lockfile, rendered runtime."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from decision_swarm import cache, manifest as manifest_mod, pack
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
        "runtime/pyproject.toml",
        "runtime/src/browser_swarm/__init__.py",
        "runtime/src/browser_swarm/cli.py",
        "runtime/src/browser_swarm/swarm.json",
    ):
        assert (dest / relative).is_file(), f"missing {relative}"


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


def test_helpers_derive_names() -> None:
    assert pack.pack_slug("Browser Swarm!") == "browser-swarm"
    assert pack.pack_module("Browser Swarm!") == "browser_swarm"
