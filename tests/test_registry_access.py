"""Tests for model registry access and model resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from decision_swarm import registry

MINI_REGISTRY: dict[str, Any] = {
    "schema_version": 1,
    "models": [
        {
            "id": "laya-english",
            "hf_repo": "convaiinnovations/laya",
            "subfolder": None,
            "revision": "main",
            "modality": "text",
            "package_source": "laya>=0.3.20",
            "loader": "load",
            "params": "421M",
            "download_bytes": 842609210,
            "license": "Apache-2.0",
            "notes": "root checkpoint",
        },
        {
            "id": "laya-multilingual",
            "hf_repo": "convaiinnovations/laya",
            "subfolder": "multilingual",
            "revision": "main",
            "modality": "text",
            "package_source": "laya>=0.3.20",
            "loader": "load",
            "params": "322M",
            "download_bytes": 643835514,
            "license": "Apache-2.0",
            "notes": "subfolder checkpoint",
        },
        {
            "id": "custom-hf-repo",
            "custom": True,
            "notes": "free-form",
        },
    ],
}


def test_loads_the_shipped_registry() -> None:
    entries = registry.load_registry()
    assert {"laya-english", "laya-multilingual", "laya-typed-decisions", "laya-vision"} <= set(entries)
    vision = entries["laya-vision"]
    assert vision.loader == "load_vlm"
    assert "CC BY-NC-SA 4.0" in vision.display_license


def test_get_entry_by_id_is_strict() -> None:
    entries = registry.load_registry()
    assert registry.get_entry("laya-english", entries).params == "421M"
    with pytest.raises(registry.RegistryError, match="unknown registry id"):
        registry.get_entry("nope", entries)


def test_resolve_by_hf_repo_root() -> None:
    entries = registry.load_registry()
    assert registry.resolve_model("thaitea/laya-vision", entries).id == "laya-vision"
    # the hub repo root is the English checkpoint; subfolder variants need ids
    assert registry.resolve_model("convaiinnovations/laya", entries).id == "laya-english"


def test_resolve_custom_hf_repo() -> None:
    entries = registry.load_registry()
    custom = registry.resolve_model("my-org/my-model", entries)
    assert custom.custom is True
    assert custom.hf_repo == "my-org/my-model"
    assert custom.revision == "main"
    assert custom.display_license == "unknown (user-supplied)"


def test_resolve_unknown_model_is_rejected() -> None:
    entries = registry.load_registry()
    with pytest.raises(registry.RegistryError, match="not a registry id"):
        registry.resolve_model("Definitely Not A Model", entries)


def test_load_registry_from_path(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(MINI_REGISTRY), encoding="utf-8")
    entries = registry.load_registry(path)
    assert set(entries) == {"laya-english", "laya-multilingual", "custom-hf-repo"}
    assert registry.resolve_model("convaiinnovations/laya", entries).id == "laya-english"
    assert registry.get_entry("laya-multilingual", entries).subfolder == "multilingual"


def test_registry_path_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(MINI_REGISTRY), encoding="utf-8")
    monkeypatch.setenv(registry.REGISTRY_ENV, str(path))
    assert registry.default_registry_path() == path
    assert registry.load_registry()["laya-english"].license == "Apache-2.0"


def test_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    duplicated = json.loads(json.dumps(MINI_REGISTRY))
    duplicated["models"].append(dict(duplicated["models"][0]))
    path.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(registry.RegistryError, match="duplicate registry id"):
        registry.load_registry(path)


def test_rejects_broken_registry_files(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(registry.RegistryError, match="cannot read registry"):
        registry.load_registry(path)

    path.write_text(json.dumps({"models": [{"license": "Apache-2.0"}]}), encoding="utf-8")
    with pytest.raises(registry.RegistryError, match="missing an 'id'"):
        registry.load_registry(path)
