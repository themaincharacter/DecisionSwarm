"""The model registry file is part of the Phase 0 scaffold and must stay sane."""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry" / "models.json"

REQUIRED_FIELDS = {
    "id",
    "hf_repo",
    "subfolder",
    "revision",
    "modality",
    "package_source",
    "loader",
    "params",
    "download_bytes",
    "license",
    "notes",
}

KNOWN_IDS = {"laya-english", "laya-multilingual", "laya-typed-decisions", "laya-vision"}


def _load() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_registry_file_loads_and_fields_are_present() -> None:
    data = _load()
    assert data["schema_version"] == 1
    models = data["models"]

    ids = [model["id"] for model in models]
    assert len(ids) == len(set(ids)), "registry ids must be unique"
    assert KNOWN_IDS <= set(ids)

    for model in models:
        assert REQUIRED_FIELDS <= set(model), f"{model.get('id')!r} is missing fields"


def test_registry_entries_carry_their_facts() -> None:
    models = _load()["models"]
    for model in models:
        if model.get("custom"):
            continue
        assert model["hf_repo"] and model["revision"]
        assert model["modality"] in ("text", "image+text")
        assert model["loader"] in ("load", "load_vlm")
        assert isinstance(model["download_bytes"], int) and model["download_bytes"] > 0
        assert model["license"]


def test_laya_checkpoints_share_the_hub_repo() -> None:
    models = _load()["models"]
    laya_family = [m for m in models if m["id"].startswith("laya-") and m["modality"] == "text"]
    assert {m["hf_repo"] for m in laya_family} == {"convaiinnovations/laya"}
    assert {m["subfolder"] for m in laya_family} == {None, "multilingual", "typed-decisions"}


def test_vision_entries_surface_non_commercial_weights() -> None:
    models = _load()["models"]
    vision = [m for m in models if m["modality"] == "image+text"]
    assert vision, "expected at least one vision checkpoint"
    for model in vision:
        assert "CC BY-NC-SA 4.0" in model["license"]
        assert model["loader"] == "load_vlm"
