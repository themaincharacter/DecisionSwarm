"""Tests for adapter registration, resolution, and merged-checkpoint materialization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_swarm import cache, manifest, pack, registry
from decision_swarm.adapt import register as adapt_register
from decision_swarm.runtime import orchestrator

PASSING_REPORT = {"pass": True, "metric_map": "X = submit detection => noul accuracy"}
FAILING_REPORT = {"pass": False, "metric_map": "X"}


def _adapter_dir(tmp_path: Path, *, name: str = "adapter") -> Path:
    adapter = tmp_path / name
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter.safetensors").write_bytes(b"fake-lora-weights")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "decision_swarm": {
                    "format": "decision-swarm-adapter/1",
                    "base": {"registry_id": "laya-english", "revision": "abc"},
                    "seeds": {"train": 0},
                    "questions": "questions.json",
                },
            }
        ),
        encoding="utf-8",
    )
    (adapter / "questions.json").write_text(
        json.dumps({"submit": {"type": "noul", "instructions": "Is there a submit button?"}}),
        encoding="utf-8",
    )
    return adapter


def _registry_file(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"models": []}), encoding="utf-8")
    return path


def test_register_refuses_a_failed_gate(tmp_path: Path) -> None:
    adapter = _adapter_dir(tmp_path)
    with pytest.raises(adapt_register.AdaptError, match="gate has not passed"):
        adapt_register.register_adapter(
            adapter, report=FAILING_REPORT, registry_path=_registry_file(tmp_path)
        )


def test_register_writes_a_versioned_adapter_entry(tmp_path: Path) -> None:
    adapter = _adapter_dir(tmp_path)
    registry_path = _registry_file(tmp_path)
    entry, snippet = adapt_register.register_adapter(
        adapter, report=PASSING_REPORT, registry_path=registry_path, name="submit-x"
    )
    assert entry["id"] == "laya-english-submit-x"
    assert entry["kind"] == "adapter"
    assert entry["base"] == "laya-english"
    assert entry["merged"]["hf_repo"] == "local/laya-english-submit-x"
    assert entry["merged"]["revision"]
    assert "noul" in snippet and "submit" in snippet
    assert entry["base"] in snippet and entry["merged"]["revision"] in snippet

    # a second registration never overwrites: the id versions up
    entry2, _ = adapt_register.register_adapter(
        adapter, report=PASSING_REPORT, registry_path=registry_path, name="submit-x"
    )
    assert entry2["id"] == "laya-english-submit-x-2"
    raw = json.loads(registry_path.read_text(encoding="utf-8"))
    assert [item["id"] for item in raw["models"]] == ["laya-english-submit-x", "laya-english-submit-x-2"]


def test_versioned_id_rejects_bad_names() -> None:
    with pytest.raises(adapt_register.AdaptError, match="slug"):
        adapt_register.versioned_id("laya-english", "Not A Slug", set())


def test_patch_manifest_swaps_one_cell(tmp_path: Path) -> None:
    swarm_path = tmp_path / "swarm.json"
    swarm_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "demo",
                "input_schema": {"views": {"text": {"type": "text"}}},
                "cells": [
                    {"id": "a", "model": "laya-english", "modality": "text", "views": ["text"]},
                    {"id": "b", "model": "laya-english", "modality": "text", "views": ["text"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    entry = {"base": "laya-typed-decisions", "merged": {"hf_repo": "local/x", "revision": "d1"}}
    adapt_register.patch_manifest(swarm_path, "b", entry)
    swarm = manifest.load_manifest(swarm_path)
    assert swarm.cells[0].model == "laya-english" and swarm.cells[0].adapter is None
    assert swarm.cells[1].model == "laya-typed-decisions"
    assert swarm.cells[1].adapter == "local/x@d1"


def test_manifest_roundtrips_cell_adapter(tmp_path: Path) -> None:
    data = {
        "schema_version": 1,
        "name": "demo",
        "input_schema": {"views": {"text": {"type": "text"}}},
        "cells": [
            {
                "id": "a",
                "model": "laya-english",
                "adapter": "local/laya-english-submit-x@0123",
                "modality": "text",
                "views": ["text"],
            },
            {"id": "b", "model": "laya-english", "modality": "text", "views": ["text"]},
        ],
    }
    swarm = manifest.parse_manifest(data)
    assert swarm.cells[0].adapter == "local/laya-english-submit-x@0123"
    assert manifest.to_dict(swarm)["cells"][0]["adapter"] == "local/laya-english-submit-x@0123"


def test_resolve_adapter_from_ref_and_from_registry() -> None:
    base = registry.get_entry("laya-english")
    merged = registry.resolve_adapter("local/some-name@deadbeef", base=base)
    assert merged.hf_repo == "local/some-name"
    assert merged.revision == "deadbeef"
    assert merged.package_source == base.package_source

    entry = {
        "id": "laya-english-x",
        "kind": "adapter",
        "base": "laya-english",
        "merged": {"hf_repo": "local/laya-english-x", "revision": "cafebabe"},
    }
    entries = {"laya-english-x": registry._entry_from_dict(entry)}
    merged2 = registry.resolve_adapter("laya-english-x", base=base, registry=entries)
    assert registry.merged_ref(merged2) == "local/laya-english-x@cafebabe"


def test_resolve_checkpoint_merges_for_adapter_cells() -> None:
    swarm = manifest.parse_manifest(
        {
            "schema_version": 1,
            "name": "demo",
            "input_schema": {"views": {"text": {"type": "text"}}},
            "cells": [
                {"id": "a", "model": "laya-english", "adapter": "local/n@ab", "modality": "text", "views": ["text"]},
                {"id": "b", "model": "laya-english", "modality": "text", "views": ["text"]},
            ],
        }
    )
    merged = registry.resolve_checkpoint(swarm.cells[0])
    assert merged.hf_repo == "local/n" and merged.revision == "ab"
    plain = registry.resolve_checkpoint(swarm.cells[1])
    assert plain.id == "laya-english"


def test_fetch_refuses_unmaterialized_merged_checkpoints(tmp_path: Path) -> None:
    entry = registry.resolve_adapter("local/n@ab", base=registry.get_entry("laya-english"))
    with pytest.raises(cache.CacheError, match="materialized"):
        cache.fetch(entry, root=tmp_path)


def test_record_local_marks_a_merged_checkpoint(tmp_path: Path) -> None:
    entry = registry.resolve_adapter("local/n@ab", base=registry.get_entry("laya-english"))
    directory = cache.snapshot_dir(entry, tmp_path)
    directory.mkdir(parents=True)
    (directory / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    (directory / "model.safetensors").write_bytes(b"weights")
    result = cache.record_local(entry, tmp_path)
    assert result.bytes > 0
    assert cache.is_cached(entry, tmp_path)
    assert cache.verify(entry, tmp_path)
    # fetch now serves it from the record without any network
    hit = cache.fetch(entry, root=tmp_path)
    assert hit.downloaded is False


def test_unique_entries_and_plan_cells_use_the_merged_checkpoint(tmp_path: Path) -> None:
    swarm = manifest.parse_manifest(
        {
            "schema_version": 1,
            "name": "demo",
            "input_schema": {"views": {"text": {"type": "text"}}},
            "cells": [
                {"id": "a", "model": "laya-english", "adapter": "local/n@ab", "modality": "text", "views": ["text"]},
                {"id": "b", "model": "laya-english", "modality": "text", "views": ["text"]},
            ],
        }
    )
    entries = pack.unique_entries(swarm)
    assert [entry.hf_repo for entry in entries] == ["local/n", "convaiinnovations/laya"]

    with pytest.raises(orchestrator.OrchestratorError, match="materialized"):
        orchestrator.plan_cells(swarm, models_root=tmp_path, interpreter=sys_executable())

    entry = registry.resolve_adapter("local/n@ab", base=registry.get_entry("laya-english"))
    directory = cache.snapshot_dir(entry, tmp_path)
    directory.mkdir(parents=True)
    (directory / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    cache.record_local(entry, tmp_path)
    plans = orchestrator.plan_cells(swarm, models_root=tmp_path, interpreter=sys_executable())
    assert plans[0].model.hf_repo == "local/n"
    assert plans[1].model.id == "laya-english"


def sys_executable() -> str:
    import sys

    return sys.executable
