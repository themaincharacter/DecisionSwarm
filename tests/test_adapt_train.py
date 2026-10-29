"""Tests for `decision-swarm adapt train` — pure parts, plus GPU-marked smoke.

The driver, config building, and artifact formats are testable without a model
stack; the training loop itself is upstream (`laya.train`) and is exercised only
under the `gpu` marker (needs a training env: laya + torch + peft).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from decision_swarm import cli
from decision_swarm.adapt import lora as lora_mod
from decision_swarm.adapt import train as adapt_train
from decision_swarm.registry import ModelEntry

BASE_ENTRY = ModelEntry(
    id="laya-english",
    hf_repo="convaiinnovations/laya",
    subfolder=None,
    revision="main",
    package_source="laya>=0.3.20",
    loader="load",
)


def _fake_env(tmp_path: Path):
    class _Env:
        python = tmp_path / "bin" / "python"

    return _Env()


def _stub_execution(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    monkeypatch.setattr(cli, "_adapt_prepare_env", lambda entry, envs_root=None: _fake_env(Path(captured["tmp"])))
    monkeypatch.setattr(
        cli,
        "_run_adapt_training",
        lambda python, config: captured.update(config=config) or 0,
    )
    monkeypatch.setattr(cli.cache, "fetch", lambda entry, **kw: None)


def test_train_help_lists_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["adapt", "train", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for option in ("--base", "--data", "--eval", "--out", "--mode", "--rank", "--metric-map", "--provenance"):
        assert option in out


def test_train_rejects_custom_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli.registry,
        "get_entry",
        lambda model, registry=None: ModelEntry(id=model, custom=True),
    )
    assert cli.main(["adapt", "train", "--base", "custom-hf-repo", "--data", "d.jsonl", "--out", "o"]) == 1


def test_train_rejects_vision_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli.registry,
        "get_entry",
        lambda model, registry=None: ModelEntry(id=model, hf_repo="t/laya-vision", package_source="laya-vision", loader="load_vlm"),
    )
    assert cli.main(["adapt", "train", "--base", "laya-vision", "--data", "d.jsonl", "--out", "o"]) == 1


def test_train_refuses_the_frozen_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
    assert cli.main(["adapt", "train", "--base", "laya-english", "--data", "d.jsonl", "--out", "o"]) == 1


def test_train_builds_config_and_runs_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {"tmp": str(tmp_path)}
    _stub_execution(monkeypatch, captured)
    monkeypatch.setattr(cli.registry, "get_entry", lambda model, registry=None: BASE_ENTRY)
    # a cached base checkpoint (ModelSpec.local_path marker)
    cached = tmp_path / "models" / "convaiinnovations" / "laya" / "main"
    cached.mkdir(parents=True)
    (cached / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli.cache, "models_dir", lambda override=None: tmp_path / "models")

    code = cli.main(
        [
            "adapt", "train",
            "--base", "laya-english",
            "--data", "train.jsonl",
            "--eval", "eval.jsonl",
            "--out", str(tmp_path / "out"),
            "--mode", "lora",
            "--epochs", "2",
            "--metric-map", "X = submit-button detection => noul accuracy",
            "--provenance", "generator=gpt-x",
            "--provenance", "seed=7",
        ]
    )
    assert code == 0
    config = captured["config"]
    assert config["mode"] == "lora"
    assert config["epochs"] == 2
    assert config["base"]["registry_id"] == "laya-english"
    assert config["base_dir"] == str(cached)
    assert config["provenance"] == {"generator": "gpt-x", "seed": "7"}
    assert config["metric_map"].startswith("X = submit-button")


def test_train_rejects_bad_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {"tmp": str(tmp_path)}
    _stub_execution(monkeypatch, captured)
    monkeypatch.setattr(cli.registry, "get_entry", lambda model, registry=None: BASE_ENTRY)
    cached = tmp_path / "models" / "convaiinnovations" / "laya" / "main"
    cached.mkdir(parents=True)
    (cached / "rl_agent_config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli.cache, "models_dir", lambda override=None: tmp_path / "models")
    assert (
        cli.main(
            ["adapt", "train", "--base", "laya-english", "--data", "d.jsonl", "--out", "o", "--provenance", "oops"]
        )
        == 1
    )


def test_questions_from_rows_keeps_first_seen() -> None:
    rows = [
        {"state": "a", "questions": {"q1": {"type": "noul"}, "q2": {"type": "choice"}}},
        {"state": "b", "questions": {"q1": {"type": "noul", "instructions": "changed"}}},
    ]
    assert adapt_train.questions_from_rows(rows) == {"q1": {"type": "noul"}, "q2": {"type": "choice"}}


def test_build_adapter_config_carries_provenance() -> None:
    config = adapt_train.build_adapter_config(
        {"peft_type": "LORA", "r": 8},
        base={"registry_id": "laya-english", "revision": "abc", "weights_sha256": "00"},
        trainable_params=123,
        total_params=421_000_000,
        seeds={"train": 0, "calibration": 20260922},
        questions_file="questions.json",
    )
    assert config["peft_type"] == "LORA"  # community fields stay at the top level
    block = config["decision_swarm"]
    assert block["format"] == adapt_train.ADAPTER_FORMAT
    assert block["base"]["weights_sha256"] == "00"
    assert block["trainable_params"] == 123
    assert block["seeds"] == {"train": 0, "calibration": 20260922}
    json.dumps(config)  # JSON-safe


def test_auto_micro_batch_is_conservative_on_small_vram() -> None:
    class _Device:
        type = "cpu"

    assert adapt_train.auto_micro_batch(_Device()) == 1


def test_resolve_target_modules_needs_a_known_encoder() -> None:
    class _Encoder:
        def named_modules(self):
            return [("x.other", object())]

    with pytest.raises(lora_mod.AdaptError, match="no LoRA target modules"):
        lora_mod.resolve_target_modules(_Encoder())


# ----------------------------------------------------------------- GPU smoke


@pytest.mark.gpu
def test_lora_wrap_merge_roundtrip() -> None:
    """LoRA wrap -> save -> load -> merge on a tiny random encoder (no downloads)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    nn = pytest.importorskip("torch.nn")

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.TransformerEncoderLayer(16, 2, 32, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, 1)
            self.scorer = nn.Linear(16, 1)

    model = _Model()
    wrapped = lora_mod.wrap_encoder(model, rank=2)
    assert wrapped.target_modules
    trainable, total = lora_mod.trainable_parameters(model)
    assert 0 < trainable < total

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        lora_mod.save_adapter(model.encoder, tmp)
        assert (Path(tmp) / lora_mod.ADAPTER_WEIGHTS_NAME).is_file()
    lora_mod.merge_encoder(model)
    assert getattr(model.encoder, "peft_config", None) is None
