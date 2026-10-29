"""Merge an adapter into its base checkpoint — what ``fetch``/``install`` materialize.

Spawned as ``python -m decision_swarm.adapt.merge --config <json>`` under a
training env (it needs ``laya`` + ``peft``).  The result is a full checkpoint
that ``laya.load`` serves unchanged — the loader never sees an adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from decision_swarm.adapt import AdaptError


def lora_config_from_adapter(adapter_config: dict[str, Any]) -> dict[str, Any]:
    """The peft fields of an adapter config, ready for ``LoraConfig(**...)``."""
    from dataclasses import fields as dataclass_fields

    from peft import LoraConfig

    known = {field.name for field in dataclass_fields(LoraConfig)}
    return {key: value for key, value in adapter_config.items() if key in known}


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Load base -> wrap -> load adapter weights -> merge -> save the checkpoint."""
    from laya.train import load_checkpoint, save_checkpoint
    from peft import LoraConfig, get_peft_model

    from decision_swarm.adapt import lora as lora_mod

    base_dir = Path(config["base_dir"])
    adapter_dir = Path(config["adapter_dir"])
    out_dir = Path(config["out_dir"])
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise AdaptError(f"adapter config not found: {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))

    model, tok, base_cfg = load_checkpoint(str(base_dir))
    model.encoder = get_peft_model(model.encoder, LoraConfig(**lora_config_from_adapter(adapter_config)))
    lora_mod.load_adapter_weights(model.encoder, adapter_dir)
    lora_mod.merge_encoder(model)

    out_cfg = dict(base_cfg)
    out_cfg["training"] = dict(
        out_cfg.get("training") or {},
        decision_swarm_adapter=adapter_config.get("decision_swarm", {"adapter": str(adapter_dir)}),
    )
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, tok, out_cfg, str(out_dir))
    return {"out": str(out_dir), "base": str(base_dir), "adapter": str(adapter_dir)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="decision-swarm-adapt-merge")
    parser.add_argument("--config", required=True, help="run configuration (JSON)")
    args = parser.parse_args(argv)
    try:
        result = run(json.loads(args.config))
    except ImportError as exc:
        print(f"decision-swarm-adapt-merge: this command runs inside the training env: {exc}", file=sys.stderr)
        return 2
    except AdaptError as exc:
        print(f"decision-swarm-adapt-merge: {exc}", file=sys.stderr)
        return 2
    print(f"merge: {result['out']}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
