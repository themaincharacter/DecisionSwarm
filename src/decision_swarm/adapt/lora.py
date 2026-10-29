"""PEFT (LoRA) wrapping for laya decision models — the one part of training we own.

Training itself is ``laya.train``'s loop (``train_model``, the losses, the
calibration); this module only wraps the encoder with LoRA adapters and merges
them back.  The spike that fixed the target modules: a laya ``DecisionModel``
(``laya.common.build_model``) is ``encoder`` — a transformers ``AutoModel`` in
the ModernBERT/mmBERT layer layout — plus ``head`` / ``type_emb`` / ``scorer`` /
``act_head``, so the LoRA targets are the encoder's attention and MLP
projections.

Wrapping *only* the encoder (not the whole ``DecisionModel``) keeps
``laya.train.train_model``'s parameter groups honest: wrapped parameters are
still named ``encoder.*`` and train at ``encoder_lr``, everything else is the
head at ``head_lr``, and ``freeze_encoder`` still means what it says.  Merge
before saving so ``laya.load`` serves the result unchanged (the plan's
merged-checkpoint materialization).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from decision_swarm.adapt import AdaptError

LORA_TARGETS = ("Wqkv", "Wo", "Wi")
"""Encoder projections to adapt: fused QKV, attention out, MLP in/out.

ModernBERT-style layers name them ``attention.{Wqkv,Wo}`` and ``mlp.{Wi,Wo}``;
``Wo`` covers both out-projections.  :func:`resolve_target_modules` verifies
what a given encoder actually has before wrapping — an unfamiliar backbone
fails loudly instead of training nothing.
"""

ADAPTER_WEIGHTS_NAME = "adapter.safetensors"


def _is_linear(module: Any) -> bool:
    """``nn.Linear`` by shape, so this pre-flight check needs no model stack."""
    return hasattr(module, "weight") and hasattr(module, "in_features") and hasattr(module, "out_features")


def resolve_target_modules(encoder: Any) -> list[str]:
    """The :data:`LORA_TARGETS` suffixes this encoder really contains."""
    suffixes = {
        name.rsplit(".", 1)[-1]
        for name, module in encoder.named_modules()
        if name and _is_linear(module)
    }
    found = [target for target in LORA_TARGETS if target in suffixes]
    if not found:
        raise AdaptError(
            f"no LoRA target modules in the encoder (looked for {list(LORA_TARGETS)}, "
            f"found linear suffixes {sorted(suffixes)[:12]})"
        )
    return found


def wrap_encoder(model: Any, *, rank: int = 8, alpha: int = 16, dropout: float = 0.05) -> Any:
    """Attach LoRA adapters to ``model.encoder`` in place; return the peft config.

    No stacking: an encoder that already carries adapters is refused (one
    adapter per cell, no adapter-on-adapter).
    """
    from peft import LoraConfig, get_peft_model

    if getattr(model.encoder, "peft_config", None):
        raise AdaptError("the encoder already carries adapters; no adapter-on-adapter")
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=resolve_target_modules(model.encoder),
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, config)
    return config


def trainable_parameters(model: Any) -> tuple[int, int]:
    """``(trainable, total)`` parameter counts, for the adapter config."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_adapter(encoder: Any, directory: str | Path) -> Path:
    """Write the adapter (PEFT ``save_pretrained``) and return its weights path.

    The weights land as :data:`ADAPTER_WEIGHTS_NAME`; the peft
    ``adapter_config.json`` stays beside them and the caller enriches it.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(str(directory))
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = directory / name
        if candidate.exists():
            candidate.rename(directory / ADAPTER_WEIGHTS_NAME)
            break
    return directory / ADAPTER_WEIGHTS_NAME


def load_adapter_weights(encoder: Any, directory: str | Path) -> None:
    """Load :data:`ADAPTER_WEIGHTS_NAME` into an already-wrapped encoder."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    path = Path(directory) / ADAPTER_WEIGHTS_NAME
    if not path.is_file():
        raise AdaptError(f"adapter weights not found: {path}")
    set_peft_model_state_dict(encoder, load_file(str(path)))


def merge_encoder(model: Any) -> None:
    """Fold the adapters into the weights: ``model.encoder`` becomes plain again."""
    merged = model.encoder.merge_and_unload()
    model.encoder = merged
