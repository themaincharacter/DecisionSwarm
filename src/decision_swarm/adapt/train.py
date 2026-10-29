"""One adaptation training run — orchestrated here, trained by ``laya.train``.

Spawned as ``python -m decision_swarm.adapt.train --config <json>`` by
``decision-swarm adapt train``, inside the training env (which has the model
stack).  Nothing is reimplemented from ``laya.train``: this file loads the
checkpoint, optionally wraps the encoder with LoRA (:mod:`decision_swarm.adapt.
lora`), calls ``train_model`` / ``calibration_records`` / the ``laya.calibrate``
fitters, then exports the artifacts the plan defines::

    out/
    ├── adapter/          # adapter.safetensors + adapter_config.json + questions.json
    ├── merged/           # full checkpoint, served unchanged by laya.load
    ├── questions.json    # the trained question schema (optional defaults)
    └── train_report.json # before/after deltas, metric mapping, provenance
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

from decision_swarm.adapt import AdaptError

MODES = ("lora", "head-only", "full")
ADAPTER_FORMAT = "decision-swarm-adapter/1"


# ----------------------------------------------------------------- pure helpers


def questions_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """First-seen question schema per id — the trained defaults, as laya saves it."""
    questions: dict[str, Any] = {}
    for row in rows:
        raw = row.get("questions")
        if not isinstance(raw, dict):
            continue
        for qid, qdef in raw.items():
            if qid not in questions:
                questions[qid] = qdef
    return questions


def build_adapter_config(
    peft_config: dict[str, Any],
    *,
    base: dict[str, Any],
    trainable_params: int,
    total_params: int,
    seeds: dict[str, int],
    questions_file: str | None = None,
    train_report: str = "../train_report.json",
) -> dict[str, Any]:
    """Community (peft) adapter config plus this tool's provenance block.

    The peft fields stay at the top level so any PEFT loader can read the file;
    ``decision_swarm`` carries what PEFT does not know: the pinned base (repo +
    revision + weights sha256), trainable counts, and the seeds the run used.
    """
    return {
        **peft_config,
        "decision_swarm": {
            "format": ADAPTER_FORMAT,
            "base": base,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "seeds": seeds,
            "questions": questions_file,
            "train_report": train_report,
        },
    }


def auto_micro_batch(device: Any) -> int:
    """Micro-batch sized to detected VRAM (the plan's table: 8 GB suffices at 1)."""
    if getattr(device, "type", "") == "cuda":
        import torch

        total = torch.cuda.get_device_properties(device.index or 0).total_memory
        if total >= 20 * 1024**3:
            return 8
        if total >= 12 * 1024**3:
            return 4
        return 1
    return 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------- the run


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Train, calibrate, export; return the train report (also written to disk)."""
    from laya import calibrate
    from laya import train as laya_train

    from decision_swarm.adapt import lora as lora_mod

    started = time.time()
    mode = config.get("mode", "lora")
    if mode not in MODES:
        raise AdaptError(f"mode must be one of {list(MODES)}, got {mode!r}")
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    device = laya_train.resolve_device(config.get("device") or "auto")

    model, tok, base_cfg = laya_train.load_checkpoint(config["base_dir"])
    model.to(device)
    max_len = config.get("max_len") or base_cfg.get("max_len", 512)
    head_max_len = config.get("head_max_len") or base_cfg.get("head_max_len", 192)

    rows = laya_train.read_data(
        config["data"],
        text_column=config.get("text_column", "text"),
        label_column=config.get("label_column", "label"),
        question_id=config.get("question_id", "label"),
        instructions=config.get("instructions"),
    )
    items, skipped = laya_train.items_from_rows(tok, rows, max_len, head_max_len)
    if not items:
        raise AdaptError(f"{config['data']} produced no training items (skipped: {skipped!r})")
    train_items, calib_items = laya_train.split_calibration(items, 400, 0.1, 20260922)

    micro_batch = config.get("micro_batch") or auto_micro_batch(device)
    train_cfg = laya_train.TrainConfig(
        epochs=config.get("epochs", 4),
        micro_batch=micro_batch,
        grad_accum=config.get("grad_accum", 8),
        encoder_lr=config.get("encoder_lr", 2.5e-5),
        head_lr=config.get("head_lr", 1e-4),
        loss=config.get("loss", "rlcd"),
        seed=config.get("seed", 0),
        freeze_encoder=(mode == "head-only"),
    )
    train_cfg.validate()
    seeds = {"train": train_cfg.seed, "calibration": 20260922}

    # Evaluation set: --eval held out, else the in-sample calibration slice.
    eval_items: list[Any] = []
    eval_skipped: dict[str, int] = {}
    eval_source = eval_mode = None
    is_held_out = False
    eval_note = "No evaluation performed."
    if config.get("eval"):
        eval_rows = laya_train.read_data(
            config["eval"],
            text_column=config.get("text_column", "text"),
            label_column=config.get("label_column", "label"),
            question_id=config.get("question_id", "label"),
            instructions=config.get("instructions"),
        )
        eval_items, eval_skipped = laya_train.items_from_rows(tok, eval_rows, max_len, head_max_len)
        eval_source = config["eval"]
        if eval_items:
            seen = {
                (tuple(it["state_ids"]), json.dumps(it["q"], sort_keys=True))
                for it in train_items + calib_items
            }
            overlap = sum(
                1
                for it in eval_items
                if (tuple(it["state_ids"]), json.dumps(it["q"], sort_keys=True)) in seen
            )
            if overlap:
                eval_mode = "overlapping_eval"
                eval_note = (
                    f"Evaluation set contains inputs that overlap training data "
                    f"({overlap}/{len(eval_items)} items); metrics do not reflect "
                    f"strictly independent held-out evaluation."
                )
            else:
                eval_mode = "held_out"
                is_held_out = True
                eval_note = "Metrics reflect independent held-out evaluation."
        else:
            eval_note = f"Evaluation file {config['eval']} yielded 0 usable items (skipped: {eval_skipped!r})."
    elif calib_items:
        eval_items = calib_items
        eval_source = "calibration_slice"
        eval_mode = "in_sample_calibration"
        eval_note = (
            "Metrics reflect in-sample calibration fit, not independent held-out "
            "evaluation. Provide --eval for held-out validation."
        )

    print(
        "adapt: %d train items, %d calibration items, %d eval items, skipped %r, device %s"
        % (len(train_items), len(calib_items), len(eval_items), skipped, device),
        flush=True,
    )
    print(laya_train.update_budget_message(len(train_items), train_cfg), flush=True)

    warnings_seen: list[str] = []
    before_eval = None
    if eval_items:
        base_records = laya_train.calibration_records(model, tok, eval_items, device, max_len, head_max_len)
        before_eval = laya_train.evaluate_records(
            base_records, base_cfg.get("temperature"), base_cfg.get("temperature_by_options")
        )

    if mode == "lora":
        peft_config = lora_mod.wrap_encoder(
            model,
            rank=config.get("rank", 8),
            alpha=config.get("lora_alpha", 16),
            dropout=config.get("lora_dropout", 0.05),
        )
        trainable, total = lora_mod.trainable_parameters(model)
        print(f"adapt: LoRA wrap — {trainable:,} trainable of {total:,} parameters", flush=True)

    history = laya_train.train_model(model, tok, train_items, train_cfg, device, max_len, head_max_len)

    records = laya_train.calibration_records(model, tok, calib_items, device, max_len, head_max_len)
    collapse_msg = laya_train.prior_collapse_message(records)
    if collapse_msg:
        warnings.warn(collapse_msg, RuntimeWarning, stacklevel=2)
        warnings_seen.append(collapse_msg)
    fitted = calibrate.fit_temperature_map(records)
    calibration = laya_train.calibration_report(records, fitted["temperature"])
    try:
        abstention = calibrate.fit_abstention_thresholds(
            records,
            temperature=fitted["temperature"],
            temperature_by_options=fitted["temperature_by_options"],
            target_error=train_cfg.target_error,
            min_bucket_n=train_cfg.min_abstain_n,
        )
    except Exception as exc:  # a failed abstention fit never sinks the run (finetune parity)
        abstention = {}
        warnings_seen.append(f"failed to fit abstention thresholds: {exc}")

    after_eval = comparison = None
    if eval_items:
        after_records = laya_train.calibration_records(model, tok, eval_items, device, max_len, head_max_len)
        after_eval = laya_train.evaluate_records(
            after_records, fitted["temperature"], fitted["temperature_by_options"]
        )
        comparison = {
            f"delta_{key}": round(after_eval[key] - before_eval[key], 4)
            for key in ("accuracy", "loss", "mean_confidence")
            if before_eval is not None and after_eval.get(key) is not None and before_eval.get(key) is not None
        }

    base = dict(config.get("base") or {})
    questions = questions_from_rows(rows)
    questions_file = None
    if questions:
        (out / "questions.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
        questions_file = "questions.json"

    report: dict[str, Any] = {
        "format": "decision-swarm-train-report/1",
        "base": base,
        "mode": mode,
        "train_items": len(train_items),
        "calibration_items": len(calib_items),
        "eval_items": len(eval_items),
        "skipped": skipped,
        "eval_skipped": eval_skipped,
        "eval_source": eval_source,
        "eval_mode": eval_mode,
        "is_held_out": is_held_out,
        "eval_note": eval_note,
        "before": before_eval,
        "after": after_eval,
        "comparison": comparison,
        "calibration": calibration,
        "abstention_thresholds": abstention,
        "epoch_loss": history,
        "training": {
            "epochs": train_cfg.epochs,
            "micro_batch": train_cfg.micro_batch,
            "grad_accum": train_cfg.grad_accum,
            "loss": train_cfg.loss,
            "seed": train_cfg.seed,
            "freeze_encoder": train_cfg.freeze_encoder,
        },
        "provenance": config.get("provenance", {}),
        "metric_map": config.get("metric_map"),
        "warnings": warnings_seen,
    }

    # Adapter first (the wrap must still be present), then merge and save the
    # full checkpoint so load_laya serves it unchanged.
    if mode == "lora":
        adapter_dir = out / "adapter"
        lora_mod.save_adapter(model.encoder, adapter_dir)
        if questions_file:
            (adapter_dir / questions_file).write_text(json.dumps(questions, indent=2), encoding="utf-8")
        weights_sha = _sha256(Path(config["base_dir"]) / "model.safetensors")
        adapter_config = build_adapter_config(
            json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8")),
            base={**base, "weights_sha256": weights_sha},
            trainable_params=trainable,
            total_params=total,
            seeds=seeds,
            questions_file=questions_file,
        )
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=2), encoding="utf-8"
        )
        lora_mod.merge_encoder(model)
        report["artifacts"] = {"adapter": "adapter", "merged": "merged", "questions": questions_file}
    else:
        report["artifacts"] = {"adapter": None, "merged": "merged", "questions": questions_file}

    out_cfg = dict(
        base_cfg,
        max_len=max_len,
        head_max_len=head_max_len,
        fine_tuned=True,
        temperature=fitted["temperature"],
    )
    out_cfg.pop("temperature_by_options", None)
    if fitted["temperature_by_options"]:
        out_cfg["temperature_by_options"] = fitted["temperature_by_options"]
    out_cfg["training"] = dict(out_cfg.get("training") or {}, decision_swarm_adapt=report["training"])
    laya_train.save_checkpoint(model, tok, out_cfg, str(out / "merged"))

    report["wall_clock_s"] = round(time.time() - started, 1)
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="decision-swarm-adapt-train")
    parser.add_argument("--config", required=True, help="run configuration (JSON)")
    args = parser.parse_args(argv)
    try:
        report = run(json.loads(args.config))
    except ImportError as exc:
        print(
            f"decision-swarm-adapt-train: this command runs inside the training env: {exc}",
            file=sys.stderr,
        )
        return 2
    except AdaptError as exc:
        print(f"decision-swarm-adapt-train: {exc}", file=sys.stderr)
        return 2
    print(
        "adapt: done — %s (%d train items, %d eval items, wall-clock %.0fs)"
        % (
            report["artifacts"]["merged"],
            report["train_items"],
            report["eval_items"],
            report["wall_clock_s"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
