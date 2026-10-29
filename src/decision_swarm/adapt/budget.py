"""Optimizer-budget numbers for a training run.

The estimate is pure Python (the formulas ``laya.train`` uses) so
``adapt validate`` can print it anywhere.  :func:`probe_dry_run` runs
``laya.train.dry_run`` inside a training env when one exists — the exact
counts, straight from the model stack that will train.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

CALIB_MAX = 400
CALIB_FRAC = 0.1
CALIB_SEED = 20260922
"""``laya.train.TrainConfig`` defaults: the held-out calibration slice."""

MICRO_BATCH = 8
GRAD_ACCUM = 8
EPOCHS = 4
"""``laya.train.TrainConfig`` training-batch defaults."""


def split_counts(n_items: int) -> tuple[int, int]:
    """``(train, calibration)`` item counts, as ``split_calibration`` draws them."""
    n_calib = min(CALIB_MAX, int(n_items * CALIB_FRAC))
    return n_items - n_calib, n_calib


def optimizer_updates(
    n_items: int,
    micro_batch: int = MICRO_BATCH,
    grad_accum: int = GRAD_ACCUM,
    epochs: int = EPOCHS,
) -> int:
    """Optimizer steps for ``n_items`` training items (mirrors ``laya.train``)."""
    if n_items < 1:
        raise ValueError(f"n_items must be a positive integer, got {n_items!r}")
    steps_per_epoch = math.ceil(n_items / micro_batch)
    updates_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    return updates_per_epoch * epochs


def budget_message(
    n_train: int,
    micro_batch: int = MICRO_BATCH,
    grad_accum: int = GRAD_ACCUM,
    epochs: int = EPOCHS,
) -> str:
    """The pre-run budget line, worded like ``laya.train.update_budget_message``."""
    updates = optimizer_updates(n_train, micro_batch, grad_accum, epochs)
    return (
        f"optimizer budget: {n_train} train item(s), effective batch up to {micro_batch * grad_accum} "
        f"(micro_batch {micro_batch} x grad_accum {grad_accum}), {updates} optimizer updates "
        f"over {epochs} epoch(s)"
    )


def _source_root() -> str:
    """Directory that makes ``decision_swarm`` importable in a training env."""
    return str(Path(__file__).resolve().parents[2])


def probe_dry_run(
    python: Path,
    data: str | Path,
    base_dir: str | Path,
    *,
    eval_data: str | Path | None = None,
) -> dict[str, Any] | None:
    """``laya.train.dry_run`` summary from a training env, or ``None`` on failure.

    Runs a throwaway interpreter in the env (which has the model stack) with
    this repo on ``PYTHONPATH``; nothing about the caller's own interpreter
    matters.
    """
    command = [
        str(python),
        "-m",
        "decision_swarm.adapt.budget",
        "--data",
        str(data),
        "--base-dir",
        str(base_dir),
    ]
    if eval_data is not None:
        command += ["--eval", str(eval_data)]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _source_root() + (os.pathsep + existing if existing else "")
    try:
        result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        summary = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None
    return summary if isinstance(summary, dict) else None


def main(argv: list[str] | None = None) -> int:
    """Probe entry point: ``python -m decision_swarm.adapt.budget`` in a training env."""
    import argparse

    parser = argparse.ArgumentParser(prog="decision-swarm-adapt-budget")
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--eval", dest="eval_data", default=None)
    args = parser.parse_args(argv)
    try:
        from laya.train import TrainConfig, dry_run
    except ImportError as exc:
        print(f"decision-swarm-adapt-budget: laya is not installed in this env: {exc}", file=sys.stderr)
        return 2
    config = TrainConfig()
    if args.eval_data:
        config.eval_data = args.eval_data
    summary = dry_run(args.data, args.base_dir, config)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
