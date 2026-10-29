"""The \"better at X\" gate: per-slice accuracy against a baseline.

Metric computation runs in the training env (it loads checkpoints via
``laya.train`` machinery — the upstream deltas ``evaluate_records`` reports).
The gate arithmetic itself is pure, so ``adapt register`` can refuse on a saved
report without any model stack.

Gates (defaults from the plan):
- **target**: accuracy on the target slice must improve by ``--min-delta`` (0.05)
  over the baseline — the X the specialization claims.
- **no-regression**: every other slice may lose at most ``--max-regression``
  (0.02) accuracy.
- **must-not-regress**: the shipped guard set (``data/adapt/must-not-regress
  .jsonl``) may lose at most ``--max-regression`` accuracy — failing it blocks
  registration even when X improved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from decision_swarm.adapt import AdaptError

REPORT_FORMAT = "decision-swarm-evaluate-report/1"
DEFAULT_MIN_DELTA = 0.05
DEFAULT_MAX_REGRESSION = 0.02
_EPS = 1e-9  # thresholds hold at equality (0.70 - 0.72 is not exactly -0.02)
GUARD_ENV = "DECISION_SWARM_GUARD"
GUARD_RELATIVE = ("data", "adapt", "must-not-regress.jsonl")


def default_guard_path() -> Path:
    """The shipped must-not-regress set: env override, bundled copy, repo checkout."""
    env_path = os.environ.get(GUARD_ENV)
    if env_path:
        return Path(env_path)
    module_dir = Path(__file__).resolve().parent
    bundled = module_dir.parent.joinpath(*GUARD_RELATIVE)
    if bundled.is_file():
        return bundled
    for parent in module_dir.parents:
        candidate = parent.joinpath(*GUARD_RELATIVE)
        if candidate.is_file():
            return candidate
    raise AdaptError(
        f"{Path(*GUARD_RELATIVE)} not found near {__file__}; set {GUARD_ENV} to a guard dataset"
    )


# ----------------------------------------------------------------- pure gate


def check_gates(
    slices: dict[str, dict[str, Any]],
    *,
    target_slice: str,
    guard: dict[str, Any] | None,
    min_delta: float = DEFAULT_MIN_DELTA,
    max_regression: float = DEFAULT_MAX_REGRESSION,
) -> dict[str, Any]:
    """Apply the three gates to per-slice ``{accuracy, baseline_accuracy, ...}`` maps.

    Returns ``{"pass": bool, "gates": [{name, passed, detail}, ...]}``.
    """
    gates: list[dict[str, Any]] = []

    target = slices.get(target_slice)
    if target is None:
        gates.append(
            {"name": "target", "passed": False,
             "detail": f"slice {target_slice!r} has no eval items — the X slice must be tagged in the data"}
        )
    elif target.get("baseline_accuracy") is None:
        gates.append(
            {"name": "target", "passed": False,
             "detail": f"slice {target_slice!r} has no baseline accuracy to compare against"}
        )
    else:
        delta = target["accuracy"] - target["baseline_accuracy"]
        gates.append(
            {"name": "target", "passed": delta >= min_delta - _EPS,
             "detail": f"slice {target_slice!r} accuracy {target['accuracy']:.4f} vs baseline "
                       f"{target['baseline_accuracy']:.4f} (delta {delta:+.4f}, needed >= {min_delta})"}
        )

    regressions = []
    for name, entry in sorted(slices.items()):
        if name == target_slice or entry.get("baseline_accuracy") is None:
            continue
        delta = entry["accuracy"] - entry["baseline_accuracy"]
        if delta < -max_regression - _EPS:
            regressions.append(f"{name} {delta:+.4f}")
    gates.append(
        {"name": "no-regression", "passed": not regressions,
         "detail": "no slice regressed beyond -%.2f" % max_regression if not regressions
         else "slices regressed beyond -%.2f: %s" % (max_regression, ", ".join(regressions))}
    )

    if guard is None or guard.get("baseline_accuracy") is None:
        gates.append(
            {"name": "must-not-regress", "passed": False,
             "detail": "the must-not-regress set has no baseline accuracy — run evaluate with the base checkpoint"}
        )
    else:
        delta = guard["accuracy"] - guard["baseline_accuracy"]
        gates.append(
            {"name": "must-not-regress", "passed": delta >= -max_regression - _EPS,
             "detail": f"guard accuracy {guard['accuracy']:.4f} vs baseline {guard['baseline_accuracy']:.4f} "
                       f"(delta {delta:+.4f}, floor {-max_regression:+.4f})"}
        )

    return {"pass": all(gate["passed"] for gate in gates), "gates": gates}


# ----------------------------------------------------------------- metrics


def _slice_rows(rows: list[dict[str, Any]], target_slice: str) -> dict[str, list[dict[str, Any]]]:
    """Eval rows per slice: the whole set plus one slice per tag (the target slice included)."""
    grouped: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for row in rows:
        tags = row.get("tags")
        if not isinstance(tags, list):
            continue
        for tag in tags:
            grouped.setdefault(str(tag), []).append(row)
    if target_slice not in grouped:
        grouped.setdefault(target_slice, [])
    return grouped


def _accuracy(laya_train: Any, model: Any, tok: Any, rows: list[dict[str, Any]],
              device: Any, max_len: int, head_max_len: int, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """``{items, accuracy}`` for one row group through upstream evaluation."""
    items, skipped = laya_train.items_from_rows(tok, rows, max_len, head_max_len)
    if not items:
        return None
    records = laya_train.calibration_records(model, tok, items, device, max_len, head_max_len)
    metrics = laya_train.evaluate_records(
        records, cfg.get("temperature"), cfg.get("temperature_by_options")
    )
    return {"items": metrics["items"], "accuracy": metrics["accuracy"]}


def _load_rows(path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    from laya import train as laya_train

    return laya_train.read_data(
        path,
        text_column=config.get("text_column", "text"),
        label_column=config.get("label_column", "label"),
        question_id=config.get("question_id", "label"),
        instructions=config.get("instructions"),
    )


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Compute per-slice metrics for the merged (and baseline) models, apply the gates."""
    from laya import train as laya_train

    merged_dir = Path(config["merged"])
    rows = _load_rows(config["data"], config)
    target_slice = config.get("slice") or "all"
    min_delta = config.get("min_delta", DEFAULT_MIN_DELTA)
    max_regression = config.get("max_regression", DEFAULT_MAX_REGRESSION)

    device = laya_train.resolve_device(config.get("device") or "auto")
    merged_model, tok, merged_cfg = laya_train.load_checkpoint(str(merged_dir))
    merged_model.to(device)
    max_len = merged_cfg.get("max_len", 512)
    head_max_len = merged_cfg.get("head_max_len", 192)

    grouped = _slice_rows(rows, target_slice)
    slices: dict[str, dict[str, Any]] = {}
    for name, group in grouped.items():
        entry = _accuracy(laya_train, merged_model, tok, group, device, max_len, head_max_len, merged_cfg)
        slices[name] = entry or {"items": 0, "accuracy": 0.0, "baseline_accuracy": None}

    guard: dict[str, Any] | None = None
    guard_rows = _load_rows(config["guard"], config) if config.get("guard") else []

    baseline_cfg: dict[str, Any] | None = None
    if config.get("baseline_dir"):
        base_model, _tok, baseline_cfg = laya_train.load_checkpoint(config["baseline_dir"])
        base_model.to(device)
        for name, group in grouped.items():
            entry = _accuracy(laya_train, base_model, tok, group, device, max_len, head_max_len, baseline_cfg)
            if entry is not None:
                slices[name]["baseline_accuracy"] = entry["accuracy"]
        if guard_rows:
            entry = _accuracy(laya_train, base_model, tok, guard_rows, device, max_len, head_max_len, baseline_cfg)
            guard = {"items": entry["items"], "accuracy": None, "baseline_accuracy": entry["accuracy"]} if entry else None
        del base_model
    elif config.get("baseline_report"):
        baseline = json.loads(Path(config["baseline_report"]).read_text(encoding="utf-8"))
        for name, entry in slices.items():
            prior = (baseline.get("slices") or {}).get(name)
            if isinstance(prior, dict) and prior.get("accuracy") is not None:
                entry["baseline_accuracy"] = prior["accuracy"]
        prior_guard = baseline.get("guard") or {}
        guard = {
            "items": prior_guard.get("items"),
            "accuracy": None,
            "baseline_accuracy": prior_guard.get("accuracy"),
        }

    if guard_rows:
        entry = _accuracy(laya_train, merged_model, tok, guard_rows, device, max_len, head_max_len, merged_cfg)
        if guard is None:
            guard = {"items": None, "accuracy": None, "baseline_accuracy": None}
        if entry is not None:
            guard["items"] = entry["items"]
            guard["accuracy"] = entry["accuracy"]

    gates = check_gates(
        slices, target_slice=target_slice, guard=guard, min_delta=min_delta, max_regression=max_regression
    )
    report = {
        "format": REPORT_FORMAT,
        "merged": str(merged_dir),
        "data": config["data"],
        "slice": target_slice,
        "min_delta": min_delta,
        "max_regression": max_regression,
        "metric_map": config.get("metric_map"),
        "slices": slices,
        "guard": guard,
        "gates": gates["gates"],
        "pass": gates["pass"],
    }
    out = Path(config.get("out") or (merged_dir.parent / "evaluate_report.json"))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="decision-swarm-adapt-evaluate")
    parser.add_argument("--config", required=True, help="run configuration (JSON)")
    args = parser.parse_args(argv)
    try:
        report = run(json.loads(args.config))
    except ImportError as exc:
        print(f"decision-swarm-adapt-evaluate: this command runs inside the training env: {exc}", file=sys.stderr)
        return 2
    except AdaptError as exc:
        print(f"decision-swarm-adapt-evaluate: {exc}", file=sys.stderr)
        return 2
    for gate in report["gates"]:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"gate {gate['name']}: {status} — {gate['detail']}", flush=True)
    print("evaluate: %s" % ("gate PASSED" if report["pass"] else "gate FAILED"), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
