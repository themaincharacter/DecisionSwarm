"""``decision-swarm`` command-line entrypoint.

The command surface is defined here: ``inspect``, ``fetch``, ``install``,
``run`` (one observation), ``serve`` (long-lived decision service over stdio),
``build``, ``adapt`` (LoRA adaptation), and ``models``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from decision_swarm import __version__, agent, cache, envs, manifest, pack, query, registry
from decision_swarm.adapt import AdaptError, TRAINING_PACKAGES, training_env
from decision_swarm.adapt import budget as adapt_budget
from decision_swarm.adapt import dataset as adapt_dataset
from decision_swarm.adapt import evaluate as adapt_evaluate
from decision_swarm.adapt import register as adapt_register
from decision_swarm.runtime import loaders, mcp, orchestrator, serve, worker


def adapt_error(message: str) -> AdaptError:
    return AdaptError(message)


def _models_root(args: argparse.Namespace) -> str | Path | None:
    """``--models-dir``, else a ``models/`` directory bundled beside the pack.

    An exported bundle carries its weights next to the pack, so unpacking one is
    enough: no shared cache and no extra flag.
    """
    explicit = getattr(args, "models_dir", None)
    if explicit:
        return explicit
    pack_arg = getattr(args, "pack", None)
    if pack_arg:
        pack_dir = Path(pack_arg)
        bundled = pack_dir / pack.BUNDLED_MODELS_DIR
        if pack_dir.is_dir() and bundled.is_dir():
            return bundled
    return None


def _envs_root(args: argparse.Namespace) -> str | Path | None:
    """``--envs-dir``, else an ``envs/`` directory bundled beside the pack."""
    explicit = getattr(args, "envs_dir", None)
    if explicit:
        return explicit
    pack_arg = getattr(args, "pack", None)
    if pack_arg:
        pack_dir = Path(pack_arg)
        bundled = pack_dir / pack.BUNDLED_ENVS_DIR
        if pack_dir.is_dir() and bundled.is_dir():
            return bundled
    return None


def _add_pack_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pack", required=True, metavar="DIR", help="path to the swarm pack directory")


def _add_models_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-dir",
        metavar="DIR",
        help="model cache location (default: ~/.decision-swarm/models, or $DECISION_SWARM_MODELS)",
    )


def _add_envs_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--envs-dir",
        metavar="DIR",
        help="env location (default: ~/.decision-swarm/envs, or $DECISION_SWARM_ENVS)",
    )


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Print cells, models, sizes, and cache status for a pack."""
    models_root = _models_root(args)
    entries = registry.load_registry()
    if args.json:
        payload = agent.describe_pack(args.pack, entries=entries, models_root=models_root)
        print(json.dumps(payload, indent=2))
        return 0
    swarm = manifest.load_manifest(args.pack)
    models = pack.collect_models(swarm, models_root=models_root, registry_entries=entries)
    print(f"pack: {swarm.name}")
    print("views: " + ", ".join(f"{view.name} ({view.type})" for view in swarm.views))
    print("cells:")
    for cell in swarm.cells:
        questions = ", ".join(question.name for question in cell.default_questions) or "-"
        print(
            f"  {cell.id}  {cell.model}  [{cell.modality}]"
            f"  views: {', '.join(cell.views)}  questions: {questions}"
        )
    print("models:")
    for model in models:
        checkpoint = model.hf_repo + (f"/{model.subfolder}" if model.subfolder else "")
        estimate = f"~{model.download_bytes // 1_000_000} MB" if model.download_bytes else "?"
        cached = f"yes ({model.bytes} bytes)" if model.cached else "no"
        print(f"  {model.id}  {checkpoint}  {model.params or '?'}  {estimate}  cached: {cached}")
    print(f"cache: {cache.models_dir(models_root)}")
    return 0


def _materialize_adapters(
    swarm: manifest.SwarmManifest,
    *,
    models_root: str | None = None,
    envs_root: str | None = None,
    registry_entries: Mapping[str, registry.ModelEntry] | None = None,
    build_envs: bool = False,
) -> list[str]:
    """Materialize merged checkpoints for adapter cells; returns notes to report."""
    notes: list[str] = []
    for cell in swarm.cells:
        if not cell.adapter:
            continue
        try:
            base = registry.resolve_model(cell.model, registry_entries)
            merged = registry.resolve_checkpoint(cell, registry_entries)
        except registry.RegistryError as exc:
            notes.append(f"cell {cell.id!r}: {exc}")
            continue
        if cache.is_cached(merged, models_root):
            continue
        adapter_entry = registry.adapter_entry_for(cell.adapter, registry_entries)
        if adapter_entry is None or not adapter_entry.path:
            notes.append(
                f"cell {cell.id!r}: adapter files for {cell.adapter!r} are not registered; "
                f"cannot materialize the merged checkpoint"
            )
            continue
        if not base.package_source:
            notes.append(f"cell {cell.id!r}: base {base.id!r} has no package_source")
            continue
        spec = loaders.ModelSpec(
            model_id=base.id,
            hf_repo=base.hf_repo,
            subfolder=base.subfolder,
            revision=base.revision or "main",
            models_dir=str(cache.models_dir(models_root)),
        )
        base_dir = spec.local_path
        if base_dir is None:
            notes.append(f"cell {cell.id!r}: base {base.id!r} is not in the cache; run 'decision-swarm fetch'")
            continue
        env = training_env(base.package_source, envs_root=envs_root)
        if not env.python.exists():
            if not build_envs:
                notes.append(
                    f"cell {cell.id!r}: run 'decision-swarm install' to materialize its merged checkpoint"
                )
                continue
            envs.ensure_venv(env)
            envs.install_packages(env, [base.package_source, *TRAINING_PACKAGES])
        config = {
            "base_dir": str(base_dir),
            "adapter_dir": adapter_entry.path,
            "out_dir": str(cache.snapshot_dir(merged, models_root)),
        }
        code = _run_adapt_module(env.python, "decision_swarm.adapt.merge", config)
        if code != 0:
            notes.append(f"cell {cell.id!r}: merging its adapter failed (exit {code})")
            continue
        cache.record_local(merged, models_root)
        notes.append(f"cell {cell.id!r}: materialized {registry.merged_ref(merged)}")
    return notes


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Download a pack's models into the shared cache (idempotent)."""
    swarm = manifest.load_manifest(args.pack or args.manifest)
    entries = registry.load_registry()
    failures = 0
    seen: set[tuple] = set()
    for cell in swarm.cells:
        try:
            entry = registry.resolve_model(cell.model, entries)
        except registry.RegistryError as exc:
            print(f"decision-swarm: error: {exc}", file=sys.stderr)
            failures += 1
            continue
        key = (entry.hf_repo, entry.revision, entry.subfolder)
        if key in seen:
            continue
        seen.add(key)
        checkpoint = entry.hf_repo + (f"/{entry.subfolder}" if entry.subfolder else "")
        try:
            result = cache.fetch(entry, root=args.models_dir)
        except (cache.CacheError, OSError) as exc:
            print(f"decision-swarm: error: {exc}", file=sys.stderr)
            failures += 1
            continue
        state = "downloaded" if result.downloaded else "already cached"
        print(f"{entry.id}  {checkpoint}  {state}  ({result.bytes} bytes)")
    for note in _materialize_adapters(swarm, models_root=args.models_dir, registry_entries=entries):
        print(f"decision-swarm: {note}", file=sys.stderr)
    return 1 if failures else 0


def _cmd_install(args: argparse.Namespace) -> int:
    """Build one virtual environment per unique package source."""
    swarm = manifest.load_manifest(args.pack)
    entries = registry.load_registry()
    planned = envs.cell_envs(swarm, envs_root=args.envs_dir, registry_entries=entries)
    for env in planned:
        existed = env.python.exists()
        envs.build_env(env)
        state = "reused" if existed else "built"
        print(f"{state}  {env.package_source}  [{env.key}]  cells: {', '.join(env.shared_by)}")
    for note in _materialize_adapters(
        swarm, models_root=args.models_dir, envs_root=args.envs_dir, registry_entries=entries, build_envs=True
    ):
        print(note)
    return 0


def _print_pretty(result: query.SwarmResult) -> None:
    for cell_id, cell in result.cells.items():
        print(f"{cell_id}  ({cell.model}, {cell.latency_ms:.0f} ms)")
        for name, answer in cell.answers.items():
            print(f"  {name}: {answer.type}={answer.value}  confidence={answer.confidence:.2f}")
        for warning in cell.warnings:
            print(f"  warning: {warning}")


def _require_cached(
    swarm: manifest.SwarmManifest,
    args: argparse.Namespace,
    entries: Mapping[str, registry.ModelEntry],
) -> None:
    """Fail fast under ``--offline`` when a model is missing from the cache."""
    if not args.offline:
        return
    missing = [
        model.id
        for model in pack.collect_models(swarm, models_root=_models_root(args), registry_entries=entries)
        if not model.cached
    ]
    if missing:
        raise cache.CacheError(f"--offline: model(s) not in the cache: {', '.join(missing)}")


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a pack against one observation."""
    swarm = manifest.load_manifest(args.pack)
    payload = query.load_query(args.input, swarm)
    entries = registry.load_registry()
    _require_cached(swarm, args, entries)
    result = orchestrator.run(
        swarm,
        payload,
        registry_entries=entries,
        envs_root=args.envs_dir,
        models_root=_models_root(args),
    )
    if args.pretty:
        _print_pretty(result)
    elif args.detail == "compact":
        print(json.dumps(result.to_compact(), indent=2))
    else:
        print(json.dumps(result.to_dict(), indent=2))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Serve a pack as a long-lived decision service over stdio (JSONL)."""
    swarm = manifest.load_manifest(args.pack)
    entries = registry.load_registry()
    _require_cached(swarm, args, entries)
    plans = orchestrator.plan_cells(
        swarm,
        registry_entries=entries,
        envs_root=_envs_root(args),
        models_root=_models_root(args),
    )
    print(f"serving {swarm.name}: {len(swarm.cells)} cells over stdio", file=sys.stderr)
    with orchestrator.Swarm(plans) as warm:
        return serve.serve_loop(warm, swarm)


def _cmd_mcp(args: argparse.Namespace) -> int:
    """Serve packs as MCP tools over stdio, for an agent host to spawn."""
    if args.print_config:
        if not args.pack:
            print("decision-swarm: --print-config needs --pack", file=sys.stderr)
            return 2
        print(json.dumps(pack.mcp_config(args.pack), indent=2))
        return 0
    config = agent.AgentConfig(
        packs_dirs=tuple(Path(raw) for raw in (args.packs_dir or ())),
        default_pack=Path(args.pack) if args.pack else None,
        allow_unvetted=args.allow_unvetted,
        max_warm=args.max_warm,
        models_root=_models_root(args),
        envs_root=_envs_root(args),
    )
    tools = agent.AgentTools(config)
    print(
        f"decision-swarm mcp: {len(tools.tools())} tools over stdio",
        file=sys.stderr,
    )
    try:
        return mcp.serve(tools.tools(), server_version=__version__)
    finally:
        tools.close()


def _cmd_build(args: argparse.Namespace) -> int:
    """Build a swarm pack from a manifest (what the GUI's Submit calls)."""
    swarm = manifest.load_manifest(args.manifest)
    entries = registry.load_registry()
    resolver = None if args.no_pin else pack.hub_commit_resolver
    models = pack.collect_models(
        swarm,
        models_root=args.models_dir,
        registry_entries=entries,
        resolve_commit=resolver,
    )
    dest = pack.build_pack(
        swarm,
        args.dest,
        models_root=args.models_dir,
        registry_entries=entries,
        models=models,
    )

    bundle: str | None = None
    include_envs = not args.no_envs
    if args.export:
        if include_envs:
            unbuilt = pack.missing_envs(dest, envs_root=_envs_root(args))
            if unbuilt:
                print(
                    "decision-swarm: warning: bundle carries no envs for: "
                    + ", ".join(unbuilt)
                    + " (run 'decision-swarm install' first to include them)",
                    file=sys.stderr,
                )
        try:
            bundle = str(
                pack.export_bundle(
                    dest,
                    models_root=args.models_dir,
                    envs_root=_envs_root(args),
                    include_envs=include_envs,
                )
            )
        except pack.PackError as exc:
            print(
                f"decision-swarm: warning: no bundle written: {exc} "
                "(the pack directory itself is complete)",
                file=sys.stderr,
            )

    summary = {
        "pack": str(dest),
        "bundle": bundle,
        "name": swarm.name,
        "cells": [cell.id for cell in swarm.cells],
        "models": [
            {
                "id": model.id,
                "hf_repo": model.hf_repo,
                "subfolder": model.subfolder,
                "revision": model.revision,
                "commit": model.commit,
                "cached": model.cached,
                "bytes": model.bytes,
            }
            for model in models
        ],
    }
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"built  {dest}  cells: {', '.join(summary['cells'])}")
        for model in models:
            state = "cached" if model.cached else "not cached"
            print(f"  {model.id}  {state}  ({model.bytes} bytes)")
        if bundle:
            print(f"bundle  {bundle}  (pack + weights)")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    """Bundle a pack with the weights and envs it needs into one tar."""
    include_envs = not args.no_envs
    if include_envs:
        unbuilt = pack.missing_envs(args.pack, envs_root=_envs_root(args))
        if unbuilt:
            print(
                "decision-swarm: warning: bundle carries no envs for: "
                + ", ".join(unbuilt)
                + " (run 'decision-swarm install --pack ...' first to include them)",
                file=sys.stderr,
            )
    dest = pack.export_bundle(
        args.pack,
        args.dest,
        models_root=_models_root(args),
        envs_root=_envs_root(args),
        include_envs=include_envs,
    )
    print(f"bundled  {dest}  ({dest.stat().st_size} bytes)")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    """List known models (the GUI's model dropdown source)."""
    entries = registry.load_registry()
    if args.json:
        print(json.dumps({"models": [asdict(entry) for entry in entries.values()]}, indent=2))
        return 0
    for entry in entries.values():
        print(
            f"{entry.id}  {entry.hf_repo or '-'}  {entry.modality or '-'}  "
            f"{entry.params or '-'}  {entry.display_license}"
        )
    return 0


# ------------------------------------------------------------------- adapt (LoRA)


def _adapt_budget_lines(args: argparse.Namespace, train: adapt_dataset.DatasetReport) -> list[str]:
    """Budget lines for ``adapt validate``: the pure estimate, plus a dry-run probe."""
    lines: list[str] = []
    if train.items < 1:
        lines.append("optimizer budget: no usable items — nothing to train on")
        return lines
    n_train, n_calib = adapt_budget.split_counts(train.items)
    lines.append(f"{adapt_budget.budget_message(n_train)}  [estimate: {n_train} train / {n_calib} calibration]")
    if not args.base:
        return lines
    entry = registry.get_entry(args.base)
    spec = loaders.ModelSpec(
        model_id=entry.id,
        hf_repo=entry.hf_repo,
        subfolder=entry.subfolder,
        revision=entry.revision or "main",
        models_dir=str(cache.models_dir()),
    )
    base_dir = spec.local_path
    if base_dir is None:
        lines.append(f"dry-run skipped: base {entry.id!r} is not in the shared model cache")
        return lines
    if not entry.package_source:
        lines.append(f"dry-run skipped: base {entry.id!r} has no package_source to train under")
        return lines
    env = training_env(entry.package_source)
    if not env.python.is_file():
        lines.append("dry-run skipped: training env not built yet (adapt train builds it)")
        return lines
    summary = adapt_budget.probe_dry_run(
        env.python, args.data, base_dir, eval_data=args.eval
    )
    if summary is None:
        lines.append("dry-run skipped: the training env probe failed (is laya installed there?)")
        return lines
    lines.append(
        "dry-run (training env): "
        f"{summary.get('valid_items', 0)} valid item(s), "
        f"{summary.get('train_items', 0)} train / {summary.get('calibration_items', 0)} calibration, "
        f"{summary.get('optimizer_updates', 0)} optimizer updates"
    )
    return lines


def _print_dataset_report(report: adapt_dataset.DatasetReport) -> None:
    print(report.source)
    print(f"  rows read            {report.rows}")
    print(f"  items                {report.items}")
    if report.skipped:
        detail = ", ".join(f"{reason} {count}" for reason, count in sorted(report.skipped.items()))
        print(f"  skipped              {sum(report.skipped.values())} ({detail})")
    else:
        print("  skipped              0")
    if report.questions:
        print("  questions")
        for stats in report.questions.values():
            labels = " · ".join(
                f"{label} {count}" for label, count in stats.labels.most_common()
            ) or "(none)"
            print(f"    {stats.qid}  {stats.type}  {stats.items} items  labels: {labels}")
    if report.warnings:
        print("  warnings")
        for warning in report.warnings:
            print(f"    {warning}")


def _cmd_adapt_validate(args: argparse.Namespace) -> int:
    """Validate and summarize adaptation datasets (pure dry run, no model stack)."""
    read_kwargs = {
        "text_column": args.text_column,
        "label_column": args.label_column,
        "question_id": args.question_id,
        "instructions": args.instructions,
    }
    train = adapt_dataset.validate_rows(
        adapt_dataset.read_rows(args.data, **read_kwargs), source=str(args.data)
    )
    eval_report = None
    leaked = 0
    if args.eval:
        eval_report = adapt_dataset.validate_rows(
            adapt_dataset.read_rows(args.eval, **read_kwargs), source=str(args.eval)
        )
        leaked = adapt_dataset.leaked_items(eval_report, train)
    budget_lines = _adapt_budget_lines(args, train)
    if args.json:
        print(
            json.dumps(
                {
                    "train": train.to_dict(),
                    "eval": eval_report.to_dict() if eval_report else None,
                    "leaked_eval_items": leaked,
                    "budget": budget_lines,
                },
                indent=2,
            )
        )
    else:
        _print_dataset_report(train)
        if eval_report is not None:
            _print_dataset_report(eval_report)
            print(f"  leakage              {leaked} eval item(s) share a (state, question) signature with train")
        for line in budget_lines:
            print(line)
    if train.items == 0:
        print("decision-swarm: error: the training data produced no usable items", file=sys.stderr)
        return 1
    return 0


def _adapt_train_config(entry: registry.ModelEntry, base_dir: Path, args: argparse.Namespace) -> dict:
    """The run configuration handed to the training env's driver."""
    provenance: dict[str, str] = {}
    for item in args.provenance or ():
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise adapt_error(f"--provenance needs KEY=VALUE, got {item!r}")
        provenance[key] = value
    return {
        "base": {
            "registry_id": entry.id,
            "hf_repo": entry.hf_repo,
            "subfolder": entry.subfolder,
            "revision": entry.revision,
        },
        "base_dir": str(base_dir),
        "data": args.data,
        "eval": args.eval,
        "out": args.out,
        "mode": args.mode,
        "epochs": args.epochs,
        "micro_batch": args.micro_batch,
        "grad_accum": args.grad_accum,
        "seed": args.seed,
        "loss": args.loss,
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "device": args.device,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "question_id": args.question_id,
        "instructions": args.instructions,
        "metric_map": args.metric_map,
        "provenance": provenance,
    }


def _adapt_prepare_env(entry: registry.ModelEntry, envs_root: str | None = None) -> envs.CellEnv:
    """The training env for a base: its package source plus the training extras."""
    env = training_env(entry.package_source or "", envs_root=envs_root)
    envs.ensure_venv(env)
    envs.install_packages(env, [entry.package_source or "", *TRAINING_PACKAGES])
    return env


def _run_adapt_module(python: Path, module: str, config: dict) -> int:
    """Run an adaptation driver under the training env's interpreter."""
    import subprocess

    from decision_swarm.runtime.orchestrator import _child_env

    result = subprocess.run(
        [str(python), "-m", module, "--config", json.dumps(config)],
        env=_child_env(),
    )
    return int(result.returncode)


def _run_adapt_training(python: Path, config: dict) -> int:
    return _run_adapt_module(python, "decision_swarm.adapt.train", config)


def _cmd_adapt_train(args: argparse.Namespace) -> int:
    """Train a LoRA adapter (or head/full fine-tune) for a registry base model."""
    if getattr(sys, "frozen", False):
        raise adapt_error("'adapt train' runs from a Python install, not the frozen launcher")
    entry = registry.get_entry(args.base)
    if entry.custom or not entry.hf_repo:
        raise adapt_error("--base must be a registry base model (the custom placeholder cannot be trained)")
    if entry.loader == "load_vlm":
        raise adapt_error("'adapt' covers the text laya family only; vision checkpoints have no training recipe")
    if not entry.package_source:
        raise adapt_error(f"base {entry.id!r} has no package_source to train under")
    cache.fetch(entry)  # idempotent: downloads the base checkpoint on a cold cache
    spec = loaders.ModelSpec(
        model_id=entry.id,
        hf_repo=entry.hf_repo,
        subfolder=entry.subfolder,
        revision=entry.revision or "main",
        models_dir=str(cache.models_dir()),
    )
    base_dir = spec.local_path
    if base_dir is None:
        raise adapt_error(f"base {entry.id!r} is missing from the cache after fetch")
    config = _adapt_train_config(entry, base_dir, args)
    env = _adapt_prepare_env(entry, args.envs_dir)
    code = _run_adapt_training(env.python, config)
    if code != 0:
        print(f"decision-swarm: error: training exited {code}", file=sys.stderr)
    return code


def _cmd_adapt_evaluate(args: argparse.Namespace) -> int:
    """Run the 'better at X' gate on a trained (merged) checkpoint."""
    if getattr(sys, "frozen", False):
        raise adapt_error("'adapt evaluate' runs from a Python install, not the frozen launcher")
    entry = registry.get_entry(args.base)
    if entry.custom or not entry.hf_repo:
        raise adapt_error("--base must be a registry base model")
    baseline_dir = args.base_dir
    if baseline_dir is None and not args.baseline:
        spec = loaders.ModelSpec(
            model_id=entry.id,
            hf_repo=entry.hf_repo,
            subfolder=entry.subfolder,
            revision=entry.revision or "main",
            models_dir=str(cache.models_dir()),
        )
        baseline_dir = str(spec.local_path) if spec.local_path else None
    if baseline_dir is None and not args.baseline:
        raise adapt_error(
            "no baseline: pass --baseline FILE or --base-dir DIR (the base is not in the shared cache)"
        )
    guard = args.guard or adapt_evaluate.default_guard_path()
    if not Path(guard).is_file():
        raise adapt_error(f"guard dataset not found: {guard}")
    merged = Path(args.merged)
    out = Path(args.out) if args.out else merged.parent / "evaluate_report.json"
    config = {
        "merged": str(merged),
        "data": args.data,
        "slice": args.slice,
        "baseline_dir": baseline_dir,
        "baseline_report": args.baseline,
        "guard": str(guard),
        "min_delta": args.min_delta,
        "max_regression": args.max_regression,
        "metric_map": args.metric_map,
        "device": args.device,
        "out": str(out),
        "text_column": args.text_column,
        "label_column": args.label_column,
        "question_id": args.question_id,
        "instructions": args.instructions,
    }
    env = _adapt_prepare_env(entry, args.envs_dir)
    code = _run_adapt_module(env.python, "decision_swarm.adapt.evaluate", config)
    if args.json and out.is_file():
        print(out.read_text(encoding="utf-8"))
    return code


def _cmd_adapt_merge(args: argparse.Namespace) -> int:
    """Merge an adapter into its base checkpoint (what install materializes)."""
    if getattr(sys, "frozen", False):
        raise adapt_error("'adapt merge' runs from a Python install, not the frozen launcher")
    entry = registry.get_entry(args.base)
    if entry.custom or not entry.hf_repo:
        raise adapt_error("--base must be a registry base model")
    adapter_dir = Path(args.adapter)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise adapt_error(f"adapter_config.json not found in {adapter_dir}")
    base_dir = args.base_dir
    if base_dir is None:
        spec = loaders.ModelSpec(
            model_id=entry.id,
            hf_repo=entry.hf_repo,
            subfolder=entry.subfolder,
            revision=entry.revision or "main",
            models_dir=str(cache.models_dir()),
        )
        base_dir = str(spec.local_path) if spec.local_path else None
    if base_dir is None:
        raise adapt_error(f"base {entry.id!r} is not in the shared cache; run 'decision-swarm fetch'")
    out = Path(args.out) if args.out else adapter_dir.parent / "merged"
    config = {"base_dir": str(base_dir), "adapter_dir": str(adapter_dir), "out_dir": str(out)}
    env = _adapt_prepare_env(entry, args.envs_dir)
    code = _run_adapt_module(env.python, "decision_swarm.adapt.merge", config)
    if code == 0:
        print(f"merged  {out}  (serve it with any loader that reads laya checkpoints)")
    return code


def _cmd_adapt_register(args: argparse.Namespace) -> int:
    """Register a trained adapter; refuses unless the evaluate gate passed."""
    try:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise adapt_error(f"{args.report}: cannot read the evaluate report ({exc})") from exc
    registry_path = Path(args.registry) if args.registry else registry.default_registry_path()
    entry, snippet = adapt_register.register_adapter(
        args.adapter, report=report, registry_path=registry_path, name=args.name
    )
    if args.patch:
        if not args.cell:
            raise adapt_error("--patch needs --cell CELL_ID")
        adapt_register.patch_manifest(args.patch, args.cell, entry)
    if args.json:
        print(json.dumps({"entry": entry, "snippet": snippet}, indent=2))
        return 0
    ref = f"{entry['merged']['hf_repo']}@{entry['merged']['revision']}"
    print(f"registered  {entry['id']}  (kind: adapter, base: {entry['base']})")
    print(f"adapter ref  {ref}")
    print("manifest cell snippet (copy-paste):")
    print(snippet)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="decision-swarm",
        description=(
            "Assemble a swarm of Laya decision-model sensors over one observation "
            "and package it as a runnable artifact."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="show cells, models, sizes, and cache status for a pack"
    )
    _add_pack_argument(inspect_parser)
    _add_models_dir_argument(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true", help="print the pack description as JSON")
    inspect_parser.set_defaults(handler=_cmd_inspect)

    fetch_parser = subparsers.add_parser(
        "fetch", help="pre-download a pack's (or manifest's) models to the shared cache"
    )
    fetch_source = fetch_parser.add_mutually_exclusive_group(required=True)
    fetch_source.add_argument("--pack", metavar="DIR", help="path to the swarm pack directory")
    fetch_source.add_argument(
        "--manifest", metavar="FILE", help="path to a swarm.json (fetch without a pack)"
    )
    _add_models_dir_argument(fetch_parser)
    fetch_parser.set_defaults(handler=_cmd_fetch)

    install_parser = subparsers.add_parser(
        "install", help="build per-package virtual environments for a pack"
    )
    _add_pack_argument(install_parser)
    _add_models_dir_argument(install_parser)
    _add_envs_dir_argument(install_parser)
    install_parser.set_defaults(handler=_cmd_install)

    run_parser = subparsers.add_parser("run", help="run a pack against an observation")
    _add_pack_argument(run_parser)
    _add_models_dir_argument(run_parser)
    _add_envs_dir_argument(run_parser)
    run_parser.add_argument("--input", required=True, metavar="FILE", help="observation payload (JSON)")
    run_parser.add_argument(
        "--offline", action="store_true", help="never touch the network; use the shared cache only"
    )
    output_group = run_parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", action="store_true", help="print results as JSON (default)")
    output_group.add_argument("--pretty", action="store_true", help="print results in human-readable form")
    run_parser.add_argument(
        "--detail",
        choices=("full", "compact"),
        default="full",
        help="compact: cell -> question -> {value, confidence} (default: full)",
    )
    run_parser.set_defaults(handler=_cmd_run)

    serve_parser = subparsers.add_parser(
        "serve",
        help="serve a pack as a long-lived decision service (JSONL over stdio)",
    )
    _add_pack_argument(serve_parser)
    _add_models_dir_argument(serve_parser)
    _add_envs_dir_argument(serve_parser)
    serve_parser.add_argument(
        "--offline", action="store_true", help="never touch the network; use the shared cache only"
    )
    serve_parser.set_defaults(handler=_cmd_serve)

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="serve packs as MCP tools over stdio (for AI agents)",
    )
    mcp_parser.add_argument(
        "--packs-dir",
        action="append",
        metavar="DIR",
        help="directory containing packs (repeatable); packs are discovered one level down",
    )
    mcp_parser.add_argument(
        "--pack", metavar="DIR", help="default pack to judge when a tool omits 'pack'"
    )
    _add_models_dir_argument(mcp_parser)
    _add_envs_dir_argument(mcp_parser)
    mcp_parser.add_argument(
        "--allow-unvetted",
        action="store_true",
        help="allow models and package sources outside the registry when building or installing",
    )
    mcp_parser.add_argument(
        "--max-warm",
        type=int,
        default=1,
        metavar="N",
        help="how many swarms to keep warm (default: 1)",
    )
    mcp_parser.add_argument(
        "--print-config",
        action="store_true",
        help="print the MCP client config for --pack and exit (use after moving a pack)",
    )
    mcp_parser.set_defaults(handler=_cmd_mcp)

    build_parser = subparsers.add_parser("build", help="build a swarm pack from a manifest")
    build_parser.add_argument("--manifest", required=True, metavar="FILE", help="path to swarm.json")
    build_parser.add_argument("--dest", required=True, metavar="DIR", help="pack output directory")
    _add_models_dir_argument(build_parser)
    _add_envs_dir_argument(build_parser)
    build_parser.add_argument(
        "--no-pin",
        action="store_true",
        help="do not resolve model revisions to commits (skips the Hub lookup)",
    )
    build_parser.add_argument(
        "--export",
        dest="export",
        action="store_true",
        default=True,
        help="also bundle the pack with its weights into <dest>.swarmpack.tar (default)",
    )
    build_parser.add_argument(
        "--no-export",
        dest="export",
        action="store_false",
        help="write only the pack directory, without the transferable bundle",
    )
    build_parser.add_argument(
        "--no-envs",
        action="store_true",
        help="bundle weights only; the recipient builds the cell envs with 'install'",
    )
    build_parser.add_argument("--json", action="store_true", help="print a machine-readable summary")
    build_parser.set_defaults(handler=_cmd_build)

    export_parser = subparsers.add_parser(
        "export", help="bundle a pack and its weights into one transferable tar"
    )
    _add_pack_argument(export_parser)
    _add_models_dir_argument(export_parser)
    export_parser.add_argument(
        "--dest",
        metavar="FILE",
        help="bundle path (default: <pack>.swarmpack.tar beside the pack directory)",
    )
    export_parser.add_argument(
        "--no-envs",
        action="store_true",
        help="bundle weights only; the recipient builds the cell envs with 'install'",
    )
    export_parser.set_defaults(handler=_cmd_export)

    models_parser = subparsers.add_parser("models", help="list known models from the registry")
    models_parser.add_argument("--json", action="store_true", help="print the registry as JSON")
    models_parser.set_defaults(handler=_cmd_models)

    adapt_parser = subparsers.add_parser(
        "adapt", help="train and manage LoRA adapters for decision models"
    )
    adapt_sub = adapt_parser.add_subparsers(dest="adapt_command", metavar="<subcommand>", required=True)

    adapt_validate = adapt_sub.add_parser(
        "validate", help="validate and summarize adaptation datasets (dry run)"
    )
    adapt_validate.add_argument("--data", required=True, metavar="FILE", help="training rows (JSONL or CSV)")
    adapt_validate.add_argument("--eval", metavar="FILE", help="evaluation rows (JSONL or CSV)")
    adapt_validate.add_argument("--text-column", default="text", metavar="NAME", help="CSV text column (default: text)")
    adapt_validate.add_argument("--label-column", default="label", metavar="NAME", help="CSV label column (default: label)")
    adapt_validate.add_argument("--question-id", default="label", metavar="NAME", help="CSV question id (default: label)")
    adapt_validate.add_argument("--instructions", metavar="TEXT", help="CSV question wording")
    adapt_validate.add_argument(
        "--base",
        metavar="ID",
        help="registry base model: enables the laya.train dry-run budget when its training env is built",
    )
    _add_envs_dir_argument(adapt_validate)
    adapt_validate.add_argument("--json", action="store_true", help="print the report as JSON")
    adapt_validate.set_defaults(handler=_cmd_adapt_validate)

    adapt_train = adapt_sub.add_parser(
        "train", help="train a LoRA adapter (or head/full fine-tune) for a registry base model"
    )
    adapt_train.add_argument("--base", required=True, metavar="ID", help="registry base model id")
    adapt_train.add_argument("--data", required=True, metavar="FILE", help="training rows (JSONL or CSV)")
    adapt_train.add_argument("--eval", metavar="FILE", help="evaluation rows (JSONL or CSV)")
    adapt_train.add_argument("--out", required=True, metavar="DIR", help="artifact output directory")
    adapt_train.add_argument(
        "--mode", choices=("lora", "head-only", "full"), default="lora", help="what to train (default: lora)"
    )
    adapt_train.add_argument("--epochs", type=int, default=4, metavar="N")
    adapt_train.add_argument("--micro-batch", type=int, default=None, metavar="N", help="default: sized to detected VRAM")
    adapt_train.add_argument("--grad-accum", type=int, default=8, metavar="N")
    adapt_train.add_argument("--seed", type=int, default=0, metavar="N")
    adapt_train.add_argument("--loss", choices=("rlcd", "soft-ce"), default="rlcd")
    adapt_train.add_argument("--rank", type=int, default=8, metavar="N", help="LoRA rank")
    adapt_train.add_argument("--lora-alpha", type=int, default=16, metavar="N")
    adapt_train.add_argument("--lora-dropout", type=float, default=0.05, metavar="P")
    adapt_train.add_argument("--device", default="auto", metavar="DEV", help="auto | cuda | mps | cpu")
    adapt_train.add_argument(
        "--metric-map", metavar="TEXT", help="the X -> metric mapping this specialization targets (recorded)"
    )
    adapt_train.add_argument(
        "--provenance", action="append", metavar="KEY=VALUE", help="dataset provenance (generator, prompt, seed, ...)"
    )
    adapt_train.add_argument("--text-column", default="text", metavar="NAME")
    adapt_train.add_argument("--label-column", default="label", metavar="NAME")
    adapt_train.add_argument("--question-id", default="label", metavar="NAME")
    adapt_train.add_argument("--instructions", metavar="TEXT")
    _add_envs_dir_argument(adapt_train)
    adapt_train.set_defaults(handler=_cmd_adapt_train)

    adapt_evaluate_parser = adapt_sub.add_parser(
        "evaluate", help="gate a trained checkpoint: 'better at X', no regressions"
    )
    adapt_evaluate_parser.add_argument("--base", required=True, metavar="ID", help="registry base model id")
    adapt_evaluate_parser.add_argument("--merged", required=True, metavar="DIR", help="merged checkpoint directory")
    adapt_evaluate_parser.add_argument("--data", required=True, metavar="FILE", help="evaluation rows (JSONL or CSV), tagged for slices")
    adapt_evaluate_parser.add_argument("--slice", default="all", metavar="NAME", help="the X slice to improve (default: all)")
    adapt_evaluate_parser.add_argument(
        "--baseline", metavar="FILE", help="baseline report (evaluate/train report with slice metrics)"
    )
    adapt_evaluate_parser.add_argument(
        "--base-dir", metavar="DIR", help="baseline checkpoint directory (default: the base's cached copy)"
    )
    adapt_evaluate_parser.add_argument("--guard", metavar="FILE", help="must-not-regress set (default: the shipped one)")
    adapt_evaluate_parser.add_argument("--min-delta", type=float, default=adapt_evaluate.DEFAULT_MIN_DELTA, metavar="D")
    adapt_evaluate_parser.add_argument(
        "--max-regression", type=float, default=adapt_evaluate.DEFAULT_MAX_REGRESSION, metavar="D"
    )
    adapt_evaluate_parser.add_argument("--metric-map", metavar="TEXT", help="the X -> metric mapping (recorded)")
    adapt_evaluate_parser.add_argument("--out", metavar="FILE", help="report path (default: <merged>/../evaluate_report.json)")
    adapt_evaluate_parser.add_argument("--device", default="auto", metavar="DEV")
    adapt_evaluate_parser.add_argument("--text-column", default="text", metavar="NAME")
    adapt_evaluate_parser.add_argument("--label-column", default="label", metavar="NAME")
    adapt_evaluate_parser.add_argument("--question-id", default="label", metavar="NAME")
    adapt_evaluate_parser.add_argument("--instructions", metavar="TEXT")
    _add_envs_dir_argument(adapt_evaluate_parser)
    adapt_evaluate_parser.add_argument("--json", action="store_true", help="print the report as JSON")
    adapt_evaluate_parser.set_defaults(handler=_cmd_adapt_evaluate)

    adapt_merge_parser = adapt_sub.add_parser(
        "merge", help="merge an adapter into its base checkpoint"
    )
    adapt_merge_parser.add_argument("--base", required=True, metavar="ID", help="registry base model id")
    adapt_merge_parser.add_argument("--adapter", required=True, metavar="DIR", help="adapter directory")
    adapt_merge_parser.add_argument("--base-dir", metavar="DIR", help="base checkpoint (default: the base's cached copy)")
    adapt_merge_parser.add_argument("--out", metavar="DIR", help="output checkpoint (default: <adapter>/../merged)")
    _add_envs_dir_argument(adapt_merge_parser)
    adapt_merge_parser.set_defaults(handler=_cmd_adapt_merge)

    adapt_register_parser = adapt_sub.add_parser(
        "register", help="register a trained adapter (refuses on a failed evaluate gate)"
    )
    adapt_register_parser.add_argument("--adapter", required=True, metavar="DIR", help="adapter directory")
    adapt_register_parser.add_argument(
        "--report", required=True, metavar="FILE", help="the evaluate report (must have passed)"
    )
    adapt_register_parser.add_argument("--name", metavar="SLUG", help="name for the registry id")
    adapt_register_parser.add_argument("--registry", metavar="FILE", help="registry file to write")
    adapt_register_parser.add_argument(
        "--patch", metavar="MANIFEST", help="opt-in: swap a cell in this manifest onto the adapter"
    )
    adapt_register_parser.add_argument("--cell", metavar="ID", help="cell id for --patch")
    adapt_register_parser.add_argument("--json", action="store_true", help="print the entry as JSON")
    adapt_register_parser.set_defaults(handler=_cmd_adapt_register)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == worker.WORKER_ROUTE:
        from decision_swarm.runtime.worker import main as worker_main

        return worker_main(args_list[1:])
    parser = build_parser()
    args = parser.parse_args(args_list)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except (
        manifest.ManifestError,
        query.QueryError,
        registry.RegistryError,
        cache.CacheError,
        envs.EnvError,
        loaders.LoaderError,
        orchestrator.OrchestratorError,
        adapt_dataset.DatasetError,
        AdaptError,
    ) as exc:
        print(f"decision-swarm: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
