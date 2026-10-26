"""``decision-swarm`` command-line entrypoint.

The command surface is defined here: ``inspect``, ``fetch``, ``install``,
``run`` (one observation), ``serve`` (long-lived decision service over stdio),
``build``, and ``models``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from decision_swarm import __version__, cache, envs, manifest, pack, query, registry
from decision_swarm.runtime import loaders, orchestrator, serve, worker


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
    swarm = manifest.load_manifest(args.pack)
    entries = registry.load_registry()
    models_root = _models_root(args)
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


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Download a pack's models into the shared cache (idempotent)."""
    swarm = manifest.load_manifest(args.pack or args.manifest)
    entries = registry.load_registry()
    failures = 0
    for entry in pack.unique_entries(swarm, registry_entries=entries):
        checkpoint = entry.hf_repo + (f"/{entry.subfolder}" if entry.subfolder else "")
        try:
            result = cache.fetch(entry, root=args.models_dir)
        except (cache.CacheError, OSError) as exc:
            print(f"decision-swarm: error: {exc}", file=sys.stderr)
            failures += 1
            continue
        state = "downloaded" if result.downloaded else "already cached"
        print(f"{entry.id}  {checkpoint}  {state}  ({result.bytes} bytes)")
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
    ) as exc:
        print(f"decision-swarm: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
