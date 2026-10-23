"""``decision-swarm`` command-line entrypoint.

The command surface is defined here (``inspect``, ``fetch``, ``install``,
``run``); the handlers land in Phases 1-3. ``decision-swarm --help`` works
today.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence

from decision_swarm import __version__, cache, envs, manifest, pack, query, registry
from decision_swarm.runtime import loaders, orchestrator


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
    models = pack.collect_models(swarm, models_root=args.models_dir, registry_entries=entries)
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
    print(f"cache: {cache.models_dir(args.models_dir)}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    """Download a pack's models into the shared cache (idempotent)."""
    swarm = manifest.load_manifest(args.pack)
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


def _cmd_run(args: argparse.Namespace) -> int:
    """Run a pack against one observation."""
    swarm = manifest.load_manifest(args.pack)
    payload = query.load_query(args.input, swarm)
    entries = registry.load_registry()
    if args.offline:
        missing = [
            model.id
            for model in pack.collect_models(swarm, models_root=args.models_dir, registry_entries=entries)
            if not model.cached
        ]
        if missing:
            raise cache.CacheError(f"--offline: model(s) not in the cache: {', '.join(missing)}")
    result = orchestrator.run(swarm, payload, registry_entries=entries, envs_root=args.envs_dir)
    if args.pretty:
        _print_pretty(result)
    else:
        print(json.dumps(result.to_dict(), indent=2))
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
        "fetch", help="pre-download a pack's models to the shared cache"
    )
    _add_pack_argument(fetch_parser)
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
