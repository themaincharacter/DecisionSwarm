"""``decision-swarm`` command-line entrypoint.

The command surface is defined here (``inspect``, ``fetch``, ``install``,
``run``); the handlers land in Phases 1-3. ``decision-swarm --help`` works
today.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from decision_swarm import __version__

_UNAVAILABLE_EXIT_CODE = 2
"""Exit code for commands whose handlers are not implemented yet."""


def _unavailable(command: str, phase: str) -> Callable[[argparse.Namespace], int]:
    """Build a handler stub for a command that lands in a later phase."""

    def handler(args: argparse.Namespace) -> int:
        print(
            f"decision-swarm: '{command}' is not implemented yet (planned for {phase})",
            file=sys.stderr,
        )
        return _UNAVAILABLE_EXIT_CODE

    return handler


def _add_pack_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pack", required=True, metavar="DIR", help="path to the swarm pack directory")


def _add_models_dir_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models-dir",
        metavar="DIR",
        help="model cache location (default: ~/.decision-swarm/models, or $DECISION_SWARM_MODELS)",
    )


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
    inspect_parser.set_defaults(handler=_unavailable("inspect", "Phase 1"))

    fetch_parser = subparsers.add_parser(
        "fetch", help="pre-download a pack's models to the shared cache"
    )
    _add_pack_argument(fetch_parser)
    _add_models_dir_argument(fetch_parser)
    fetch_parser.set_defaults(handler=_unavailable("fetch", "Phase 1"))

    install_parser = subparsers.add_parser(
        "install", help="build per-package virtual environments for a pack"
    )
    _add_pack_argument(install_parser)
    _add_models_dir_argument(install_parser)
    install_parser.set_defaults(handler=_unavailable("install", "Phase 2"))

    run_parser = subparsers.add_parser("run", help="run a pack against an observation")
    _add_pack_argument(run_parser)
    _add_models_dir_argument(run_parser)
    run_parser.add_argument("--input", required=True, metavar="FILE", help="observation payload (JSON)")
    run_parser.add_argument(
        "--offline", action="store_true", help="never touch the network; use the shared cache only"
    )
    output_group = run_parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", action="store_true", help="print results as JSON (default)")
    output_group.add_argument("--pretty", action="store_true", help="print results in human-readable form")
    run_parser.set_defaults(handler=_unavailable("run", "Phase 3"))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
