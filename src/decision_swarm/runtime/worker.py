"""Per-cell worker process.

Each worker loads its model once and stays warm across queries, receives only
the views its cell is bound to plus the resolved questions (query overrides win
over pack defaults), and answers them with typed ``choice`` / ``score`` /
``noul`` answers. It runs inside the env built for its cell's
``package_source`` and speaks the JSON-lines stdio protocol.

Launch: ``python -m decision_swarm.runtime.worker --config '<json>'`` where the
config names the loader, the model spec, and the cell's bound view types. Once
the model is loaded the worker announces ``ready`` on stdout, then answers
queries until shutdown or end-of-stream.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from decision_swarm.runtime import loaders, protocol

WORKER_ROUTE = "_worker"
"""Hidden CLI route: a frozen launcher spawns itself as a worker this way."""


@dataclass(frozen=True)
class WorkerConfig:
    """Everything a worker needs at startup (arrives as one JSON argument)."""

    loader: str
    model: loaders.ModelSpec = field(default_factory=lambda: loaders.ModelSpec(model_id=""))
    view_types: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, text: str) -> "WorkerConfig":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise loaders.LoaderError(f"worker config is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise loaders.LoaderError("worker config must be a JSON object")
        loader = data.get("loader")
        if not isinstance(loader, str) or not loader:
            raise loaders.LoaderError("worker config needs a 'loader' name")
        raw_model = data.get("model", {})
        if not isinstance(raw_model, dict):
            raise loaders.LoaderError("worker config 'model' must be an object")
        view_types = data.get("view_types", {})
        if not isinstance(view_types, dict):
            raise loaders.LoaderError("worker config 'view_types' must be an object")
        spec = loaders.ModelSpec(
            model_id=str(raw_model.get("model_id", "")),
            hf_repo=raw_model.get("hf_repo"),
            subfolder=raw_model.get("subfolder"),
            revision=str(raw_model.get("revision", "main")),
            models_dir=raw_model.get("models_dir"),
        )
        return cls(loader=loader, model=spec, view_types={str(k): str(v) for k, v in view_types.items()})


def run_session(
    predictor: loaders.Predictor, stdin: TextIO | None = None, stdout: TextIO | None = None
) -> int:
    """Answer queries from stdin until shutdown or end-of-stream."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    while True:
        try:
            message = protocol.read(stdin)
        except protocol.ProtocolError as exc:
            protocol.write(stdout, protocol.ErrorMessage(message=str(exc)))
            continue
        if message is None or isinstance(message, protocol.ShutdownMessage):
            return 0
        if not isinstance(message, protocol.QueryMessage):
            protocol.write(
                stdout,
                protocol.ErrorMessage(
                    id=getattr(message, "id", None),
                    message=f"unexpected {type(message).__name__} in worker session",
                ),
            )
            continue
        try:
            started = time.perf_counter()
            answers, warnings = predictor.predict(message.views, message.questions)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001 - report to the orchestrator, stay alive
            protocol.write(stdout, protocol.ErrorMessage(id=message.id, message=str(exc)))
            continue
        protocol.write(
            stdout,
            protocol.ResultMessage(
                id=message.id,
                answers=dict(answers),
                latency_ms=latency_ms,
                warnings=tuple(warnings),
            ),
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decision-swarm-worker",
        description="Per-cell worker: load one model and answer queries over stdio.",
    )
    parser.add_argument("--config", required=True, metavar="JSON", help="worker config")
    args = parser.parse_args(argv)

    try:
        config = WorkerConfig.from_json(args.config)
        predictor = loaders.build_predictor(config.loader, config.model, config.view_types)
    except loaders.LoaderError as exc:
        protocol.write(sys.stdout, protocol.ErrorMessage(message=str(exc)))
        return 1
    protocol.write(sys.stdout, protocol.ReadyMessage())
    return run_session(predictor)


if __name__ == "__main__":
    sys.exit(main())
