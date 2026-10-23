"""Swarm orchestrator.

Spawns one worker process per cell (2-8, in parallel), fans out each query to
every worker, and collects the raw per-cell answers. Each worker receives ONLY
the views its cell is bound to, so input diversity is structural. Workers stay
warm across queries. v1 performs no aggregation or consensus — results are
returned per cell; the worker topology leaves room for neighbor links later.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decision_swarm import cache, envs, registry
from decision_swarm.manifest import CellSpec, QuestionSpec, SwarmManifest
from decision_swarm.query import (
    Answer,
    CellResult,
    QueryPayload,
    SwarmResult,
    resolve_questions,
)
from decision_swarm.runtime import loaders, protocol


class OrchestratorError(RuntimeError):
    """Raised when a swarm cannot be planned or started."""


@dataclass(frozen=True)
class CellPlan:
    """Everything needed to run one cell's worker."""

    cell: CellSpec
    model: registry.ModelEntry
    loader: str
    package_source: str | None
    python: Path
    view_types: Mapping[str, str]
    warnings: tuple[str, ...] = ()


def plan_cells(
    swarm: SwarmManifest,
    *,
    registry_entries: Mapping[str, registry.ModelEntry] | None = None,
    envs_root: str | Path | None = None,
    interpreter: str | Path | None = None,
) -> list[CellPlan]:
    """Resolve each cell to a runnable plan (model, loader, interpreter, view types).

    With ``interpreter`` set, every worker runs on it (tests/demos); otherwise
    each cell's env python is used when built, falling back to the current
    interpreter with a warning naming ``decision-swarm install``.
    """
    env_by_cell: dict[str, envs.CellEnv] = {}
    if interpreter is None:
        for env in envs.cell_envs(swarm, envs_root=envs_root, registry_entries=registry_entries):
            for cell_id in env.shared_by:
                env_by_cell[cell_id] = env

    known_views = {view.name: view.type for view in swarm.views}
    plans: list[CellPlan] = []
    for cell in swarm.cells:
        entry = registry.resolve_model(cell.model, registry_entries)
        package_source, loader = registry.resolve_runtime(cell, registry_entries)
        if not loader:
            raise OrchestratorError(f"cell {cell.id!r} has no loader; set cell.runtime or use a registry model")

        warnings: list[str] = []
        if interpreter is not None:
            python = Path(interpreter)
        else:
            env = env_by_cell.get(cell.id)
            if env is not None and env.python.exists():
                python = env.python
            else:
                python = Path(sys.executable)
                warnings.append(
                    f"cell {cell.id!r}: env is not built (run 'decision-swarm install'); using {python}"
                )
        plans.append(
            CellPlan(
                cell=cell,
                model=entry,
                loader=loader,
                package_source=package_source,
                python=python,
                view_types={name: known_views[name] for name in cell.views},
                warnings=tuple(warnings),
            )
        )
    return plans


def _worker_source_root() -> str:
    """Directory that makes ``decision_swarm`` importable for workers.

    Workers run on the cell env's interpreter, which has the model stack but
    not decision_swarm: this root goes on their PYTHONPATH. In a frozen
    executable the package only exists as bytecode, so the executable bundles
    the pure-Python sources under ``decision_swarm_src``.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return str(Path(meipass) / "decision_swarm_src")
    return str(Path(__file__).resolve().parents[2])


def _child_env() -> dict[str, str]:
    """Environment for workers: decision_swarm importable in any interpreter."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _worker_source_root() + (os.pathsep + existing if existing else "")
    return env


class WorkerHandle:
    """One warm worker subprocess speaking the stdio protocol."""

    def __init__(self, plan: CellPlan, process: subprocess.Popen[str]) -> None:
        self.plan = plan
        self._process = process

    @classmethod
    def spawn(cls, plan: CellPlan) -> "WorkerHandle":
        config = {
            "loader": plan.loader,
            "model": {
                "model_id": plan.model.id,
                "hf_repo": plan.model.hf_repo,
                "subfolder": plan.model.subfolder,
                "revision": plan.model.revision,
                "models_dir": str(cache.models_dir()),
            },
            "view_types": dict(plan.view_types),
        }
        process = subprocess.Popen(
            [str(plan.python), "-m", "decision_swarm.runtime.worker", "--config", json.dumps(config)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=_child_env(),
        )
        return cls(plan, process)

    def send(self, message: protocol.QueryMessage) -> protocol.ErrorMessage | None:
        """Write one query; return an error message when the worker cannot take it."""
        try:
            if self._process.stdin is None:
                raise OSError("worker stdin is closed")
            protocol.write(self._process.stdin, message)
        except (BrokenPipeError, OSError) as exc:
            return protocol.ErrorMessage(
                id=message.id, message=f"worker is not accepting queries ({exc})"
            )
        return None

    def receive(self) -> protocol.Message:
        """Read the worker's response (result or error)."""
        if self._process.stdout is None:  # pragma: no cover - defensive
            return protocol.ErrorMessage(message="worker stdout is closed")
        response = protocol.read(self._process.stdout)
        return response if response is not None else protocol.ErrorMessage(message="worker ended its session")

    def shutdown(self) -> None:
        try:
            if self._process.poll() is None and self._process.stdin is not None:
                protocol.write(self._process.stdin, protocol.ShutdownMessage())
                self._process.wait(timeout=10)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            self._process.kill()
            self._process.wait(timeout=10)
        finally:
            for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
                if stream is not None:
                    stream.close()


def _answers(questions: Mapping[str, QuestionSpec], wire: Mapping[str, Any]) -> dict[str, Answer]:
    """Turn wire answer dicts back into typed :class:`Answer` objects."""
    answers: dict[str, Answer] = {}
    for name, question in questions.items():
        raw = wire.get(name)
        raw = dict(raw) if isinstance(raw, Mapping) else {}
        value = raw.pop(question.type, None)
        confidence = float(raw.pop("confidence", 0.0))
        answers[name] = Answer(
            name=name, type=question.type, value=value, confidence=confidence, extra=raw
        )
    return answers


class Swarm:
    """Warm swarm: one worker process per cell, queried in parallel rounds."""

    def __init__(self, plans: Sequence[CellPlan]) -> None:
        self._plans = list(plans)
        self._workers: dict[str, WorkerHandle] = {}
        self._ids = itertools.count(1)

    def start(self) -> "Swarm":
        for plan in self._plans:
            self._workers[plan.cell.id] = WorkerHandle.spawn(plan)
        return self

    def query(self, payload: QueryPayload) -> SwarmResult:
        """Fan one query out to every worker and collect raw per-cell answers."""
        requests: dict[str, tuple[CellPlan, dict[str, QuestionSpec], protocol.QueryMessage]] = {}
        for plan in self._plans:
            cell = plan.cell
            # structural diversity: each worker sees only its bound views
            views = {name: payload.views[name] for name in cell.views}
            questions = {q.name: q for q in resolve_questions(cell, payload)}
            message = protocol.QueryMessage(id=next(self._ids), views=views, questions=questions)
            requests[cell.id] = (plan, questions, message)

        # write to all workers first: they compute in parallel, then we collect
        errors: dict[str, protocol.ErrorMessage] = {}
        for cell_id, (_plan, _questions, message) in requests.items():
            send_error = self._workers[cell_id].send(message)
            if send_error is not None:
                errors[cell_id] = send_error

        cells: dict[str, CellResult] = {}
        for cell_id, (plan, questions, message) in requests.items():
            if cell_id in errors:
                response: protocol.Message = errors[cell_id]
            else:
                response = self._workers[cell_id].receive()
            cells[cell_id] = self._result_for(plan, questions, response)
        return SwarmResult(cells=cells)

    @staticmethod
    def _result_for(
        plan: CellPlan, questions: Mapping[str, QuestionSpec], response: protocol.Message
    ) -> CellResult:
        setup_warnings = list(plan.warnings)
        if isinstance(response, protocol.ResultMessage):
            return CellResult(
                cell_id=plan.cell.id,
                model=plan.cell.model,
                answers=_answers(questions, response.answers),
                latency_ms=response.latency_ms,
                warnings=tuple(setup_warnings) + tuple(response.warnings),
            )
        message = response.message if isinstance(response, protocol.ErrorMessage) else "unexpected response"
        return CellResult(
            cell_id=plan.cell.id,
            model=plan.cell.model,
            answers={},
            latency_ms=0.0,
            warnings=tuple(setup_warnings) + (f"worker error: {message}",),
        )

    def close(self) -> None:
        for worker in self._workers.values():
            worker.shutdown()
        self._workers.clear()

    def __enter__(self) -> "Swarm":
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def run(
    swarm: SwarmManifest,
    payload: QueryPayload,
    *,
    registry_entries: Mapping[str, registry.ModelEntry] | None = None,
    envs_root: str | Path | None = None,
    interpreter: str | Path | None = None,
) -> SwarmResult:
    """Plan the cells, run one query across warm workers, and return the result schema."""
    plans = plan_cells(
        swarm, registry_entries=registry_entries, envs_root=envs_root, interpreter=interpreter
    )
    with Swarm(plans) as warm:
        return warm.query(payload)
