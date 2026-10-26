"""Long-lived decision service: JSONL over stdio.

The serve loop turns a warm :class:`~decision_swarm.runtime.orchestrator.Swarm`
into a service a real-time consumer can query at model latency. One JSON object
per line, in the same framing as :mod:`decision_swarm.runtime.protocol`:

- warm-up:  ``{"type": "ready", "name": "...", "cells": [...], "views": [...],
  "warnings": [...]}`` (emitted once every model is loaded)
- request:  ``{"type": "query", "id": 1, "views": {...}, "questions": {...}}``
  (exactly the ``run --input`` payload plus ``type`` and ``id``)
- response: ``{"type": "result", "id": 1, "cells": {...}}`` (the ``cells``
  object of the run result schema)
- failure:  ``{"type": "error", "id": 1, "message": "..."}`` (``id`` is null
  for session-level failures; a failed query never ends the session)
- control:  ``{"type": "shutdown"}`` (end-of-input does the same)

``id`` is echoed back for correlation and may be omitted. Episode logging is
consumer-side: tee this stream for replay.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any, Protocol, TextIO

from decision_swarm.manifest import SwarmManifest
from decision_swarm.query import QueryPayload, SwarmResult, parse_query

ENVELOPE_FIELDS = frozenset({"type", "id"})
CONTROL_TYPES = frozenset({"shutdown"})


class DecisionService(Protocol):
    """What the serve loop needs from a warm swarm."""

    def query(self, payload: QueryPayload) -> SwarmResult: ...

    def startup_warnings(self) -> tuple[str, ...]: ...


def _write(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload) + "\n")
    stream.flush()


def _envelope_id(message: Any) -> int | None:
    """The correlation id, or ``None`` when it is absent or not an integer."""
    if not isinstance(message, Mapping):
        return None
    raw_id = message.get("id")
    return raw_id if isinstance(raw_id, int) else None


def run_query(
    service: DecisionService,
    manifest: SwarmManifest,
    message: Any,
) -> tuple[int | None, SwarmResult]:
    """Validate one query envelope and run it through the swarm.

    Transport-neutral: the stdio adapter and the agent tools both call this, so
    envelope rules, ``parse_query`` validation and error text cannot drift
    between transports. Raises on any envelope or query problem.
    """
    if not isinstance(message, Mapping):
        raise ValueError("message must be a JSON object")
    raw_id = message.get("id")
    message_id = raw_id if isinstance(raw_id, int) else None
    if raw_id is not None and message_id is None:
        raise ValueError("'id' must be an integer")
    kind = message.get("type")
    if kind != "query":
        raise ValueError(f"unknown message type {kind!r}")
    data = {key: value for key, value in message.items() if key not in ENVELOPE_FIELDS}
    return message_id, service.query(parse_query(data, manifest))


def answer(
    service: DecisionService,
    manifest: SwarmManifest,
    message: Any,
) -> dict[str, Any]:
    """One decoded envelope -> one response dict (``result`` or ``error``).

    Control messages are transport concerns (the stdio adapter returns on
    ``shutdown``), so they fall through to an ``error`` here.
    """
    try:
        message_id, result = run_query(service, manifest, message)
    except Exception as exc:  # noqa: BLE001 - one bad frame never ends the session
        return {"type": "error", "id": _envelope_id(message), "message": str(exc)}
    return {"type": "result", "id": message_id, "cells": result.to_dict()["cells"]}


def serve_loop(
    service: DecisionService,
    manifest: SwarmManifest,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Answer query lines until shutdown or end-of-stream; always exit cleanly."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout

    _write(
        stdout,
        {
            "type": "ready",
            "name": manifest.name,
            "cells": [cell.id for cell in manifest.cells],
            "views": [view.name for view in manifest.views],
            "warnings": list(service.startup_warnings()),
        },
    )

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write(stdout, {"type": "error", "id": None, "message": f"invalid JSON line: {exc}"})
            continue
        if isinstance(message, Mapping) and message.get("type") in CONTROL_TYPES:
            return 0
        _write(stdout, answer(service, manifest, message))
    return 0
