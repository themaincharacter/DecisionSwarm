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
        if not isinstance(message, dict):
            _write(
                stdout,
                {"type": "error", "id": None, "message": "message must be a JSON object"},
            )
            continue

        kind = message.get("type")
        if kind in CONTROL_TYPES:
            return 0

        raw_id = message.get("id")
        message_id = raw_id if isinstance(raw_id, int) else None
        if raw_id is not None and message_id is None:
            _write(
                stdout,
                {"type": "error", "id": None, "message": "'id' must be an integer"},
            )
            continue
        if kind != "query":
            _write(
                stdout,
                {"type": "error", "id": message_id, "message": f"unknown message type {kind!r}"},
            )
            continue

        try:
            data = {key: value for key, value in message.items() if key not in ENVELOPE_FIELDS}
            result = service.query(parse_query(data, manifest))
        except Exception as exc:  # noqa: BLE001 - one bad frame never ends the session
            _write(stdout, {"type": "error", "id": message_id, "message": str(exc)})
            continue
        _write(stdout, {"type": "result", "id": message_id, "cells": result.to_dict()["cells"]})
    return 0
