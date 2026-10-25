"""JSON-lines stdio protocol between the orchestrator and cell workers.

One JSON object per line, newline-framed:

- request:  ``{"type": "query", "id": 1, "views": {...}, "questions": {...}}``
- response: ``{"type": "result", "id": 1, "answers": {...}, "latency_ms": 41.0,
  "warnings": [...]}``
- failure:  ``{"type": "error", "id": 1, "message": "..."}``
- warm-up:   ``{"type": "ready"}`` (worker has loaded its model)
- control:  ``{"type": "shutdown"}``

Only the views a cell is bound to travel in its request; answers on the wire
are plain dicts shaped like the result schema's ``answers`` sub-object.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from decision_swarm.manifest import (
    ManifestError,
    QuestionSpec,
    parse_question,
    question_to_dict,
)

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    """Raised when a message cannot be encoded or decoded."""


@dataclass(frozen=True)
class QueryMessage:
    """One query for a cell: its bound views and resolved questions."""

    id: int
    views: Mapping[str, str]
    questions: Mapping[str, QuestionSpec]


@dataclass(frozen=True)
class ResultMessage:
    """A cell's answers for one query."""

    id: int
    answers: Mapping[str, Any]
    latency_ms: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ErrorMessage:
    """A worker-side failure for one query (or the session when id is None)."""

    message: str
    id: int | None = None


@dataclass(frozen=True)
class ReadyMessage:
    """The worker has loaded its model and can answer queries."""


@dataclass(frozen=True)
class ShutdownMessage:
    """End the worker session."""


Message = QueryMessage | ResultMessage | ErrorMessage | ReadyMessage | ShutdownMessage


def encode(message: Message) -> str:
    """Serialize a message to one JSON line (newline included)."""
    if isinstance(message, QueryMessage):
        payload: dict[str, Any] = {
            "type": "query",
            "id": message.id,
            "views": dict(message.views),
            "questions": {name: question_to_dict(q) for name, q in message.questions.items()},
        }
    elif isinstance(message, ResultMessage):
        payload = {
            "type": "result",
            "id": message.id,
            "answers": dict(message.answers),
            "latency_ms": message.latency_ms,
            "warnings": list(message.warnings),
        }
    elif isinstance(message, ErrorMessage):
        payload = {"type": "error", "id": message.id, "message": message.message}
    elif isinstance(message, ReadyMessage):
        payload = {"type": "ready"}
    elif isinstance(message, ShutdownMessage):
        payload = {"type": "shutdown"}
    else:  # pragma: no cover - exhaustive union
        raise ProtocolError(f"cannot encode {type(message).__name__}")
    return json.dumps(payload) + "\n"


def _require_int_id(payload: Mapping[str, Any]) -> int:
    message_id = payload.get("id")
    if not isinstance(message_id, int):
        raise ProtocolError(f"message 'id' must be an integer (got {message_id!r})")
    return message_id


def decode(line: str) -> Message:
    """Parse one JSON line into a message."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON line: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"message must be a JSON object (got {type(payload).__name__})")

    kind = payload.get("type")
    if kind == "query":
        views = payload.get("views", {})
        raw_questions = payload.get("questions", {})
        if not isinstance(views, dict) or not isinstance(raw_questions, dict):
            raise ProtocolError("query 'views' and 'questions' must be objects")
        specs: dict[str, QuestionSpec] = {}
        for name, raw in raw_questions.items():
            try:
                specs[name] = parse_question(name, raw, f"questions.{name}")
            except ManifestError as exc:
                raise ProtocolError(str(exc)) from exc
        return QueryMessage(id=_require_int_id(payload), views=views, questions=specs)
    if kind == "result":
        answers = payload.get("answers", {})
        if not isinstance(answers, dict):
            raise ProtocolError("result 'answers' must be an object")
        return ResultMessage(
            id=_require_int_id(payload),
            answers=answers,
            latency_ms=float(payload.get("latency_ms", 0.0)),
            warnings=tuple(payload.get("warnings", ())),
        )
    if kind == "error":
        return ErrorMessage(id=payload.get("id"), message=str(payload.get("message", "")))
    if kind == "ready":
        return ReadyMessage()
    if kind == "shutdown":
        return ShutdownMessage()
    raise ProtocolError(f"unknown message type {kind!r}")


def write(stream: TextIO, message: Message) -> None:
    """Write one message to a text stream and flush."""
    stream.write(encode(message))
    stream.flush()


def read(stream: TextIO) -> Message | None:
    """Read one message; ``None`` at end-of-stream."""
    line = stream.readline()
    if not line:
        return None
    return decode(line)
