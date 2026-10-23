"""Tests for the JSON-lines stdio protocol."""

from __future__ import annotations

import io

import pytest

from decision_swarm import manifest as manifest_mod
from decision_swarm.runtime import protocol


def _questions() -> dict[str, manifest_mod.QuestionSpec]:
    spec = manifest_mod.parse_question(
        "submit_visible",
        {"type": "noul", "instructions": "Do you see a submit button?"},
        "questions.submit_visible",
    )
    return {spec.name: spec}


def test_query_round_trip() -> None:
    message = protocol.QueryMessage(
        id=7,
        views={"screenshot": "shot.png"},
        questions=_questions(),
    )
    decoded = protocol.decode(protocol.encode(message))
    assert decoded == message


def test_result_round_trip() -> None:
    message = protocol.ResultMessage(
        id=7,
        answers={"submit_visible": {"noul": 0.94, "confidence": 0.91}},
        latency_ms=41.5,
        warnings=("model is warm",),
    )
    decoded = protocol.decode(protocol.encode(message))
    assert decoded == message


def test_error_and_shutdown_round_trip() -> None:
    for message in (protocol.ErrorMessage(message="boom", id=3), protocol.ErrorMessage(message="fatal"), protocol.ShutdownMessage()):
        assert protocol.decode(protocol.encode(message)) == message


def test_stream_write_and_read() -> None:
    stream = io.StringIO()
    protocol.write(stream, protocol.QueryMessage(id=1, views={"dom": "<html>"}, questions={}))
    protocol.write(stream, protocol.ResultMessage(id=1, answers={}, latency_ms=1.0))
    protocol.write(stream, protocol.ShutdownMessage())

    stream.seek(0)
    assert isinstance(protocol.read(stream), protocol.QueryMessage)
    assert isinstance(protocol.read(stream), protocol.ResultMessage)
    assert isinstance(protocol.read(stream), protocol.ShutdownMessage)
    assert protocol.read(stream) is None  # EOF


def test_encode_is_single_line_terminated() -> None:
    line = protocol.encode(protocol.ShutdownMessage())
    assert line.endswith("\n") and line.count("\n") == 1


@pytest.mark.parametrize(
    "line, fragment",
    [
        ("{not json\n", "invalid JSON line"),
        ("[1, 2]\n", "must be a JSON object"),
        ('{"type": "mystery"}\n', "unknown message type"),
        ('{"type": "query", "views": 3}\n', "must be objects"),
        ('{"type": "query", "id": "x"}\n', "'id' must be an integer"),
        ('{"type": "result", "id": 1, "answers": []}\n', "'answers' must be an object"),
        (
            '{"type": "query", "id": 1, "questions": {"q": {"type": "noul", "instructions": "ok?", "criteria": ["x"]}}}\n',
            "noul questions take no criteria",
        ),
    ],
)
def test_decode_rejects_bad_lines(line: str, fragment: str) -> None:
    with pytest.raises(protocol.ProtocolError) as excinfo:
        protocol.decode(line)
    assert fragment in str(excinfo.value)
