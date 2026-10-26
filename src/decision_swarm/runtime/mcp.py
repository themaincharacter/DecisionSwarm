"""Model Context Protocol (tools only) over stdio.

Newline-framed JSON-RPC 2.0, the way agent hosts spawn a local server: the
client writes a request line, the server writes one response line. Only the
``initialize`` handshake, ``ping``, ``tools/list`` and ``tools/call`` are
implemented — no resources, prompts or sampling.

The protocol is hand-rolled on purpose. The official SDK pulls pydantic, anyio
and httpx into the engine, which also ships as a frozen PyInstaller binary and a
Tauri sidecar where an optional extra cannot be installed. ``tests/
test_mcp_conformance.py`` runs this implementation against the real SDK when it
is available, so the wire cannot drift unnoticed.

stdout is the transport, so :func:`serve` moves the *file descriptor* to stderr for
the life of the session (see :func:`_claim_stdout`). Redirecting ``sys.stdout``
alone would not be enough: subprocesses inherit the descriptor, not the Python
object, so a `pip` run or a `venv` creation would still land on the wire.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from typing import Any, Protocol, TextIO

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
"""Protocol revisions this server speaks, newest first."""

SERVER_NAME = "decision-swarm"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ToolLike(Protocol):
    """What the wire needs from one tool (structurally satisfied by ToolSpec)."""

    name: str
    description: str
    schema: Mapping[str, Any]

    def handler(self, arguments: Mapping[str, Any]) -> dict[str, Any]: ...


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _text_result(payload: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": payload}], "isError": is_error}


def _tool_payload(tool: ToolLike) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": dict(tool.schema),
    }


def handle_message(
    message: Any,
    tools: Mapping[str, ToolLike],
    *,
    server_name: str = SERVER_NAME,
    server_version: str = "0",
) -> dict[str, Any] | None:
    """One decoded JSON-RPC message -> one response, or ``None`` for a notification."""
    if not isinstance(message, Mapping):
        return _error(None, INVALID_REQUEST, "invalid request: expected a JSON object")

    has_id = "id" in message
    request_id = message.get("id")
    method = message.get("method")
    if not isinstance(method, str):
        if not has_id:
            return None
        return _error(request_id, INVALID_REQUEST, "invalid request: 'method' must be a string")

    params = message.get("params")
    params = params if isinstance(params, Mapping) else {}

    if method.startswith("notifications/"):
        return None

    if method == "initialize":
        requested = params.get("protocolVersion")
        protocol = (
            requested
            if isinstance(requested, str) and requested in SUPPORTED_PROTOCOL_VERSIONS
            else SUPPORTED_PROTOCOL_VERSIONS[0]
        )
        result: dict[str, Any] = {
            "protocolVersion": protocol,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": server_name, "version": server_version},
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": [_tool_payload(tool) for tool in tools.values()]}
    elif method == "tools/call":
        return _call_tool(request_id, has_id, params, tools)
    else:
        if not has_id:
            return None
        return _error(request_id, METHOD_NOT_FOUND, f"method not found: {method}")

    if not has_id:
        return None
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _call_tool(
    request_id: Any,
    has_id: bool,
    params: Mapping[str, Any],
    tools: Mapping[str, ToolLike],
) -> dict[str, Any] | None:
    if not has_id:
        return None
    name = params.get("name")
    if not isinstance(name, str):
        return _error(request_id, INVALID_PARAMS, "invalid params: 'name' must be a string")
    tool = tools.get(name)
    if tool is None:
        return _error(request_id, INVALID_PARAMS, f"unknown tool: {name}")
    arguments = params.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        return _error(request_id, INVALID_PARAMS, "invalid params: 'arguments' must be an object")

    try:
        payload = tool.handler(arguments)
    except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
        return {"jsonrpc": "2.0", "id": request_id, "result": _text_result(str(exc), is_error=True)}
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": _text_result(json.dumps(payload), is_error=False),
    }


def handle_line(
    line: str,
    tools: Mapping[str, ToolLike],
    *,
    server_name: str = SERVER_NAME,
    server_version: str = "0",
) -> dict[str, Any] | None:
    """Parse one line and handle it; malformed JSON becomes a parse error."""
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        return _error(None, PARSE_ERROR, f"parse error: {exc}")
    return handle_message(
        message, tools, server_name=server_name, server_version=server_version
    )


def _claim_stdout() -> TextIO:
    """Take the real stdout for the protocol and point fd 1 at stderr.

    Every child process the tools spawn — `venv`, `pip`, anything a loader
    shells out to — writes to fd 1. Handing fd 1 to stderr and keeping a private
    handle on the original descriptor is what keeps them off the wire.
    """
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:  # pragma: no cover - no usable stdout (closed or detached)
        return sys.stdout
    os.dup2(2, 1)
    return os.fdopen(saved, "w", encoding="utf-8", buffering=1)


def _release_stdout(wire: TextIO) -> None:
    """Flush the protocol stream and give fd 1 back to the caller."""
    wire.flush()
    if wire is sys.stdout:  # pragma: no cover - the fallback above
        return
    try:
        os.dup2(wire.fileno(), 1)
    except (OSError, ValueError):  # pragma: no cover - already detached
        return
    wire.close()


def serve(
    tools: Mapping[str, ToolLike],
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    server_name: str = SERVER_NAME,
    server_version: str = "0",
) -> int:
    """Run the MCP loop until end-of-input; returns the process exit code.

    With no ``stdout`` (the real process) the wire takes over fd 1 and every
    other writer on this process is pointed at stderr. Passing ``stdout`` is for
    tests and leaves the process descriptors alone.
    """
    own_streams = stdout is None
    wire = _claim_stdout() if own_streams else stdout

    stream = sys.stdin if stdin is None else stdin
    try:
        while True:
            line = stream.readline()
            if not line:
                return 0
            line = line.strip()
            if not line:
                continue
            response = handle_line(
                line, tools, server_name=server_name, server_version=server_version
            )
            if response is None:
                continue
            wire.write(json.dumps(response) + "\n")
            wire.flush()
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 0
    finally:
        if own_streams:
            _release_stdout(wire)
