"""Tests for the MCP wire: JSON-RPC framing, tool listing and calls."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from decision_swarm import agent
from decision_swarm.runtime import mcp


def _echo_tool(name: str = "echo") -> agent.ToolSpec:
    def handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        return {"echo": dict(arguments)}

    return agent.ToolSpec(
        name=name,
        description="echo the arguments back",
        schema={"type": "object", "properties": {"value": {"type": "string"}}},
        handler=handler,
    )


def _failing_tool() -> agent.ToolSpec:
    def handler(arguments: Mapping[str, Any]) -> dict[str, Any]:
        raise agent.AgentError("model(s) not in the cache: laya; call 'fetch' first")

    return agent.ToolSpec(
        name="judge",
        description="always fails",
        schema={"type": "object"},
        handler=handler,
    )


def _table() -> dict[str, agent.ToolSpec]:
    return {"echo": _echo_tool(), "judge": _failing_tool()}


def _initialize(requested: str | None = None) -> str:
    params: dict[str, Any] = {"capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}
    if requested is not None:
        params["protocolVersion"] = requested
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": params})


def test_initialize_negotiates_the_requested_version() -> None:
    response = mcp.handle_line(_initialize("2025-03-26"), _table())

    assert response is not None
    result = response["result"]
    assert response["id"] == 1
    assert result["protocolVersion"] == "2025-03-26"
    assert result["capabilities"]["tools"] == {"listChanged": False}
    assert result["serverInfo"]["name"] == "decision-swarm"


def test_initialize_falls_back_to_the_newest_version() -> None:
    response = mcp.handle_line(_initialize("1999-01-01"), _table())

    assert response is not None
    assert response["result"]["protocolVersion"] == mcp.SUPPORTED_PROTOCOL_VERSIONS[0]


def test_notifications_get_no_response() -> None:
    line = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert mcp.handle_line(line, _table()) is None


def test_tools_list_describes_each_tool() -> None:
    line = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    response = mcp.handle_line(line, _table())

    assert response is not None
    tools = {tool["name"]: tool for tool in response["result"]["tools"]}
    assert set(tools) == {"echo", "judge"}
    assert tools["echo"]["description"] == "echo the arguments back"
    assert tools["echo"]["inputSchema"]["properties"]["value"]["type"] == "string"


def test_tools_call_returns_json_text_content() -> None:
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"value": "hi"}},
        }
    )
    response = mcp.handle_line(line, _table())

    assert response is not None
    result = response["result"]
    assert result["isError"] is False
    assert result["content"][0]["type"] == "text"
    assert json.loads(result["content"][0]["text"]) == {"echo": {"value": "hi"}}


def test_tools_call_reports_a_tool_failure_as_is_error() -> None:
    line = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "judge", "arguments": {"views": {"dom": "x"}}},
        }
    )
    response = mcp.handle_line(line, _table())

    assert response is not None
    result = response["result"]
    assert result["isError"] is True
    assert "call 'fetch' first" in result["content"][0]["text"]


def test_unknown_tool_is_an_invalid_params_error() -> None:
    line = json.dumps(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "nope"}}
    )
    response = mcp.handle_line(line, _table())

    assert response is not None
    assert response["error"]["code"] == mcp.INVALID_PARAMS


def test_unknown_method_is_method_not_found() -> None:
    line = json.dumps({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
    response = mcp.handle_line(line, _table())

    assert response is not None
    assert response["error"]["code"] == mcp.METHOD_NOT_FOUND


def test_malformed_json_is_a_parse_error() -> None:
    response = mcp.handle_line("{broken", _table())

    assert response is not None
    assert response["error"]["code"] == mcp.PARSE_ERROR
    assert response["id"] is None


def test_ping_is_answered() -> None:
    line = json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"})
    assert mcp.handle_line(line, _table()) == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_serve_writes_only_json_rpc_lines() -> None:
    stdin = io.StringIO(
        "\n".join(
            [
                _initialize("2025-06-18"),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                "",
                "{bad json",
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "echo", "arguments": {}},
                    }
                ),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()

    code = mcp.serve(_table(), stdin=stdin, stdout=stdout)

    assert code == 0
    lines = stdout.getvalue().splitlines()
    # one response for initialize, tools/list, the parse error and the call
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert [item["id"] for item in parsed] == [1, 2, None, 3]


def test_serve_stops_at_end_of_input() -> None:
    assert mcp.serve(_table(), stdin=io.StringIO(""), stdout=io.StringIO()) == 0


def test_serve_restores_stdout_when_it_owns_the_wire() -> None:
    original = sys.stdout
    assert mcp.serve(_table(), stdin=io.StringIO("")) == 0
    assert sys.stdout is original


def test_claim_stdout_moves_fd_one_to_stderr(tmp_path: Path) -> None:
    """A child process must inherit stderr, not the protocol stream."""
    target = tmp_path / "redirected.txt"
    with open(target, "w", encoding="utf-8") as sink:
        saved = os.dup(1)
        os.dup2(sink.fileno(), 1)
        try:
            wire = mcp._claim_stdout()
            try:
                assert os.fstat(1) == os.fstat(2)
            finally:
                mcp._release_stdout(wire)
            # fd 1 is back on the original sink, so a child's print lands there
            subprocess.run([sys.executable, "-c", "print('child')"], check=True)
        finally:
            os.dup2(saved, 1)
            os.close(saved)

    assert "child" in target.read_text(encoding="utf-8")
