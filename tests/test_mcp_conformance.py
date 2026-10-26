"""Conformance: the hand-rolled wire against the official MCP SDK client.

Skipped unless the ``conformance`` extra is installed
(``pip install -e ".[conformance]"``). The wire has no runtime dependency on the
SDK — this test exists so the hand-written protocol cannot drift unnoticed, and it
is the only place the protocol meets an implementation we did not write.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp", reason="install the 'conformance' extra to run this test")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from decision_swarm import manifest as manifest_mod, pack  # noqa: E402
from test_orchestrator import BROWSER_SWARM  # noqa: E402

EXPECTED_TOOLS = {
    "describe",
    "judge",
    "list_models",
    "build",
    "fetch",
    "install",
    "export",
    "job_status",
}


def test_sdk_client_handshakes_lists_tools_and_describes(tmp_path: Path) -> None:
    pack_dir = pack.build_pack(
        manifest_mod.parse_manifest(BROWSER_SWARM),
        tmp_path / "packs" / "browser-swarm",
        models_root=tmp_path / "cache",
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "decision_swarm.cli",
            "mcp",
            "--packs-dir",
            str(pack_dir.parent),
            "--pack",
            str(pack_dir),
        ],
    )

    async def run() -> tuple[list[str], str, bool]:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
                called = await session.call_tool("describe", {})
                text = called.content[0].text if called.content else ""
                return [tool.name for tool in listed.tools], text, bool(called.is_error)

    names, text, is_error = asyncio.run(run())

    assert set(names) == EXPECTED_TOOLS
    assert is_error is False
    assert "browser-swarm" in text
    assert "submit_button" in text
