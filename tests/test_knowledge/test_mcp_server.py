"""Testes do MCP server stdio (QG-C11-06)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.knowledge import mcp_server

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.qg_c11
class TestMcpTools:
    """Validação das tools expostas pelo MCP server."""

    def test_list_layers_returns_expected_layers(self) -> None:
        layers = mcp_server.list_layers()
        assert isinstance(layers, list)
        assert "C01" in layers
        assert "C10" in layers
        assert "SDD" in layers
        assert "UNIFIED" in layers

    def test_search_docs_returns_formatted_context(
        self,
        populated_db: Any,
    ) -> None:
        result = mcp_server.search_docs(query="firmware STM32", layer="C08", k=3)
        assert isinstance(result, str)
        assert "C08" in result
        assert "Tags:" in result

    def test_get_doc_by_source_returns_matching_chunks(
        self,
        populated_db: Any,
    ) -> None:
        source = "docs/Camada-04-Modelagem-v1.1.md"
        result = mcp_server.get_doc_by_source(source, k=3)
        assert isinstance(result, str)
        assert source in result
        assert "F1-macro" in result

    def test_tools_are_registered_in_mcp_app(self) -> None:
        """Verifica que as três tools foram registradas no FastMCP."""
        tools = mcp_server.mcp._tool_manager._tools
        assert "search_docs" in tools
        assert "list_layers" in tools
        assert "get_doc_by_source" in tools


@pytest.mark.qg_c11
class TestMcpProtocol:
    """QG-C11-06: servidor responde a initialize e tools/list via JSON-RPC."""

    def test_initialize_and_tools_list(self) -> None:
        proc: Any = subprocess.Popen(
            [sys.executable, "-m", "src.knowledge.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": str(PROJECT_ROOT)},
        )

        try:
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest-mcp", "version": "1.0.0"},
                },
            }
            _send(proc, init_request)
            init_response = _recv(proc.stdout)
            assert init_response["id"] == 1
            assert "result" in init_response
            assert init_response["result"]["serverInfo"]["name"] == "project-lewis-knowledge"

            tools_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
            _send(proc, tools_request)
            tools_response = _recv(proc.stdout)
            assert tools_response["id"] == 2
            assert "result" in tools_response

            tools = tools_response["result"]["tools"]
            names = {tool["name"] for tool in tools}
            assert "search_docs" in names
            assert "list_layers" in names
            assert "get_doc_by_source" in names
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def _send(proc: Any, message: dict[str, Any]) -> None:
    line = json.dumps(message) + "\n"
    assert proc.stdin is not None
    proc.stdin.write(line)
    proc.stdin.flush()


def _recv(stream: Any) -> dict[str, Any]:
    line = stream.readline()
    if not line:
        raise RuntimeError("MCP server não retornou resposta (stdout vazio)")
    return json.loads(line)
