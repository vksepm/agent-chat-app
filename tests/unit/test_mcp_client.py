"""
tests/unit/test_mcp_client.py — Boundary tests for src/mcp_client.

All smolagents types are mocked with plain Python objects.
No live MCP server or credentials required.
"""

import logging
import sys
from unittest.mock import MagicMock

import pytest

# smolagents is not installed in this environment; mock it before importing
# src.mcp_client which does `from smolagents import MCPClient` at module level.
sys.modules.setdefault("smolagents", MagicMock())

from src.mcp_client import (  # noqa: E402 — must come after sys.modules patch
    MCPToolError,
    _TOOL_CALL_ATTR,
    _sanitize_error,
    _wrap_tool,
)


# ---------------------------------------------------------------------------
# Mock tool objects
# ---------------------------------------------------------------------------


class _MockTool:
    name = "search"

    def forward(self, query: str) -> str:
        return f"results for {query}"


class _MockFailingTool:
    name = "search"

    def forward(self, query: str) -> str:
        raise ConnectionError("Connection refused: 10.0.0.1:8080")


class _MockToolNoForward:
    name = "broken"
    # Intentionally missing the 'forward' attribute


# ---------------------------------------------------------------------------
# MCPToolError attributes
# ---------------------------------------------------------------------------


class TestMCPToolError:
    def test_mcp_tool_error_exposes_tool_name_and_category(self):
        exc = MCPToolError("[Tool unavailable: auth failed.]", tool_name="weather", category="auth")
        assert exc.tool_name == "weather"
        assert exc.category == "auth"
        assert exc.safe_message == "[Tool unavailable: auth failed.]"
        assert str(exc) == "[Tool unavailable: auth failed.]"

    def test_mcp_tool_error_defaults(self):
        exc = MCPToolError("oops")
        assert exc.tool_name == "unknown"
        assert exc.category == "unknown"


# ---------------------------------------------------------------------------
# _sanitize_error — pure function tests
# ---------------------------------------------------------------------------


class TestSanitizeError:
    def test_sanitize_network_error_connection_error(self):
        safe_msg, category = _sanitize_error(
            ConnectionError("Connection refused: 10.0.0.1:8080")
        )
        assert category == "network"
        assert "10.0.0.1" not in safe_msg
        assert "8080" not in safe_msg
        assert "unreachable" in safe_msg.lower() or "network" in safe_msg.lower()

    def test_sanitize_network_error_timeout(self):
        safe_msg, category = _sanitize_error(TimeoutError("request timed out"))
        assert category == "network"

    def test_sanitize_network_error_timeout_keyword_in_message(self):
        safe_msg, category = _sanitize_error(RuntimeError("socket timeout after 30s"))
        assert category == "network"

    def test_sanitize_auth_error_401(self):
        safe_msg, category = _sanitize_error(RuntimeError("HTTP 401 Unauthorized"))
        assert category == "auth"
        assert "authentication" in safe_msg.lower() or "auth" in safe_msg.lower()

    def test_sanitize_auth_error_403(self):
        safe_msg, category = _sanitize_error(RuntimeError("403 Forbidden"))
        assert category == "auth"

    def test_sanitize_auth_error_keyword(self):
        safe_msg, category = _sanitize_error(RuntimeError("auth token expired"))
        assert category == "auth"

    def test_sanitize_tool_error_isError_flag(self):
        safe_msg, category = _sanitize_error(RuntimeError("isError: true — bad argument"))
        assert category == "tool_error"
        assert "Tool returned an error" in safe_msg

    def test_sanitize_tool_error_truncated(self):
        long_msg = "tool_error: " + "x" * 300
        safe_msg, category = _sanitize_error(RuntimeError(long_msg))
        assert category == "tool_error"
        # The raw portion shown must be ≤ 200 chars (plus the prefix and "...")
        # Extract the part after "Tool returned an error: "
        raw_part = safe_msg[len("Tool returned an error: "):]
        assert len(raw_part) <= 204  # 200 chars + "..."

    def test_sanitize_unknown_error(self):
        safe_msg, category = _sanitize_error(RuntimeError("internal server error at /secret/path"))
        assert category == "unknown"
        assert "/secret/path" not in safe_msg
        assert "unknown" in safe_msg.lower() or "failed" in safe_msg.lower()


# ---------------------------------------------------------------------------
# _wrap_tool tests
# ---------------------------------------------------------------------------


class TestWrapTool:
    def test_wrap_tool_calls_original_forward(self):
        tool = _MockTool()
        wrapped = _wrap_tool(tool)
        result = wrapped.forward("python")
        assert result == "results for python"

    def test_wrap_tool_raises_mcp_tool_error_on_failure(self):
        tool = _MockFailingTool()
        wrapped = _wrap_tool(tool)
        with pytest.raises(MCPToolError) as exc_info:
            wrapped.forward("test")
        error = exc_info.value
        # Safe message must NOT contain the raw IP address
        assert "10.0.0.1" not in str(error)
        assert "Connection refused" not in str(error)

    def test_wrap_tool_raises_on_missing_forward(self):
        with pytest.raises(ValueError, match=_TOOL_CALL_ATTR):
            _wrap_tool(_MockToolNoForward())

    def test_tool_call_attr_constant_guards_interface(self):
        """Wrapping a tool missing the constant attr raises ValueError."""

        class _NoForwardTool:
            name = "noop"

        with pytest.raises(ValueError):
            _wrap_tool(_NoForwardTool())

    def test_wrap_tool_sets_correct_tool_name(self):
        tool = _MockFailingTool()
        wrapped = _wrap_tool(tool)
        with pytest.raises(MCPToolError) as exc_info:
            wrapped.forward("x")
        assert exc_info.value.tool_name == "search"

    def test_wrap_tool_sets_correct_category(self):
        tool = _MockFailingTool()  # raises ConnectionError → "network"
        wrapped = _wrap_tool(tool)
        with pytest.raises(MCPToolError) as exc_info:
            wrapped.forward("x")
        assert exc_info.value.category == "network"

    def test_wrap_tool_logs_warning_on_failure(self, caplog):
        tool = _MockFailingTool()
        wrapped = _wrap_tool(tool)
        with caplog.at_level(logging.WARNING, logger="src.mcp_client"):
            with pytest.raises(MCPToolError):
                wrapped.forward("x")
        assert any("search" in r.message for r in caplog.records)
        assert any("network" in r.message for r in caplog.records)

    def test_wrap_tool_original_exception_chained(self):
        tool = _MockFailingTool()
        wrapped = _wrap_tool(tool)
        with pytest.raises(MCPToolError) as exc_info:
            wrapped.forward("x")
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    def test_wrap_tool_returns_same_tool_object(self):
        tool = _MockTool()
        returned = _wrap_tool(tool)
        assert returned is tool
