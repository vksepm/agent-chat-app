"""
src/mcp_client.py — MCP server client utilities.

Provides build_mcp_tools() which opens MCPClient connections to each configured
MCP server (HTTP Streamable transport), merges the returned tool catalogues, and
returns the combined list ready to pass to the agent factory.

Error handling:
- MCPToolError is raised (and caught internally per-server) when a tool call
  returns isError=true or when the transport layer fails during a tool invocation.
  This class is also imported by app.py so it can be excluded from the top-level
  catch-all handler (T013 boundary).
- A ConnectionError during tool discovery (tools/list at startup) is caught
  per-server: the server is skipped with a warning, tools from other servers
  still load.
"""

import logging
from contextlib import ExitStack
from typing import Any

from smolagents import MCPClient

logger = logging.getLogger(__name__)


class MCPToolError(Exception):
    """Raised when an MCP tool call returns isError=true or the transport fails."""


def _validate_url(url: str) -> None:
    if not url.startswith("https://"):
        raise ValueError(
            f"MCP server URL must use HTTPS. Got: {url!r}. "
            "Plain HTTP is not permitted."
        )


def _wrap_tool(tool: Any) -> Any:
    """
    Patch tool.forward() to convert any MCP-layer exception into MCPToolError.

    The smolagents ToolCallingAgent catches exceptions raised by forward() and
    feeds the message back to the LLM as the tool's "observation" — so raising
    MCPToolError causes the agent to reason with "[Tool unavailable: ...]" as
    context rather than crashing.  All other (non-MCP) exceptions propagate
    upward from agent.run() to the T013 catch-all in app.py.
    """
    original_forward = tool.forward

    def _safe_forward(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_forward(*args, **kwargs)
        except Exception as exc:
            safe_msg = f"[Tool unavailable: {exc}]"
            raise MCPToolError(safe_msg) from exc

    tool.forward = _safe_forward
    return tool


def build_mcp_tools(urls: list[str]) -> tuple[list[Any], ExitStack]:
    """
    Open MCPClient connections to each URL in *urls* and return
    (combined_tool_list, exit_stack).

    The caller is responsible for calling exit_stack.close() when the app shuts
    down (or use it as a context manager).  A failure to connect to one server
    is logged as a warning and does not prevent tools from other servers loading.
    """
    stack = ExitStack()
    all_tools: list[Any] = []

    for url in urls:
        if not url:
            continue
        try:
            _validate_url(url)
            client = MCPClient({"url": url, "transport": "streamable-http"})
            tools = stack.enter_context(client)
            wrapped = [_wrap_tool(t) for t in tools]
            all_tools.extend(wrapped)
            logger.info("Loaded %d tool(s) from MCP server: %s", len(wrapped), url)
        except ConnectionError as exc:
            logger.warning(
                "Could not connect to MCP server %r — skipping. Error: %s",
                url,
                exc,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load tools from MCP server %r — skipping. Error: %s",
                url,
                exc,
            )

    if not all_tools:
        logger.warning(
            "No MCP tools loaded from any server. Agent will operate without tools."
        )

    return all_tools, stack
