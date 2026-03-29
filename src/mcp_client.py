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

# Sentinel — the smolagents Tool callable attribute name.
# If smolagents renames this method, update here only.
# See: https://huggingface.co/docs/smolagents/reference/tools
_TOOL_CALL_ATTR = "forward"


class MCPToolError(Exception):
    """Raised when an MCP tool call fails at the transport or protocol layer.

    Attributes
    ----------
    tool_name : str
        Name of the tool that failed (for logging/metrics).
    category : str
        One of ``"network"``, ``"auth"``, ``"tool_error"``, ``"unknown"``.
    safe_message : str
        Sanitised message safe to expose to the LLM and UI.
    """

    def __init__(
        self,
        safe_message: str,
        tool_name: str = "unknown",
        category: str = "unknown",
    ) -> None:
        super().__init__(safe_message)
        self.tool_name = tool_name
        self.category = category
        self.safe_message = safe_message


def _validate_url(url: str) -> None:
    if not url.startswith("https://"):
        raise ValueError(
            f"MCP server URL must use HTTPS. Got: {url!r}. "
            "Plain HTTP is not permitted."
        )


def _sanitize_error(exc: Exception) -> tuple[str, str]:
    """Return ``(safe_message, category)`` from a raw exception.

    Strips sensitive content (URLs, tokens, full tracebacks) from the message.
    Category is one of: ``"network"``, ``"auth"``, ``"tool_error"``,
    ``"unknown"``.

    Parameters
    ----------
    exc:
        The original exception raised by the tool call.

    Returns
    -------
    tuple[str, str]
        ``(safe_message, category)`` where ``safe_message`` is safe to expose
        to the LLM / UI and ``category`` identifies the failure type.
    """
    raw = str(exc)

    if isinstance(exc, (ConnectionError, TimeoutError)) or "timeout" in raw.lower():
        return "Tool is temporarily unreachable (network error).", "network"
    if "401" in raw or "403" in raw or "auth" in raw.lower():
        return "Tool authentication failed.", "auth"
    if "tool_error" in raw.lower() or "isError" in raw:
        # Limit exposed message length — remote error body may be large.
        truncated = raw[:200] + ("..." if len(raw) > 200 else "")
        return f"Tool returned an error: {truncated}", "tool_error"
    return "Tool call failed (unknown error).", "unknown"


def _wrap_tool(tool: Any) -> Any:
    """Patch tool's callable attribute to convert exceptions into MCPToolError.

    The smolagents ToolCallingAgent catches exceptions raised by forward() and
    feeds the message back to the LLM as the tool's "observation" — so raising
    MCPToolError causes the agent to reason with "[Tool unavailable: ...]" as
    context rather than crashing.  All other (non-MCP) exceptions propagate
    upward from agent.run() to the T013 catch-all in app.py.

    Parameters
    ----------
    tool:
        A smolagents Tool object returned by MCPClient.

    Returns
    -------
    The same tool object, with its ``forward`` method patched.

    Raises
    ------
    ValueError
        If the tool does not have the expected callable attribute
        (guarded by ``_TOOL_CALL_ATTR``).
    """
    if not hasattr(tool, _TOOL_CALL_ATTR):
        raise ValueError(
            f"Cannot wrap tool {tool!r}: missing attribute '{_TOOL_CALL_ATTR}'. "
            f"Expected a smolagents Tool with a '{_TOOL_CALL_ATTR}()' method. "
            f"If smolagents renamed the callable, update _TOOL_CALL_ATTR in "
            f"src/mcp_client.py."
        )

    tool_name = getattr(tool, "name", "unknown")
    original = getattr(tool, _TOOL_CALL_ATTR)

    def _safe_forward(*args: Any, **kwargs: Any) -> Any:
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            safe_msg, category = _sanitize_error(exc)
            logger.warning(
                "MCP tool '%s' failed [%s]: %s",
                tool_name,
                category,
                safe_msg,
            )
            raise MCPToolError(
                f"[Tool unavailable: {safe_msg}]",
                tool_name=tool_name,
                category=category,
            ) from exc

    setattr(tool, _TOOL_CALL_ATTR, _safe_forward)
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
