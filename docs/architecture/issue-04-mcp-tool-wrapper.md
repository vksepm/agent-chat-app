# Issue 04 — Deepen MCP Tool Wrapping + Error Sanitization in `mcp_client.py`

## Problem

`_wrap_tool()` in `src/mcp_client.py` (lines 39–59) monkey-patches the `.forward()`
method of every tool object returned by `smolagents.MCPClient`:

```python
def _wrap_tool(tool: Any) -> Any:
    original_forward = tool.forward

    def _safe_forward(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_forward(*args, **kwargs)
        except Exception as exc:
            safe_msg = f"[Tool unavailable: {exc}]"
            raise MCPToolError(safe_msg) from exc

    tool.forward = _safe_forward
    return tool
```

Three problems with this implementation:

1. **No interface guard.** The function assumes every tool has a `.forward()` method
   — true for `smolagents.Tool` today, but not validated.  If smolagents renames the
   callable to `.run()` or `.invoke()`, this silently produces a tool whose
   `.forward` is the original (unguarded) callable and `_safe_forward` is never
   installed.

2. **Full exception text leaked in error messages.** `f"[Tool unavailable: {exc}]"`
   includes the original exception string, which may contain sensitive data from
   transport-layer errors (e.g., internal server addresses, auth headers from HTTP
   error bodies, stack traces from remote MCP servers).  These strings are fed
   directly into the LLM as observations and logged to stdout.

3. **No distinction between error types.** A network timeout, an auth failure, and
   a tool logic error all produce the same `MCPToolError` string.  The agent cannot
   reason differently about "server is down" vs. "bad arguments".

Additionally, `_wrap_tool()` is only exercised via the integration test in
`tests/integration/test_agent_mcp.py`, which requires live MCP server credentials.
There is no unit test for the wrapping behaviour itself.

**Integration risk at the seam (mcp_client ↔ agent):**
`MCPToolError` is imported by `app.py` to catch the rare case where it escapes
agent reasoning (lines 439–451).  The comment says the error "is already converted
to a safe string" by the wrapper — but there is no test to verify that the wrapper
actually prevents sensitive content from reaching the LLM or the UI.

## Proposed Interface

Strengthen `_wrap_tool()` with an explicit interface check, error sanitization, and
categorized error messages:

```python
# src/mcp_client.py  (revised internals)

# Sentinel — the smolagents Tool callable attribute name.
# Update here if upstream renames it.
_TOOL_CALL_ATTR = "forward"

class MCPToolError(Exception):
    """
    Raised when an MCP tool call fails at the transport or protocol layer.

    Attributes
    ----------
    tool_name : str
        Name of the tool that failed (for logging/metrics).
    category : str
        One of "network", "auth", "tool_error", "unknown".
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


def _sanitize_error(exc: Exception) -> tuple[str, str]:
    """
    Return (safe_message, category) from a raw exception.

    Strips sensitive content (URLs, tokens, full tracebacks) from the message.
    Category is one of: "network", "auth", "tool_error", "unknown".
    """
    raw = str(exc)
    exc_type = type(exc).__name__

    if isinstance(exc, (ConnectionError, TimeoutError)) or "timeout" in raw.lower():
        return "Tool is temporarily unreachable (network error).", "network"
    if "401" in raw or "403" in raw or "auth" in raw.lower():
        return "Tool authentication failed.", "auth"
    if "tool_error" in raw.lower() or "isError" in raw:
        # Limit exposed message length — remote error body may be large
        truncated = raw[:200] + ("..." if len(raw) > 200 else "")
        return f"Tool returned an error: {truncated}", "tool_error"
    return "Tool call failed (unknown error).", "unknown"


def _wrap_tool(tool: Any) -> Any:
    """
    Patch tool's callable attribute to convert exceptions into MCPToolError.

    Raises
    ------
    ValueError
        If the tool does not have the expected callable attribute.
    """
    if not hasattr(tool, _TOOL_CALL_ATTR):
        raise ValueError(
            f"Cannot wrap tool {tool!r}: missing attribute '{_TOOL_CALL_ATTR}'. "
            f"Expected a smolagents Tool with a '{_TOOL_CALL_ATTR}()' method."
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
                tool_name, category, safe_msg,
            )
            raise MCPToolError(
                f"[Tool unavailable: {safe_msg}]",
                tool_name=tool_name,
                category=category,
            ) from exc

    setattr(tool, _TOOL_CALL_ATTR, _safe_forward)
    return tool
```

What the revised implementation hides:
- Raw exception text from transport layer.
- The fragile assumption about attribute name (now a single constant).
- Category logic for distinguishing network vs. auth vs. tool errors.

## Dependency Strategy

**True external (Mock)** — smolagents `Tool` objects and `MCPClient` are third-party.

- Tests construct minimal mock tool objects (plain Python objects with a `.forward`
  attribute) — no live MCP server required.
- `_sanitize_error()` is a pure function and can be tested with raw exception
  instances.

```python
# tests/unit/test_mcp_client.py

class _MockTool:
    name = "search"
    def forward(self, query: str) -> str:
        return f"results for {query}"

class _MockFailingTool:
    name = "search"
    def forward(self, query: str) -> str:
        raise ConnectionError("Connection refused: 10.0.0.1:8080")
```

## Testing Strategy

**New unit tests to write (`tests/unit/test_mcp_client.py`):**

- `test_wrap_tool_calls_original_forward` — wrapped tool with passing `forward()`;
  assert return value unchanged.
- `test_wrap_tool_raises_mcp_tool_error_on_failure` — wrapped tool whose `forward()`
  raises; assert `MCPToolError` raised, `safe_message` does not contain the original
  exception string.
- `test_wrap_tool_raises_on_missing_forward` — tool without `.forward()` attribute;
  assert `ValueError` raised with informative message.
- `test_sanitize_network_error` — `ConnectionError` input; assert category
  `"network"`, safe message does not contain IP address.
- `test_sanitize_auth_error` — exception with `"401"` in message; assert category
  `"auth"`.
- `test_sanitize_tool_error` — exception with `"isError"` in message; assert
  category `"tool_error"`, message truncated to ≤ 200 chars.
- `test_sanitize_unknown_error` — generic `RuntimeError`; assert category
  `"unknown"`, safe message does not include raw exception text.
- `test_wrap_tool_logs_warning_on_failure` — use `caplog`; assert `WARNING` logged
  with tool name and category.
- `test_mcp_tool_error_exposes_tool_name_and_category` — assert `MCPToolError`
  attributes `tool_name` and `category` populated correctly.
- `test_tool_call_attr_constant_guards_interface` — mock tool missing the constant
  attr; assert `ValueError`.

**Old tests to delete / update:**
- Integration test `tests/integration/test_agent_mcp.py` covers graceful degradation
  end-to-end and should be retained.  The new unit tests complement it; nothing is
  deleted.

**Test environment needs:**
- No live MCP server.
- No credentials.
- All smolagents types mocked with plain Python objects.

## Implementation Recommendations

**What the module should own:**
- The `_TOOL_CALL_ATTR` sentinel constant.
- `_sanitize_error()` — pure function, maps `Exception → (safe_message, category)`.
- `_wrap_tool()` — interface guard + monkey-patch + logging.
- `MCPToolError` with `tool_name` and `category` attributes.

**What it should hide:**
- Raw exception text from transport layer.
- The exact attribute name used to call tools (encapsulated in the constant).
- The category classification logic.

**What it should expose:**
- `MCPToolError` (imported by `app.py` for the escape-hatch catch).
- `build_mcp_tools(urls) -> tuple[list, ExitStack]` (unchanged external API).

**How callers should migrate:**
1. Update `_wrap_tool()` and `MCPToolError` in `src/mcp_client.py` in place.
2. Update `app.py` error handler (lines 439–451) to log `exc.tool_name` and
   `exc.category` for observability (no interface change required).
3. Write unit tests before modifying the existing implementation.
4. Add `_TOOL_CALL_ATTR` as a module-level constant with a comment pointing to the
   smolagents `Tool` class documentation.
