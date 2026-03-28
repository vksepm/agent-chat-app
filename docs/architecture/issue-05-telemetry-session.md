# Issue 05 — Deepen Langfuse Context Propagation into a `TelemetrySession`

## Problem

The telemetry setup in this codebase is split across three files with two distinct
fragility points.

### Fragility 1 — `propagate_attributes().__enter__()` without `__exit__()`

In `app.py` (lines 307–324), the Langfuse session context manager is entered but
never exited:

```python
if langfuse is not None:
    try:
        from langfuse import propagate_attributes
        propagate_attributes(
            session_id=session_id,
            metadata={"app_version": cfg.app_version},
        ).__enter__()        # ← __exit__() intentionally omitted
    except Exception as exc:
        logger.warning("Could not propagate session attributes: %s", exc)
```

The comment explains this is a workaround for an OpenTelemetry context-var detach
bug: Gradio resumes the streaming generator in a different async task context,
causing `ContextVar.reset()` in `__exit__()` to raise `ValueError`.  The workaround
relies on Gradio's `copy_context()` isolation per request to clean up the context
variable automatically.

This is correct as of the current smolagents + Gradio versions, but it:
- Violates the context manager protocol — a future developer may "fix" this by
  adding `__exit__()`, silently reintroducing the upstream bug.
- Has no upstream issue link embedded in the code (the comment says "upstream issue"
  but contains no URL).
- Is untested — there is no test that verifies the telemetry path degrades safely
  when `propagate_attributes` raises.

### Fragility 2 — Side-effectful env-var mutation in `bootstrap_telemetry()`

`src/telemetry.py` (lines 54–56) sets three environment variables as a side effect
of startup:

```python
os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
os.environ["LANGFUSE_BASE_URL"]   = cfg.langfuse_base_url
```

This is done so the OTEL exporter can pick them up, but it makes telemetry
initialization order-sensitive and mutates the process environment globally — which
affects any other code in the same process that reads those env vars after startup.
It also makes `bootstrap_telemetry()` hard to call in tests without polluting the
test process environment.

### Fragility 3 — ExitStack leak in `app.py`

`_mcp_stack` (line 38 in app.py) is created at module load but never explicitly
closed.  The Gradio app runs indefinitely; there is no `atexit` hook or
`demo.close()` handler that calls `_mcp_stack.close()`.  HTTP connections to MCP
servers may not close cleanly on shutdown.

While strictly an `app.py` / `mcp_client.py` concern, it is architecturally
symmetric with the telemetry lifecycle problem: both involve resources opened at
startup with no defined shutdown path.

**Why this is tightly coupled:**
`app.py` manages the Langfuse client reference (`langfuse`), calls
`bootstrap_telemetry()`, and manually manages the context propagation — three
responsibilities mixed into the startup and per-request paths.  The workaround
comment is necessary but insufficient: it describes *what* is being done but not
*why this specific approach is safe*, and there is no automated verification.

**Integration risk:**
- If `propagate_attributes` is not available (older Langfuse version), the
  `ImportError` is caught but the `from langfuse import propagate_attributes` import
  is attempted on every request — unnecessary repeated import overhead.
- If Gradio changes its `copy_context()` behaviour, the context-var leak becomes
  real, and there is no test to detect it.
- `_mcp_stack` connections may not close cleanly, potentially exhausting the MCP
  server's connection pool in long-running deployments.

## Proposed Interface

Extract a `TelemetrySession` context object that encapsulates the Langfuse
per-request attribution and the known `__exit__` workaround:

```python
# src/telemetry.py  (addition)

class TelemetrySession:
    """
    Scopes Langfuse session attributes to one Gradio request turn.

    Encapsulates the known OpenTelemetry context-var detach workaround:
    propagate_attributes().__enter__() is called without __exit__() because
    Gradio's copy_context() isolation per request cleans up context variables
    automatically when the request task ends.  See:
    https://github.com/Arize-ai/openinference/issues/<issue-number>

    Usage
    -----
    session = TelemetrySession(langfuse_client, session_id, app_version)
    session.attach()        # call once at start of chat() turn

    # No teardown required — Gradio request context is automatically cleaned up.
    """

    def __init__(
        self,
        langfuse,          # Langfuse client or None (degraded mode)
        session_id: str,
        app_version: str,
    ) -> None:
        self._langfuse    = langfuse
        self._session_id  = session_id
        self._app_version = app_version

    def attach(self) -> None:
        """
        Propagate session attributes to all child OTEL spans for this request.

        No-op if Langfuse client is None (degraded mode).
        Logs a warning and continues if propagation fails.
        """
        if self._langfuse is None:
            return
        try:
            from langfuse import propagate_attributes
            propagate_attributes(
                session_id=self._session_id,
                metadata={"app_version": self._app_version},
            ).__enter__()
            # __exit__() intentionally omitted.
            # See class docstring for rationale.
        except Exception as exc:
            logger.warning("TelemetrySession.attach() failed: %s", exc)
```

Also add an `atexit`-compatible shutdown helper to `bootstrap_telemetry()`:

```python
def shutdown_telemetry(langfuse) -> None:
    """
    Flush pending spans and close the Langfuse OTEL exporter.

    Call from an atexit handler or demo.close() callback.
    Safe to call when langfuse is None.
    """
    if langfuse is None:
        return
    try:
        langfuse.flush()
    except Exception as exc:
        logger.warning("Telemetry flush failed: %s", exc)
```

**Usage at the call site (app.py):**

```python
from src.telemetry import TelemetrySession, shutdown_telemetry

# Startup — unchanged:
langfuse = bootstrap_telemetry(cfg)
atexit.register(shutdown_telemetry, langfuse)

# Per-request — in chat():
TelemetrySession(langfuse, session_id, cfg.app_version).attach()
```

Also add an `atexit` handler for `_mcp_stack`:

```python
import atexit
atexit.register(_mcp_stack.close)
```

What `TelemetrySession` hides:
- The deferred import of `propagate_attributes` (import once, cache if needed).
- The `__enter__()` without `__exit__()` pattern and its rationale.
- The `langfuse is None` degraded-mode check.
- The try/except guard for propagation failures.

## Dependency Strategy

**Remote but owned (Ports & Adapters)** — Langfuse is your own tracing
infrastructure.

- Production: `TelemetrySession.attach()` calls the real `propagate_attributes`.
- Tests: Pass `langfuse=None` to verify no-op degraded mode.  Use
  `unittest.mock.patch("langfuse.propagate_attributes")` to verify the happy path
  without live Langfuse credentials.

## Testing Strategy

**New unit tests to write (`tests/unit/test_telemetry.py`):**

- `test_attach_noop_when_langfuse_is_none` — `TelemetrySession(None, ...).attach()`
  does not raise and does not call `propagate_attributes`.
- `test_attach_calls_propagate_enter` — mock `langfuse` + `propagate_attributes`;
  assert `__enter__()` called exactly once with correct `session_id`.
- `test_attach_swallows_propagation_error` — mock `propagate_attributes` to raise;
  assert no exception propagates from `attach()`, and `logger.warning` called.
- `test_attach_skips_exit` — verify `__exit__()` is NOT called on the context
  manager returned by `propagate_attributes` (the workaround must remain in place).
- `test_bootstrap_returns_none_on_auth_failure` — mock `Langfuse.auth_check()`
  to return `False`; assert `bootstrap_telemetry()` returns `None`.
- `test_bootstrap_returns_none_on_import_error` — mock `ImportError` on
  `from langfuse import Langfuse`; assert returns `None`.
- `test_shutdown_calls_flush` — mock Langfuse client; assert `flush()` called by
  `shutdown_telemetry(langfuse)`.
- `test_shutdown_noop_when_none` — `shutdown_telemetry(None)` does not raise.

**Old tests to delete:**
- None (there are currently no tests for `telemetry.py`).

**Test environment needs:**
- No live Langfuse credentials.
- `unittest.mock.patch` for `langfuse.propagate_attributes` and `Langfuse` class.
- `caplog` for asserting `logger.warning` calls.

## Implementation Recommendations

**What `TelemetrySession` should own:**
- The `__enter__()` without `__exit__()` workaround — fully documented with upstream
  issue URL once filed.
- The deferred import of `propagate_attributes` (import at class level or cache in
  module after first successful import).
- Degraded-mode handling (no-op when `langfuse is None`).

**What `bootstrap_telemetry()` should own:**
- Env-var mutation (still required for OTEL exporter, but annotated clearly).
- Langfuse client instantiation and auth check.
- `SmolagentsInstrumentor().instrument()` call.

**What it should expose:**
- `TelemetrySession(langfuse, session_id, app_version).attach()` — per-request use.
- `bootstrap_telemetry(cfg) -> Optional[Langfuse]` — unchanged.
- `shutdown_telemetry(langfuse) -> None` — new, for `atexit` registration.

**How callers should migrate:**
1. Add `TelemetrySession` and `shutdown_telemetry` to `src/telemetry.py`.
2. In `app.py`: replace the inline `propagate_attributes().__enter__()` block with
   `TelemetrySession(langfuse, session_id, cfg.app_version).attach()`.
3. Add `atexit.register(shutdown_telemetry, langfuse)` after `bootstrap_telemetry()`.
4. Add `atexit.register(_mcp_stack.close)` after `build_mcp_tools()`.
5. File an upstream issue at https://github.com/Arize-ai/openinference for the
   context-var detach bug and link it in the `TelemetrySession` docstring.
6. Write unit tests before removing the inline block from `app.py`.
