# Issue 01 — Deepen `_parse_memory_steps()` into a `SmolagentsMemoryAdapter`

## Problem

`_parse_memory_steps()` in `app.py` (lines 104–177) reverse-engineers the internal structure of smolagents `ActionStep` objects via chains of defensive `hasattr` / `getattr` calls:

```python
timing = getattr(step, "timing", None)
if timing is not None:
    start_time = getattr(timing, "start_time", None)
    end_time   = getattr(timing, "end_time",   None)
    duration   = (end_time - start_time) if (...) else None

token_usage   = getattr(step, "token_usage", None)
input_tokens  = getattr(token_usage, "input_tokens",  None) if token_usage else None
output_tokens = getattr(token_usage, "output_tokens", None) if token_usage else None
total_tokens  = getattr(token_usage, "total_tokens",  None) if token_usage else None
```

smolagents does not publish a versioned schema for `ActionStep`, `StepTiming`, or `TokenUsage`.  These field names were reverse-engineered from library source and will silently start returning `None` if an upstream refactor renames or reorganises the attributes — with no warning and no test to catch it.

The function also shares the same "guess the shape" pattern with `_stream_with_tool_capture()` (lines 253–275), which imports private smolagents symbols (`ChatMessageStreamDelta`, `ToolOutput`, `agglomerate_stream_deltas`, `pull_messages_from_step`) that are not part of the public API.

**Why this is tightly coupled:** `_parse_memory_steps()` is a shallow module — its entire interface is "a list of mystery objects → a list of dicts".  The complexity lies in the structural assumptions about smolagents internals, which are invisible at the call site.  Both the log schema (`ActionStepLog` in `data_logger.py`) and the streaming pipeline (`chat()`) depend on this interpretation being correct, making them transitive dependents of an unversioned
third-party data format.

**Integration risk:**
- `data_logger.py` consumes the output of `_parse_memory_steps()` via `ActionStepLog` Pydantic validation.  If timing or token fields silently degrade to `None`, the Langfuse evaluation pipeline receives incomplete traces without raising any error.
- `_stream_with_tool_capture()` directly imports `smolagents.agents.ToolOutput` — if the event class is renamed or removed, the entire streaming pipeline fails at import time.

## Proposed Interface

Extract a dedicated `SmolagentsMemoryAdapter` module (`src/smolagents_adapter.py`) that owns all structural assumptions about smolagents internals.  The rest of the codebase interacts exclusively with clean, validated output types:

```python
# src/smolagents_adapter.py

from dataclasses import dataclass
from typing import Any

@dataclass
class StepSummary:
    """Parsed representation of one smolagents ActionStep."""
    step_number: int
    model_output: str
    tool_invocations: list[dict]   # [{id, tool_name, arguments, response}]
    observations: str
    duration_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None


def parse_action_steps(
    steps: list[Any],
    tool_responses: dict[str, str] | None = None,
) -> list[StepSummary]:
    """
    Extract ActionStep objects from agent memory and return StepSummary records.

    Logs a WARNING for each expected field that is missing, so degradation is
    visible in logs rather than silent.
    """
    ...


def stream_with_tool_capture(
    agent: Any,
    task: str,
    tool_responses: dict[str, str],
) -> Generator:
    """
    Stream agent.run() events, capturing ToolOutput observations.

    All smolagents private imports are confined here.  If an import fails,
    raises ImportError with an actionable message instead of crashing mid-stream.
    """
    ...
```

**Usage at the call site (app.py):**

```python
from src.smolagents_adapter import parse_action_steps, stream_with_tool_capture

# In chat():
for chunk in stream_with_tool_capture(agent, message, _tool_responses):
    ...

steps = parse_action_steps(_new_steps, _tool_responses)
log_interaction(..., agent_steps=[s.__dict__ for s in steps])
```

What complexity the adapter hides:
- All `hasattr` / `getattr` defensive attribute chains
- Private smolagents symbol imports (`ChatMessageStreamDelta`, `ToolOutput`, etc.)
- The `final_answer` pseudo-tool exclusion rule
- `float(duration)` coercions and integer casts for token counts

## Dependency Strategy

**True external (Mock)** — smolagents is a third-party library that does not publish a
versioned schema for `ActionStep`.

- Production: `SmolagentsMemoryAdapter` imports and reads real `ActionStep` objects.
- Tests: Construct minimal mock step objects (plain Python objects with the expected attributes) and pass them directly to `parse_action_steps()`.  No live agent needed.

```python
# tests/unit/test_smolagents_adapter.py

class _MockTiming:
    start_time = 0.0
    end_time   = 1.5

class _MockTokenUsage:
    input_tokens  = 100
    output_tokens =  50
    total_tokens  = 150

class _MockToolCall:
    id        = "tc-1"
    name      = "search"
    arguments = {"query": "hello"}

class _MockActionStep:
    step_number  = 1
    model_output = "Calling search..."
    tool_calls   = [_MockToolCall()]
    observations = "result text"
    timing       = _MockTiming()
    token_usage  = _MockTokenUsage()
    error        = None

def test_parse_extracts_timing():
    summaries = parse_action_steps([_MockActionStep()])
    assert summaries[0].duration_seconds == 1.5
    assert summaries[0].input_tokens == 100
```

## Testing Strategy

**New boundary tests to write (`tests/unit/test_smolagents_adapter.py`):**
- `test_parse_extracts_timing` — timing fields populate correctly
- `test_parse_extracts_token_usage` — token counts populate correctly
- `test_parse_skips_non_action_steps` — `TaskStep` / `FinalAnswerStep` are filtered
- `test_parse_excludes_final_answer_tool` — pseudo-tool is excluded from invocations
- `test_parse_warns_on_missing_timing` — assert `logger.warning` called when `timing` attribute absent (use `caplog`)
- `test_parse_populates_tool_responses` — `tool_responses` dict matched by `tc_id`
- `test_stream_captures_tool_output` — mock `agent.run(stream=True)` with
  `ToolOutput` events; assert `tool_responses` populated correctly
- `test_stream_handles_missing_private_symbols` — mock `ImportError` on smolagents
  private import; assert `ImportError` raised with informative message

**Old tests to delete:**
- None (there are currently no unit tests for `_parse_memory_steps()` or
  `_stream_with_tool_capture()`).

**Test environment needs:**
- No live agent or MCP server required.
- All smolagents types replaced by in-process mock objects.

## Implementation Recommendations

**What the module should own:**
- All structural assumptions about `ActionStep`, `StepTiming`, `TokenUsage`,
  `ToolCall`, and streaming event types (`ChatMessageStreamDelta`, `ToolOutput`,
  `ActionStep`, `FinalAnswerStep`, `PlanningStep`).
- Defensive attribute extraction with explicit `logger.warning` when fields are
  absent — degradation must be observable.
- The private import block (`from smolagents.agents import ...`, 
  `from smolagents.gradio_ui import ...`), isolated to one file.

**What it should hide:**
- smolagents internal class hierarchy and attribute names.
- `hasattr` / `getattr` chains.
- Type coercions (`float(duration)`, `int(input_tokens)`).

**What it should expose:**
- `parse_action_steps(steps, tool_responses) -> list[StepSummary]`
- `stream_with_tool_capture(agent, task, tool_responses) -> Generator`
- `StepSummary` dataclass (used directly by `app.py` and `data_logger.py`).

**How callers should migrate:**
1. Create `src/smolagents_adapter.py` with `StepSummary`, `parse_action_steps`, `stream_with_tool_capture`.
2. In `app.py`: replace `_parse_memory_steps()` call with `parse_action_steps()` and `_stream_with_tool_capture()` with `stream_with_tool_capture()`.
3. Delete the old private functions from `app.py`.
4. Update `log_interaction()` call to pass `[dataclasses.asdict(s) for s in steps]`.
5. Write boundary tests before deleting the old functions.
