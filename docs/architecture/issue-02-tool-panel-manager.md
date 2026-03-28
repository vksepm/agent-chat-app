# Issue 02 — Deepen Message Accumulation + Deduplication into a `ToolPanelManager`

## Problem

The `chat()` function in `app.py` (lines 342–425) contains a message accumulation
loop that tracks, deduplicates, and supplements tool-call UI panels.  The logic is
spread across four shallow functions that all share an implicit contract about the
smolagents title format:

| Function | Location | Role |
|---|---|---|
| `_has_tool_title(msg)` | app.py:88–91 | Detects tool-call `ChatMessage` by metadata |
| `_supplement_tool_panels(msgs, steps)` | app.py:180–225 | Adds missing panels for parallel tool calls |
| `_extract_assistant_response(msgs)` | app.py:94–101 | Finds final plain-text answer |
| Message loop in `chat()` | app.py:342–373 | Deduplicates streaming updates in-place |

The shared contract — smolagents' title format `"🛠️ Used tool <name>"` — is a
**hardcoded string buried in two separate functions** (lines 202–203 and 219) with no
symbolic constant:

```python
# _supplement_tool_panels(), line 202
if "Used tool" in title:
    tool_name = title.split("Used tool", 1)[-1].strip()
    present_tools.add(tool_name)

# _supplement_tool_panels(), line 219  — construction mirrors the pattern
metadata={"title": f"🛠️ Used tool {name}", "status": "done"},
```

If smolagents changes the title format (it is not part of the public API), the
deduplication silently breaks: panels are appended instead of updated, and
`_supplement_tool_panels()` no longer recognises existing panels, causing duplicate
tool entries in the UI.

**Why this is tightly coupled:**
The accumulation loop in `chat()` and `_supplement_tool_panels()` must both agree on
what constitutes "the same tool call panel".  Today that agreement is replicated
string matching.  The loop is also responsible for deciding the final Gradio history
list shape, which is both streamed incrementally and emitted as a final state —
mixing streaming concerns with deduplication logic in one place.

**Integration risk:**
- No unit tests exist for any of these four functions.  Bugs in panel deduplication
  only surface at runtime via UI glitches (duplicate expandable tool sections).
- `_supplement_tool_panels()` also depends on `agent.memory.steps`, introducing a
  second smolagents internal dependency (see Issue 01) at the UI layer.

## Proposed Interface

Extract a `ToolPanelManager` class that owns the full message-list lifecycle for one
chat turn — accumulation, deduplication, supplementation, and final answer extraction:

```python
# src/tool_panel_manager.py

import gradio as gr

# Single source of truth for the smolagents title contract.
SMOLAGENTS_TOOL_TITLE_PREFIX = "🛠️ Used tool "

class ToolPanelManager:
    """
    Manages the list of gr.ChatMessage objects produced during one agent turn.

    Responsibilities:
    - Deduplicate streaming tool-call panel updates (pending → done transitions).
    - Track which tool names already have a panel.
    - Supplement missing panels for parallel tool calls not emitted by smolagents.
    - Extract the final plain-text assistant answer.
    """

    def __init__(self) -> None:
        self._messages: list[gr.ChatMessage] = []

    def ingest(self, chunk: gr.ChatMessage) -> None:
        """
        Accept one chunk from the streaming generator.

        - Tool-call chunks (chunks with metadata title) update the matching
          existing panel in-place, or are appended if new.
        - Text chunks replace the last non-tool message (streaming update),
          or are appended if this is the first text chunk.
        """
        ...

    def supplement(self, steps: list) -> None:
        """
        Add panels for any tool calls in *steps* not yet present in messages.

        Uses SMOLAGENTS_TOOL_TITLE_PREFIX to check existing panel titles.
        smolagents only emits tool_calls[0] per step; this makes parallel
        calls visible in the UI.
        """
        ...

    @staticmethod
    def tool_title(name: str) -> str:
        """Return the canonical title string for a tool-call panel."""
        return f"{SMOLAGENTS_TOOL_TITLE_PREFIX}{name}"

    @staticmethod
    def is_tool_message(msg: gr.ChatMessage) -> bool:
        """Return True if *msg* is a tool-call panel (has a metadata title)."""
        meta = getattr(msg, "metadata", None)
        return isinstance(meta, dict) and bool(meta.get("title"))

    def final_answer(self) -> str:
        """Return the last plain-text assistant message content."""
        ...

    @property
    def messages(self) -> list[gr.ChatMessage]:
        """Current message list (snapshot)."""
        return list(self._messages)
```

**Usage at the call site (app.py):**

```python
from src.tool_panel_manager import ToolPanelManager

panel_mgr = ToolPanelManager()

for chunk in stream_with_tool_capture(agent, message, _tool_responses):
    panel_mgr.ingest(chunk)
    yield history + [{"role": "user", "content": message}] + panel_mgr.messages, ...

panel_mgr.supplement(_new_steps)
yield history + [{"role": "user", "content": message}] + panel_mgr.messages, ...

log_interaction(..., final_answer=panel_mgr.final_answer(), ...)
```

What complexity `ToolPanelManager` hides:
- Title string format and parsing.
- In-place list index lookups (the nested `next(i for i, m in enumerate(...))` loops).
- The asymmetry between "update existing panel" and "update streaming text chunk".
- `_supplement_tool_panels()` dependency on smolagents memory steps.

## Dependency Strategy

**True external (Mock)** — depends on `gr.ChatMessage` (Gradio) and smolagents
title format string.

- `SMOLAGENTS_TOOL_TITLE_PREFIX` is a module-level constant.  If upstream changes the
  format, there is exactly one place to update.
- Tests construct `gr.ChatMessage` objects directly (Gradio is a declared dependency
  with no network requirements).
- Smolagents memory step objects in `supplement()` are injected as a list and can be
  mocked with the same mock types defined in Issue 01.

## Testing Strategy

**New boundary tests to write (`tests/unit/test_tool_panel_manager.py`):**

- `test_ingest_appends_first_tool_chunk` — first tool-call chunk appended to empty
  message list.
- `test_ingest_updates_existing_tool_panel` — second chunk with same title replaces
  the first (pending → done transition).
- `test_ingest_appends_new_tool_with_different_name` — two distinct tool names produce
  two separate panels.
- `test_ingest_streaming_text_replaces_last_text` — second text chunk replaces the
  first (streaming update behaviour).
- `test_ingest_appends_text_after_tool_panels` — text chunk appended after tool
  panels.
- `test_supplement_adds_missing_panel` — `supplement()` adds a panel for a tool name
  not present in messages.
- `test_supplement_skips_already_present_tool` — `supplement()` does not duplicate an
  existing panel.
- `test_supplement_skips_final_answer_tool` — `final_answer` pseudo-tool never gets
  a panel.
- `test_final_answer_returns_last_text_content` — `final_answer()` skips tool-call
  panels and returns the plain-text content.
- `test_tool_title_constant_used_for_both_emit_and_detect` — round-trip: a panel
  built with `tool_title(name)` is correctly detected by `is_tool_message()`.

**Old tests to delete:**
- None (no existing tests for the functions being replaced).

**Test environment needs:**
- `gradio` must be installed (it is in `requirements.txt`).
- No live agent, Gradio server, or network access required.

## Implementation Recommendations

**What the module should own:**
- The canonical smolagents title format string as a module-level constant.
- All list manipulation logic for accumulating, deduplicating, and supplementing
  `gr.ChatMessage` lists.
- The `is_tool_message` / `tool_title` helpers (currently duplicated between
  `_has_tool_title` and `_supplement_tool_panels`).

**What it should hide:**
- Index arithmetic for in-place updates (`next(i for i, m in enumerate(...))`).
- The asymmetric handling of tool chunks vs. text chunks.
- The `present_tools` set management.

**What it should expose:**
- `ToolPanelManager` class with `ingest()`, `supplement()`, `final_answer()`,
  `messages` property.
- `SMOLAGENTS_TOOL_TITLE_PREFIX` constant (exported so smolagents_adapter.py and
  any future callers can use the same value).
- `is_tool_message()` and `tool_title()` as static methods (or module-level helpers).

**How callers should migrate:**
1. Create `src/tool_panel_manager.py` with `ToolPanelManager` and
   `SMOLAGENTS_TOOL_TITLE_PREFIX`.
2. In `app.py`: replace the `new_messages` list + accumulation loop + calls to
   `_has_tool_title`, `_supplement_tool_panels`, `_extract_assistant_response` with
   `ToolPanelManager`.
3. Delete `_has_tool_title()`, `_supplement_tool_panels()`, and
   `_extract_assistant_response()` from `app.py`.
4. Update `_stream_with_tool_capture()` (or the new `stream_with_tool_capture()` from
   Issue 01) to yield `gr.ChatMessage` objects consistently.
5. Write boundary tests before deleting the old functions.
