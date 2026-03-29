"""
src/tool_panel_manager.py — Manages the list of gr.ChatMessage objects for one
agent turn.

Centralises all message accumulation, deduplication, and supplementation logic
that was previously spread across ``_has_tool_title``, ``_supplement_tool_panels``,
``_extract_assistant_response``, and the in-line accumulation loop in ``chat()``.

The smolagents title format string is declared once as a module-level constant so
there is a single place to update if the upstream format ever changes.

Public API
----------
* ``SMOLAGENTS_TOOL_TITLE_PREFIX`` — canonical title prefix for tool-call panels
* ``ToolPanelManager``              — accumulate / deduplicate / supplement / query
"""

import logging
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# Single source of truth for the smolagents tool-panel title contract.
# smolagents emits: "🛠️ Used tool <name>" for every tool-call panel.
# If the upstream format changes, update this constant and only this constant.
SMOLAGENTS_TOOL_TITLE_PREFIX = "🛠️ Used tool "


class ToolPanelManager:
    """Manages the list of ``gr.ChatMessage`` objects produced during one agent turn.

    Responsibilities
    ----------------
    - Accept streaming chunks from ``stream_with_tool_capture`` via ``ingest()``.
    - Deduplicate tool-call panel updates (pending → done transitions) by title.
    - Track which tool names already have a panel.
    - Supplement missing panels for parallel tool calls not emitted by smolagents
      via ``supplement()``.
    - Extract the final plain-text assistant answer via ``final_answer()``.
    - Append a suffix (e.g. a truncation notice) to the last text message via
      ``append_to_last_text()``.

    One instance is created per chat turn and discarded afterwards.
    """

    def __init__(self) -> None:
        self._messages: list[gr.ChatMessage] = []

    # ------------------------------------------------------------------
    # Static helpers (exported so callers share the same contract)
    # ------------------------------------------------------------------

    @staticmethod
    def is_tool_message(msg: Any) -> bool:
        """Return True if *msg* is a tool-call panel (has a non-empty metadata title).

        Parameters
        ----------
        msg:
            Any object — typically a ``gr.ChatMessage``.
        """
        meta = getattr(msg, "metadata", None)
        return isinstance(meta, dict) and bool(meta.get("title"))

    @staticmethod
    def tool_title(name: str) -> str:
        """Return the canonical title string for a tool-call panel.

        Parameters
        ----------
        name:
            Tool name, e.g. ``"search"``.
        """
        return f"{SMOLAGENTS_TOOL_TITLE_PREFIX}{name}"

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def ingest(self, chunk: Any) -> None:
        """Accept one chunk from the streaming generator.

        Non-``gr.ChatMessage`` values (e.g. plain ``str`` streaming deltas) and
        chunks with ``None`` content are silently ignored.

        - **Tool-call chunks** (chunks whose ``metadata`` contains a ``title``)
          update the matching existing panel in-place, or are appended if this is
          the first chunk for that panel title.
        - **Text chunks** replace the last non-tool message (streaming update),
          or are appended if no text message exists yet.

        Parameters
        ----------
        chunk:
            A value yielded by ``stream_with_tool_capture``.
        """
        if not hasattr(chunk, "content") or chunk.content is None:
            return

        if self.is_tool_message(chunk):
            title: str = chunk.metadata["title"]
            idx = next(
                (
                    i
                    for i, m in enumerate(self._messages)
                    if self.is_tool_message(m) and m.metadata["title"] == title
                ),
                None,
            )
            if idx is not None:
                self._messages[idx] = chunk
            else:
                self._messages.append(chunk)
        else:
            last_text_idx = next(
                (
                    i
                    for i in range(len(self._messages) - 1, -1, -1)
                    if not self.is_tool_message(self._messages[i])
                ),
                None,
            )
            if last_text_idx is not None:
                self._messages[last_text_idx] = chunk
            else:
                self._messages.append(chunk)

    def supplement(self, steps: list) -> None:
        """Add panels for any tool calls in *steps* not yet present in messages.

        smolagents only emits a panel for ``tool_calls[0]`` per step.  This
        method adds panels for any additional parallel tool calls so every
        invocation is visible in the Gradio chatbot.

        The ``final_answer`` pseudo-tool is always excluded.

        Parameters
        ----------
        steps:
            A list of smolagents memory step objects (``ActionStep`` etc.).
            Injected rather than imported so the method can be tested with
            plain mock objects.
        """
        present_tools: set[str] = set()
        for msg in self._messages:
            if self.is_tool_message(msg):
                title: str = msg.metadata.get("title", "")
                if title.startswith(SMOLAGENTS_TOOL_TITLE_PREFIX):
                    tool_name = title[len(SMOLAGENTS_TOOL_TITLE_PREFIX):]
                    present_tools.add(tool_name)

        for step in steps:
            for tc in getattr(step, "tool_calls", None) or []:
                name = getattr(tc, "name", "") or ""
                if name in ("final_answer", "") or name in present_tools:
                    continue
                args = getattr(tc, "arguments", {})
                self._messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=str(args),
                        metadata={"title": self.tool_title(name), "status": "done"},
                    )
                )
                present_tools.add(name)
                logger.debug(
                    "tool_panel_manager: supplemented UI panel for tool '%s'", name
                )

    def append_to_last_text(self, suffix: str) -> None:
        """Append *suffix* to the content of the last non-tool message.

        If no text message exists yet (edge case), a new message is appended.

        Parameters
        ----------
        suffix:
            Text to append, e.g. a truncation notice.
        """
        last_text_idx = next(
            (
                i
                for i in range(len(self._messages) - 1, -1, -1)
                if not self.is_tool_message(self._messages[i])
            ),
            None,
        )
        if last_text_idx is not None:
            last_content = getattr(self._messages[last_text_idx], "content", "") or ""
            self._messages[last_text_idx] = gr.ChatMessage(
                role="assistant", content=last_content + suffix
            )
        else:
            self._messages.append(gr.ChatMessage(role="assistant", content=suffix))

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def final_answer(self) -> str:
        """Return the last plain-text assistant message content.

        Skips tool-call panels.  Returns an empty string when no text message
        has been ingested yet.
        """
        for msg in reversed(self._messages):
            if not self.is_tool_message(msg):
                content = getattr(msg, "content", "") or ""
                if isinstance(content, str):
                    return content
        return ""

    @property
    def messages(self) -> list[gr.ChatMessage]:
        """Current message list as a snapshot copy."""
        return list(self._messages)
