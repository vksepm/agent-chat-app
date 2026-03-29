"""
tests/unit/test_tool_panel_manager.py — Boundary tests for src/tool_panel_manager.

Gradio is a declared dependency and is used directly.
No live agent, Gradio server, or network access required.
"""

import gradio as gr
import pytest

from src.tool_panel_manager import SMOLAGENTS_TOOL_TITLE_PREFIX, ToolPanelManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_msg(name: str, content: str = "args", status: str = "done") -> gr.ChatMessage:
    """Build a tool-call panel ChatMessage."""
    return gr.ChatMessage(
        role="assistant",
        content=content,
        metadata={"title": ToolPanelManager.tool_title(name), "status": status},
    )


def _text_msg(content: str) -> gr.ChatMessage:
    """Build a plain-text assistant ChatMessage."""
    return gr.ChatMessage(role="assistant", content=content)


# ---------------------------------------------------------------------------
# ingest() — tool-call panel behaviour
# ---------------------------------------------------------------------------


class TestIngestToolChunks:
    def test_ingest_appends_first_tool_chunk(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        assert len(mgr.messages) == 1
        assert mgr.is_tool_message(mgr.messages[0])

    def test_ingest_updates_existing_tool_panel(self):
        mgr = ToolPanelManager()
        pending = gr.ChatMessage(
            role="assistant",
            content="...",
            metadata={"title": ToolPanelManager.tool_title("search"), "status": "pending"},
        )
        done = _tool_msg("search", content="result", status="done")

        mgr.ingest(pending)
        mgr.ingest(done)

        assert len(mgr.messages) == 1
        assert mgr.messages[0].metadata["status"] == "done"
        assert mgr.messages[0].content == "result"

    def test_ingest_appends_new_tool_with_different_name(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        mgr.ingest(_tool_msg("calculator"))
        assert len(mgr.messages) == 2
        titles = {m.metadata["title"] for m in mgr.messages}
        assert ToolPanelManager.tool_title("search") in titles
        assert ToolPanelManager.tool_title("calculator") in titles


# ---------------------------------------------------------------------------
# ingest() — text chunk behaviour
# ---------------------------------------------------------------------------


class TestIngestTextChunks:
    def test_ingest_appends_text_after_tool_panels(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        mgr.ingest(_text_msg("Here is the answer."))
        assert len(mgr.messages) == 2
        assert mgr.messages[-1].content == "Here is the answer."

    def test_ingest_streaming_text_replaces_last_text(self):
        mgr = ToolPanelManager()
        mgr.ingest(_text_msg("Partial answer…"))
        mgr.ingest(_text_msg("Full answer."))
        assert len(mgr.messages) == 1
        assert mgr.messages[0].content == "Full answer."

    def test_ingest_streaming_text_replaces_last_text_not_tool_panel(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        mgr.ingest(_text_msg("First text."))
        mgr.ingest(_text_msg("Updated text."))
        assert len(mgr.messages) == 2
        assert mgr.messages[0].metadata["title"] == ToolPanelManager.tool_title("search")
        assert mgr.messages[1].content == "Updated text."

    def test_ingest_skips_chunk_without_content_attr(self):
        mgr = ToolPanelManager()
        mgr.ingest("plain string delta")
        assert mgr.messages == []

    def test_ingest_skips_chunk_with_none_content(self):
        mgr = ToolPanelManager()
        msg = gr.ChatMessage(role="assistant", content=None)
        mgr.ingest(msg)
        assert mgr.messages == []


# ---------------------------------------------------------------------------
# supplement()
# ---------------------------------------------------------------------------


class _MockToolCall:
    def __init__(self, name: str, tc_id: str = "tc-1", arguments=None):
        self.name = name
        self.id = tc_id
        self.arguments = arguments or {"q": "test"}


class _MockStep:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.step_number = 1


class TestSupplement:
    def test_supplement_adds_missing_panel(self):
        mgr = ToolPanelManager()
        step = _MockStep([_MockToolCall("search")])
        mgr.supplement([step])
        assert len(mgr.messages) == 1
        assert mgr.messages[0].metadata["title"] == ToolPanelManager.tool_title("search")

    def test_supplement_skips_already_present_tool(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        step = _MockStep([_MockToolCall("search")])
        mgr.supplement([step])
        assert len(mgr.messages) == 1

    def test_supplement_skips_final_answer_tool(self):
        mgr = ToolPanelManager()
        step = _MockStep([_MockToolCall("final_answer")])
        mgr.supplement([step])
        assert mgr.messages == []

    def test_supplement_adds_only_missing_parallel_tool(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        step = _MockStep([_MockToolCall("search"), _MockToolCall("calculator")])
        mgr.supplement([step])
        assert len(mgr.messages) == 2
        names = {m.metadata["title"] for m in mgr.messages}
        assert ToolPanelManager.tool_title("calculator") in names

    def test_supplement_empty_steps(self):
        mgr = ToolPanelManager()
        mgr.supplement([])
        assert mgr.messages == []


# ---------------------------------------------------------------------------
# final_answer()
# ---------------------------------------------------------------------------


class TestFinalAnswer:
    def test_final_answer_returns_last_text_content(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        mgr.ingest(_text_msg("The answer is 42."))
        assert mgr.final_answer() == "The answer is 42."

    def test_final_answer_returns_empty_when_no_text(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        assert mgr.final_answer() == ""

    def test_final_answer_empty_manager(self):
        mgr = ToolPanelManager()
        assert mgr.final_answer() == ""


# ---------------------------------------------------------------------------
# append_to_last_text()
# ---------------------------------------------------------------------------


class TestAppendToLastText:
    def test_append_to_last_text_updates_content(self):
        mgr = ToolPanelManager()
        mgr.ingest(_text_msg("Base answer."))
        mgr.append_to_last_text(" (truncated)")
        assert mgr.messages[0].content == "Base answer. (truncated)"

    def test_append_to_last_text_skips_tool_panels(self):
        mgr = ToolPanelManager()
        mgr.ingest(_tool_msg("search"))
        mgr.ingest(_text_msg("Answer."))
        mgr.append_to_last_text(" Extra.")
        assert mgr.messages[1].content == "Answer. Extra."
        assert mgr.is_tool_message(mgr.messages[0])

    def test_append_to_last_text_creates_message_when_empty(self):
        mgr = ToolPanelManager()
        mgr.append_to_last_text("Note.")
        assert len(mgr.messages) == 1
        assert mgr.messages[0].content == "Note."


# ---------------------------------------------------------------------------
# Static helpers / constant contract
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    def test_is_tool_message_true_for_tool_panel(self):
        msg = _tool_msg("search")
        assert ToolPanelManager.is_tool_message(msg) is True

    def test_is_tool_message_false_for_text_message(self):
        msg = _text_msg("hello")
        assert ToolPanelManager.is_tool_message(msg) is False

    def test_is_tool_message_false_for_plain_string(self):
        assert ToolPanelManager.is_tool_message("string") is False

    def test_tool_title_constant_used_for_both_emit_and_detect(self):
        """Round-trip: panel built with tool_title() is detected by is_tool_message()."""
        title = ToolPanelManager.tool_title("my_tool")
        msg = gr.ChatMessage(
            role="assistant", content="x", metadata={"title": title}
        )
        assert ToolPanelManager.is_tool_message(msg)
        assert title.startswith(SMOLAGENTS_TOOL_TITLE_PREFIX)

    def test_tool_title_contains_prefix_and_name(self):
        title = ToolPanelManager.tool_title("calculator")
        assert title == f"{SMOLAGENTS_TOOL_TITLE_PREFIX}calculator"

    def test_messages_property_returns_copy(self):
        mgr = ToolPanelManager()
        mgr.ingest(_text_msg("hello"))
        snapshot = mgr.messages
        snapshot.clear()
        assert len(mgr.messages) == 1
