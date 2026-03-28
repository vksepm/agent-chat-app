"""
tests/unit/test_smolagents_adapter.py — Boundary tests for src/smolagents_adapter.

All smolagents types are replaced by in-process mock objects.
No live agent, model, or MCP server is required.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.smolagents_adapter import StepSummary, parse_action_steps, stream_with_tool_capture


# ---------------------------------------------------------------------------
# Mock smolagents step objects
# ---------------------------------------------------------------------------


class _MockTiming:
    start_time = 0.0
    end_time = 1.5


class _MockTokenUsage:
    input_tokens = 100
    output_tokens = 50
    total_tokens = 150


class _MockToolCall:
    id = "tc-1"
    name = "search"
    arguments = {"query": "hello"}


class _MockActionStep:
    step_number = 1
    model_output = "Calling search..."
    tool_calls = [_MockToolCall()]
    observations = "result text"
    timing = _MockTiming()
    token_usage = _MockTokenUsage()
    error = None


class _MockTaskStep:
    """Not an ActionStep — missing tool_calls and step_number."""
    task = "some task"


class _MockFinalAnswerStep:
    """Not an ActionStep — has no tool_calls."""
    final_answer = "done"


class _MockFinalAnswerToolCall:
    id = "tc-fa"
    name = "final_answer"
    arguments = {"answer": "42"}


# ---------------------------------------------------------------------------
# parse_action_steps — happy-path tests
# ---------------------------------------------------------------------------


class TestParseActionSteps:
    def test_parse_extracts_timing(self):
        summaries = parse_action_steps([_MockActionStep()])
        assert len(summaries) == 1
        assert summaries[0].duration_seconds == pytest.approx(1.5)

    def test_parse_extracts_token_usage(self):
        summaries = parse_action_steps([_MockActionStep()])
        assert summaries[0].input_tokens == 100
        assert summaries[0].output_tokens == 50
        assert summaries[0].total_tokens == 150

    def test_parse_extracts_basic_fields(self):
        summaries = parse_action_steps([_MockActionStep()])
        s = summaries[0]
        assert s.step_number == 1
        assert s.model_output == "Calling search..."
        assert s.observations == "result text"
        assert s.error is None

    def test_parse_extracts_tool_invocations(self):
        summaries = parse_action_steps([_MockActionStep()])
        invocations = summaries[0].tool_invocations
        assert len(invocations) == 1
        assert invocations[0]["tool_name"] == "search"
        assert invocations[0]["id"] == "tc-1"
        assert invocations[0]["arguments"] == {"query": "hello"}

    def test_parse_populates_tool_responses(self):
        responses = {"tc-1": "search result here"}
        summaries = parse_action_steps([_MockActionStep()], tool_responses=responses)
        assert summaries[0].tool_invocations[0]["response"] == "search result here"

    def test_parse_empty_tool_responses_gives_empty_response(self):
        summaries = parse_action_steps([_MockActionStep()])
        assert summaries[0].tool_invocations[0]["response"] == ""

    def test_parse_returns_step_summary_instances(self):
        summaries = parse_action_steps([_MockActionStep()])
        assert isinstance(summaries[0], StepSummary)

    def test_parse_multiple_steps(self):
        step1 = _MockActionStep()
        step2 = _MockActionStep()
        step2.step_number = 2
        summaries = parse_action_steps([step1, step2])
        assert len(summaries) == 2
        assert summaries[1].step_number == 2

    def test_parse_empty_list(self):
        assert parse_action_steps([]) == []


# ---------------------------------------------------------------------------
# parse_action_steps — filtering tests
# ---------------------------------------------------------------------------


class TestParseSkipsNonActionSteps:
    def test_parse_skips_task_step(self):
        summaries = parse_action_steps([_MockTaskStep()])
        assert summaries == []

    def test_parse_skips_final_answer_step(self):
        summaries = parse_action_steps([_MockFinalAnswerStep()])
        assert summaries == []

    def test_parse_skips_non_action_steps_mixed(self):
        steps = [_MockTaskStep(), _MockActionStep(), _MockFinalAnswerStep()]
        summaries = parse_action_steps(steps)
        assert len(summaries) == 1
        assert summaries[0].step_number == 1


class TestParseExcludesFinalAnswerTool:
    def test_parse_excludes_final_answer_tool(self):
        step = _MockActionStep()
        step.tool_calls = [_MockFinalAnswerToolCall(), _MockToolCall()]
        summaries = parse_action_steps([step])
        names = [inv["tool_name"] for inv in summaries[0].tool_invocations]
        assert "final_answer" not in names
        assert "search" in names

    def test_parse_step_with_only_final_answer_tool_has_no_invocations(self):
        step = _MockActionStep()
        step.tool_calls = [_MockFinalAnswerToolCall()]
        summaries = parse_action_steps([step])
        assert summaries[0].tool_invocations == []


# ---------------------------------------------------------------------------
# parse_action_steps — degradation / warning tests
# ---------------------------------------------------------------------------


class TestParseWarnsOnMissingFields:
    def test_parse_warns_on_missing_timing(self, caplog):
        step = _MockActionStep()
        step.timing = None
        with caplog.at_level(logging.WARNING, logger="src.smolagents_adapter"):
            summaries = parse_action_steps([step])
        assert summaries[0].duration_seconds is None
        assert any("timing" in r.message for r in caplog.records)

    def test_parse_warns_on_missing_token_usage(self, caplog):
        step = _MockActionStep()
        step.token_usage = None
        with caplog.at_level(logging.WARNING, logger="src.smolagents_adapter"):
            summaries = parse_action_steps([step])
        assert summaries[0].input_tokens is None
        assert summaries[0].output_tokens is None
        assert summaries[0].total_tokens is None
        assert any("token_usage" in r.message for r in caplog.records)

    def test_parse_warns_on_missing_start_time(self, caplog):
        step = _MockActionStep()

        class _PartialTiming:
            start_time = None
            end_time = 1.0

        step.timing = _PartialTiming()
        with caplog.at_level(logging.WARNING, logger="src.smolagents_adapter"):
            summaries = parse_action_steps([step])
        assert summaries[0].duration_seconds is None
        assert any("start_time" in r.message or "end_time" in r.message for r in caplog.records)

    def test_parse_string_arguments_coerced(self):
        step = _MockActionStep()
        tc = _MockToolCall()
        tc.arguments = "raw string args"
        step.tool_calls = [tc]
        summaries = parse_action_steps([step])
        assert summaries[0].tool_invocations[0]["arguments"] == "raw string args"

    def test_parse_error_field_captured(self):
        step = _MockActionStep()
        step.error = RuntimeError("tool failed")
        summaries = parse_action_steps([step])
        assert "tool failed" in summaries[0].error


# ---------------------------------------------------------------------------
# stream_with_tool_capture tests
# ---------------------------------------------------------------------------


class TestStreamWithToolCapture:
    def _make_mock_imports(self):
        """Return mock classes for the private smolagents symbols."""

        class _MockActionStep:
            pass

        class _MockPlanningStep:
            pass

        class _MockFinalAnswerStep:
            pass

        class _MockStreamDelta:
            pass

        class _MockToolOutput:
            def __init__(self, tc_id, observation):
                self.id = tc_id
                self.observation = observation

        return (
            _MockActionStep,
            _MockPlanningStep,
            _MockFinalAnswerStep,
            _MockStreamDelta,
            _MockToolOutput,
        )

    def test_stream_captures_tool_output(self):
        (
            MockActionStep,
            MockPlanningStep,
            MockFinalAnswerStep,
            MockStreamDelta,
            MockToolOutput,
        ) = self._make_mock_imports()

        tool_output_event = MockToolOutput("tc-99", "observation text")
        action_event = MockActionStep()

        mock_agent = MagicMock()
        mock_agent.run.return_value = iter([tool_output_event, action_event])
        mock_agent.stream_outputs = False

        mock_pull = MagicMock(return_value=["msg1"])
        mock_agglomerate = MagicMock()

        tool_responses: dict = {}

        with patch.dict(
            "sys.modules",
            {
                "smolagents.agents": MagicMock(
                    ChatMessageStreamDelta=MockStreamDelta,
                    ToolOutput=MockToolOutput,
                ),
                "smolagents.gradio_ui": MagicMock(
                    agglomerate_stream_deltas=mock_agglomerate,
                    pull_messages_from_step=mock_pull,
                ),
                "smolagents.memory": MagicMock(
                    ActionStep=MockActionStep,
                    FinalAnswerStep=MockFinalAnswerStep,
                    PlanningStep=MockPlanningStep,
                ),
            },
        ):
            # Re-import to pick up patched modules
            import importlib
            import src.smolagents_adapter as adapter_mod
            importlib.reload(adapter_mod)

            list(adapter_mod.stream_with_tool_capture(mock_agent, "task", tool_responses))

        assert tool_responses.get("tc-99") == "observation text"

    def test_stream_handles_missing_private_symbols(self):
        """If smolagents private symbols are missing, ImportError with message."""
        with patch.dict("sys.modules", {"smolagents.agents": None}):
            import importlib
            import src.smolagents_adapter as adapter_mod
            importlib.reload(adapter_mod)

            mock_agent = MagicMock()
            tool_responses: dict = {}

            with pytest.raises(ImportError, match="smolagents_adapter"):
                list(adapter_mod.stream_with_tool_capture(mock_agent, "task", tool_responses))
