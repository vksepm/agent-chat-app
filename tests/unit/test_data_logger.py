"""
tests/unit/test_data_logger.py — Unit tests for src/data_logger.

Tests cover:
- Pydantic schema validation (valid and invalid inputs)
- JSONL file write round-trip
- log_interaction() enqueueing and worker flush
- Timer utility
"""

import json
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data_logger import (
    ActionStepLog,
    ConversationTurnLog,
    ModelInfo,
    ToolInvocation,
    _write_jsonl,
    init_logger,
    log_interaction,
    _log_queue,
)
from src.timer import Timer


# ---------------------------------------------------------------------------
# Schema tests — ToolInvocation
# ---------------------------------------------------------------------------


class TestToolInvocation:
    def test_valid_dict_arguments(self):
        t = ToolInvocation(id="tc-1", tool_name="search", arguments={"query": "python"})
        assert t.tool_name == "search"
        assert t.arguments == {"query": "python"}
        assert t.id == "tc-1"

    def test_string_arguments_allowed(self):
        t = ToolInvocation(tool_name="noop", arguments="raw args")
        assert t.arguments == "raw args"

    def test_empty_id_allowed(self):
        t = ToolInvocation(tool_name="x", arguments={})
        assert t.id == ""

    def test_response_field_defaults_empty(self):
        t = ToolInvocation(tool_name="search", arguments={})
        assert t.response == ""

    def test_response_field_stored(self):
        t = ToolInvocation(tool_name="weather", arguments={}, response="9.6°C in Paris")
        assert t.response == "9.6°C in Paris"

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ToolInvocation(tool_name="x", arguments={}, unknown="y")


# ---------------------------------------------------------------------------
# Schema tests — ModelInfo
# ---------------------------------------------------------------------------


class TestModelInfo:
    def test_valid(self):
        m = ModelInfo(model_id="openai/gpt-4o", app_version="1.2.3")
        assert m.model_id == "openai/gpt-4o"

    def test_default_app_version(self):
        m = ModelInfo(model_id="openai/gpt-4o")
        assert m.app_version == "dev"

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ModelInfo(model_id="x", bad="y")


# ---------------------------------------------------------------------------
# Schema tests — ActionStepLog
# ---------------------------------------------------------------------------


class TestActionStepLog:
    def test_valid_minimal(self):
        s = ActionStepLog(step_number=1)
        assert s.step_number == 1
        assert s.tool_invocations == []
        assert s.observations == ""
        assert s.error is None

    def test_with_tool_invocations(self):
        s = ActionStepLog(
            step_number=1,
            tool_invocations=[
                ToolInvocation(tool_name="weather", arguments={"city": "Paris"}),
                ToolInvocation(tool_name="news", arguments={"query": "France"}),
            ],
            observations="weather: 9.6°C\nnews: article 1",
            duration_seconds=3.09,
            input_tokens=3041,
            output_tokens=75,
            total_tokens=3116,
        )
        assert len(s.tool_invocations) == 2
        assert s.tool_invocations[0].tool_name == "weather"
        assert s.duration_seconds == 3.09
        assert s.total_tokens == 3116

    def test_step_number_must_be_positive(self):
        with pytest.raises(ValidationError):
            ActionStepLog(step_number=0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ActionStepLog(step_number=1, surprise="boom")

    def test_error_field(self):
        s = ActionStepLog(step_number=1, error="TimeoutError: tool timed out")
        assert s.error == "TimeoutError: tool timed out"


# ---------------------------------------------------------------------------
# Schema tests — ConversationTurnLog
# ---------------------------------------------------------------------------


class TestConversationTurnLog:
    def _make(self, **overrides):
        defaults = dict(
            conversation_id="conv-123",
            model=ModelInfo(model_id="openai/gpt-4o"),
            user_input="Hello",
            final_answer="Hi there!",
        )
        defaults.update(overrides)
        return ConversationTurnLog(**defaults)

    def test_valid_minimal(self):
        entry = self._make()
        assert entry.conversation_id == "conv-123"
        assert entry.final_answer == "Hi there!"
        assert entry.agent_steps == []
        assert entry.turn_number == 0

    def test_auto_log_id_and_timestamp(self):
        e1 = self._make()
        e2 = self._make()
        assert e1.log_id != e2.log_id
        assert "T" in e1.timestamp  # ISO-8601

    def test_with_agent_steps(self):
        entry = self._make(
            agent_steps=[
                ActionStepLog(
                    step_number=1,
                    tool_invocations=[
                        ToolInvocation(tool_name="weather", arguments={"city": "Paris"},
                                       response='{"temperature": 9.6}'),
                        ToolInvocation(tool_name="news", arguments={"query": "France"},
                                       response="1. Mbappé scores..."),
                    ],
                    observations="combined obs",
                    duration_seconds=3.09,
                    input_tokens=3041,
                    output_tokens=75,
                    total_tokens=3116,
                )
            ]
        )
        assert len(entry.agent_steps) == 1
        assert len(entry.agent_steps[0].tool_invocations) == 2
        assert entry.agent_steps[0].tool_invocations[0].tool_name == "weather"
        assert "9.6" in entry.agent_steps[0].tool_invocations[0].response
        assert entry.agent_steps[0].tool_invocations[1].tool_name == "news"
        assert "Mbappé" in entry.agent_steps[0].tool_invocations[1].response
        assert entry.agent_steps[0].duration_seconds == 3.09

    def test_empty_input_rejected(self):
        with pytest.raises(ValidationError):
            self._make(user_input="")

    def test_whitespace_input_rejected(self):
        with pytest.raises(ValidationError):
            self._make(user_input="   ")

    def test_negative_turn_number_rejected(self):
        with pytest.raises(ValidationError):
            self._make(turn_number=-1)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            self._make(surprise="boom")

    def test_model_dump_json_roundtrip(self):
        entry = self._make(
            agent_steps=[
                ActionStepLog(
                    step_number=1,
                    tool_invocations=[ToolInvocation(tool_name="calc", arguments={"expr": "1+1"})],
                    observations="2",
                )
            ]
        )
        raw = entry.model_dump_json()
        loaded = json.loads(raw)
        assert loaded["user_input"] == "Hello"
        assert loaded["conversation_id"] == "conv-123"
        assert loaded["agent_steps"][0]["tool_invocations"][0]["tool_name"] == "calc"


# ---------------------------------------------------------------------------
# JSONL write tests
# ---------------------------------------------------------------------------


class TestWriteJsonl:
    def test_appends_line(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        entry = ConversationTurnLog(
            conversation_id="c1",
            model=ModelInfo(model_id="openai/gpt-4o"),
            user_input="test",
        )
        _write_jsonl(path, entry)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["conversation_id"] == "c1"

    def test_appends_multiple_turns(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        for i in range(3):
            entry = ConversationTurnLog(
                conversation_id="conv",
                turn_number=i,
                model=ModelInfo(model_id="openai/gpt-4o"),
                user_input=f"turn {i}",
            )
            _write_jsonl(path, entry)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        turns = [json.loads(l)["turn_number"] for l in lines]
        assert turns == [0, 1, 2]

    def test_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "nested" / "deep" / "log.jsonl"
        entry = ConversationTurnLog(
            conversation_id="c1",
            model=ModelInfo(model_id="openai/gpt-4o"),
            user_input="hello",
        )
        _write_jsonl(path, entry)
        assert path.exists()


# ---------------------------------------------------------------------------
# log_interaction integration (queue + worker)
# ---------------------------------------------------------------------------


class TestLogInteraction:
    def test_enqueues_and_flushes_simple(self, tmp_path: Path):
        init_logger(log_dir=str(tmp_path), repo_id=None, hf_token=None)
        log_interaction(
            conversation_id="conv-abc",
            model_id="openai/gpt-4o",
            app_version="test",
            user_input="What is 2+2?",
            final_answer="4",
            agent_steps=[],
        )
        _log_queue.join()
        log_path = tmp_path / "interactions.jsonl"
        assert log_path.exists()
        data = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert data["user_input"] == "What is 2+2?"
        assert data["final_answer"] == "4"
        assert data["model"]["model_id"] == "openai/gpt-4o"
        assert data["conversation_id"] == "conv-abc"

    def test_with_full_agent_steps(self, tmp_path: Path):
        init_logger(log_dir=str(tmp_path), repo_id=None, hf_token=None)
        log_interaction(
            conversation_id="conv-xyz",
            model_id="openai/gpt-4o",
            app_version="1.0",
            user_input="Weather in Paris and French headlines",
            final_answer="Weather: 9.6°C; Headlines: ...",
            agent_steps=[
                {
                    "step_number": 1,
                    "model_output": "I'll fetch weather and news.",
                    "tool_invocations": [
                        {"id": "tc-1", "tool_name": "get_current_weather",
                         "arguments": {"location_name": "Paris", "temperature_unit": "celsius"},
                         "response": '{"temperature": 9.6, "weather_description": "Mainly clear"}'},
                        {"id": "tc-2", "tool_name": "news_search",
                         "arguments": {"query": "France", "max_results": 5},
                         "response": "1. Mbappé scores..."},
                    ],
                    "observations": '{"temperature": 9.6}\n1. Mbappé scores...',
                    "duration_seconds": 3.09,
                    "input_tokens": 3041,
                    "output_tokens": 75,
                    "total_tokens": 3116,
                    "error": None,
                }
            ],
        )
        _log_queue.join()
        log_path = tmp_path / "interactions.jsonl"
        data = json.loads(log_path.read_text(encoding="utf-8").strip())
        assert len(data["agent_steps"]) == 1
        step = data["agent_steps"][0]
        assert step["step_number"] == 1
        assert step["duration_seconds"] == 3.09
        assert step["input_tokens"] == 3041
        assert step["total_tokens"] == 3116
        assert len(step["tool_invocations"]) == 2
        # Verify per-tool responses are stored
        weather_inv = step["tool_invocations"][0]
        assert weather_inv["tool_name"] == "get_current_weather"
        assert weather_inv["arguments"]["location_name"] == "Paris"
        assert "9.6" in weather_inv["response"]
        news_inv = step["tool_invocations"][1]
        assert news_inv["tool_name"] == "news_search"
        assert "Mbappé" in news_inv["response"]

    def test_invalid_input_is_swallowed(self, tmp_path: Path):
        """Validation failure must not raise — entry is discarded silently."""
        init_logger(log_dir=str(tmp_path), repo_id=None, hf_token=None)
        log_interaction(
            conversation_id="conv-1",
            model_id="openai/gpt-4o",
            app_version="dev",
            user_input="",  # invalid — empty
            final_answer="",
            agent_steps=[],
        )
        time.sleep(0.1)
        log_path = tmp_path / "interactions.jsonl"
        assert not log_path.exists()

    def test_multi_turn_same_conversation(self, tmp_path: Path):
        """Two turns with the same conversation_id should produce 2 JSONL lines."""
        init_logger(log_dir=str(tmp_path), repo_id=None, hf_token=None)
        for i in range(2):
            log_interaction(
                conversation_id="conv-multi",
                model_id="openai/gpt-4o",
                app_version="dev",
                user_input=f"Turn {i} question",
                final_answer=f"Turn {i} answer",
                agent_steps=[],
                turn_number=i,
            )
        _log_queue.join()
        log_path = tmp_path / "interactions.jsonl"
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        turns = [json.loads(l)["turn_number"] for l in lines]
        assert sorted(turns) == [0, 1]
        conv_ids = {json.loads(l)["conversation_id"] for l in lines}
        assert conv_ids == {"conv-multi"}


# ---------------------------------------------------------------------------
# Timer tests
# ---------------------------------------------------------------------------


class TestTimer:
    def test_basic_flow(self):
        t = Timer("test")
        t.start()
        time.sleep(0.05)
        t.add_step("step1")
        t.end()
        result = t.to_json()
        assert result["name"] == "test"
        assert result["total_time"] > 0
        assert "step1" in result

    def test_formatted_result(self):
        t = Timer()
        t.start()
        t.add_step("do thing")
        t.end()
        text = t.formatted_result()
        assert "do thing" in text
        assert "Total time" in text

    def test_end_without_start_raises(self):
        t = Timer()
        with pytest.raises(RuntimeError):
            t.end()

    def test_end_without_steps_raises(self):
        t = Timer()
        t.start()
        with pytest.raises(RuntimeError):
            t.end()

