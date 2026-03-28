"""
tests/unit/test_evaluation.py — Unit tests for src/evaluation.py.

All Langfuse and LLM calls are mocked so these tests run offline with no
credentials required.  Run with: pytest tests/unit/
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import (
    _clamp,
    _deterministic_score_id,
    _parse_score,
    CRITERIA,
)


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_clamp_in_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below_zero(self):
        assert _clamp(-0.1) == 0.0

    def test_clamp_above_one(self):
        assert _clamp(1.1) == 1.0

    def test_clamp_exact_boundaries(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------


class TestParseScore:
    def test_parses_valid_float(self):
        assert _parse_score("0.85") == pytest.approx(0.85)

    def test_strips_whitespace(self):
        assert _parse_score("  0.9\n") == pytest.approx(0.9)

    def test_clamps_above_one(self):
        assert _parse_score("1.5") == 1.0

    def test_clamps_below_zero(self):
        assert _parse_score("-0.3") == 0.0

    def test_returns_neutral_on_garbage(self):
        assert _parse_score("not a number") == pytest.approx(0.5)

    def test_returns_neutral_on_empty(self):
        assert _parse_score("") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _deterministic_score_id
# ---------------------------------------------------------------------------


class TestDeterministicScoreId:
    def test_same_inputs_produce_same_id(self):
        id1 = _deterministic_score_id("trace-abc", "relevance", "run-1")
        id2 = _deterministic_score_id("trace-abc", "relevance", "run-1")
        assert id1 == id2

    def test_different_criteria_produce_different_ids(self):
        id_rel = _deterministic_score_id("trace-abc", "relevance", "run-1")
        id_cor = _deterministic_score_id("trace-abc", "correctness", "run-1")
        assert id_rel != id_cor

    def test_different_run_ids_produce_different_ids(self):
        id1 = _deterministic_score_id("trace-abc", "relevance", "run-1")
        id2 = _deterministic_score_id("trace-abc", "relevance", "run-2")
        assert id1 != id2

    def test_output_is_valid_uuid_string(self):
        score_id = _deterministic_score_id("trace-xyz", "correctness", "run-99")
        # Should not raise
        parsed = uuid.UUID(score_id)
        assert str(parsed) == score_id


# ---------------------------------------------------------------------------
# CRITERIA constant
# ---------------------------------------------------------------------------


class TestCriteriaDefinitions:
    def test_three_criteria_defined(self):
        assert len(CRITERIA) == 3

    def test_criterion_names(self):
        names = {c["name"] for c in CRITERIA}
        assert names == {"relevance", "correctness", "tool_efficiency"}

    def test_each_criterion_has_prompt_template(self):
        for criterion in CRITERIA:
            assert "prompt_template" in criterion
            assert len(criterion["prompt_template"]) > 20


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgParsing:
    def test_from_and_to_required(self):
        import argparse
        from src.evaluation import main

        with patch("sys.argv", ["evaluation"]):
            with pytest.raises(SystemExit):
                main()

    def test_valid_date_args_accepted(self, capsys):
        from src.evaluation import run_evaluation

        mock_records = [
            {
                "trace_id": "test-trace-001",
                "run_id": "test-run",
                "relevance": 0.8,
                "correctness": 0.9,
                "tool_efficiency": 0.7,
            }
        ]

        with patch("src.evaluation.run_evaluation", return_value=mock_records) as mock_run:
            with patch(
                "sys.argv",
                ["evaluation", "--from", "2026-03-01", "--to", "2026-03-31"],
            ):
                from src.evaluation import main

                main()

            call_args = mock_run.call_args
            from_dt = call_args[0][0]
            to_dt = call_args[0][1]
            assert from_dt.year == 2026 and from_dt.month == 3 and from_dt.day == 1
            assert to_dt.year == 2026 and to_dt.month == 3 and to_dt.day == 31


# ---------------------------------------------------------------------------
# run_evaluation — mocked end-to-end
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def _make_mock_trace(self, trace_id: str = "trace-001") -> MagicMock:
        trace = MagicMock()
        trace.id = trace_id
        trace.input = "What is 2 + 2?"
        trace.output = "2 + 2 equals 4."
        trace.observations = []
        return trace

    def test_returns_one_record_per_trace(self):
        fake_trace = self._make_mock_trace()
        mock_langfuse = MagicMock()
        mock_langfuse.fetch_traces.return_value = SimpleNamespace(data=[fake_trace])
        mock_langfuse.score = MagicMock()

        mock_model = MagicMock(return_value="0.9")

        with (
            patch("src.evaluation.Langfuse", return_value=mock_langfuse),  # type: ignore
            patch("src.evaluation.LiteLLMModel", return_value=mock_model),  # type: ignore
            patch.dict(
                "os.environ",
                {"MODEL_ID": "openai/gpt-4o", "MODEL_API_KEY": "sk-test"},
            ),
        ):
            from src.evaluation import run_evaluation

            from_dt = datetime(2026, 3, 1, tzinfo=timezone.utc)
            to_dt = datetime(2026, 3, 31, tzinfo=timezone.utc)
            records = run_evaluation(from_dt, to_dt, run_id="unit-test")

        assert len(records) == 1
        assert records[0]["trace_id"] == "trace-001"

    def test_scores_clamped_to_range(self):
        fake_trace = self._make_mock_trace()
        mock_langfuse = MagicMock()
        mock_langfuse.fetch_traces.return_value = SimpleNamespace(data=[fake_trace])
        mock_langfuse.score = MagicMock()

        # LLM returns out-of-range values
        mock_model = MagicMock(return_value="99.0")

        with (
            patch("src.evaluation.Langfuse", return_value=mock_langfuse),
            patch("src.evaluation.LiteLLMModel", return_value=mock_model),
            patch.dict(
                "os.environ",
                {"MODEL_ID": "openai/gpt-4o", "MODEL_API_KEY": "sk-test"},
            ),
        ):
            from src.evaluation import run_evaluation

            records = run_evaluation(
                datetime(2026, 3, 1, tzinfo=timezone.utc),
                datetime(2026, 3, 31, tzinfo=timezone.utc),
                run_id="clamp-test",
            )

        for criterion in ("relevance", "correctness", "tool_efficiency"):
            assert 0.0 <= records[0][criterion] <= 1.0

    def test_idempotent_score_ids(self):
        """Re-running with the same run_id produces the same score IDs."""
        fake_trace = self._make_mock_trace("trace-idem")
        mock_langfuse = MagicMock()
        mock_langfuse.fetch_traces.return_value = SimpleNamespace(data=[fake_trace])
        posted_ids: list[str] = []
        mock_langfuse.score.side_effect = lambda **kwargs: posted_ids.append(
            kwargs["id"]
        )

        mock_model = MagicMock(return_value="0.8")

        def _run():
            posted_ids.clear()
            with (
                patch("src.evaluation.Langfuse", return_value=mock_langfuse),
                patch("src.evaluation.LiteLLMModel", return_value=mock_model),
                patch.dict(
                    "os.environ",
                    {"MODEL_ID": "openai/gpt-4o", "MODEL_API_KEY": "sk-test"},
                ),
            ):
                from src.evaluation import run_evaluation

                run_evaluation(
                    datetime(2026, 3, 1, tzinfo=timezone.utc),
                    datetime(2026, 3, 31, tzinfo=timezone.utc),
                    run_id="fixed-run-id",
                )
            return list(posted_ids)

        ids_run1 = _run()
        ids_run2 = _run()
        assert ids_run1 == ids_run2, "Score IDs must be deterministic across re-runs."

    def test_empty_date_range_returns_empty_list(self):
        mock_langfuse = MagicMock()
        mock_langfuse.fetch_traces.return_value = SimpleNamespace(data=[])

        with (
            patch("src.evaluation.Langfuse", return_value=mock_langfuse),
            patch("src.evaluation.LiteLLMModel", return_value=MagicMock()),
            patch.dict(
                "os.environ",
                {"MODEL_ID": "openai/gpt-4o", "MODEL_API_KEY": "sk-test"},
            ),
        ):
            from src.evaluation import run_evaluation

            records = run_evaluation(
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 2, tzinfo=timezone.utc),
            )

        assert records == []
