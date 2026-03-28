"""
src/evaluation.py — Batch evaluation workflow (P3).

Fetches agent traces from Langfuse for a given date range, scores each trace
on three quality dimensions using an LLM-as-judge, and posts the scores back
to Langfuse via langfuse.create_score().

Scoring criteria
----------------
relevance        (0–1) How relevant is the assistant response to the user query?
correctness      (0–1) How factually accurate is the response given available tool data?
tool_efficiency  (0–1) How efficiently did the agent use its tools?

All scores are NUMERIC in [0.0, 1.0].  Scores are idempotent: each (trace_id,
criterion, run_id) tuple maps to a deterministic UUID so re-running the script
against the same traces does not create duplicate records.

Usage
-----
python -m src.evaluation --from 2026-03-01 --to 2026-03-31
python -m src.evaluation --from 2026-03-01 --to 2026-03-31 --run-id my-run-v2
"""

from __future__ import annotations

import argparse
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse  # noqa: E402 — must be importable for test patching
from smolagents import LiteLLMModel  # noqa: E402 — must be importable for test patching

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1-shot judge prompt templates (U2 from analysis report)
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = (
    "You are an evaluator. Given the user query and the assistant response below, "
    "rate how relevant the response is to the query on a scale of 0.0 to 1.0, "
    "where 1.0 = fully relevant and addresses all aspects, "
    "0.0 = completely irrelevant. Return ONLY a floating-point number.\n\n"
    "Query: {input}\n"
    "Response: {output}"
)

_CORRECTNESS_PROMPT = (
    "You are an evaluator. Given the user query, the assistant response, and any "
    "tool outputs used, rate factual correctness on a scale of 0.0 to 1.0, "
    "where 1.0 = fully accurate, 0.0 = factually wrong. "
    "Return ONLY a floating-point number.\n\n"
    "Query: {input}\n"
    "Tool outputs: {tool_outputs}\n"
    "Response: {output}"
)

_TOOL_EFFICIENCY_PROMPT = (
    "You are an evaluator. Given the user query and the list of tool calls made, "
    "rate how efficiently the agent used tools "
    "(0.0 = unnecessary/excessive tool calls, "
    "1.0 = exactly the right tools called once each). "
    "Return ONLY a floating-point number.\n\n"
    "Query: {input}\n"
    "Tool calls: {tool_calls}"
)

CRITERIA: list[dict] = [
    {"name": "relevance", "prompt_template": _RELEVANCE_PROMPT},
    {"name": "correctness", "prompt_template": _CORRECTNESS_PROMPT},
    {"name": "tool_efficiency", "prompt_template": _TOOL_EFFICIENCY_PROMPT},
]


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _deterministic_score_id(trace_id: str, criterion: str, run_id: str) -> str:
    """
    Generate a deterministic UUID5 score ID for idempotent re-runs.

    The same (trace_id, criterion, run_id) triple always produces the same UUID,
    so calling langfuse.create_score() twice with the same score_id is a no-op.
    """
    namespace = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # URL namespace
    name = f"{trace_id}:{criterion}:{run_id}"
    return str(uuid.uuid5(namespace, name))


def _clamp(value: float) -> float:
    """Clamp a score to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


def _parse_score(raw: str) -> float:
    """
    Parse the LLM's raw output into a float in [0.0, 1.0].
    Returns 0.5 (neutral) on parse failure so scoring continues.
    """
    try:
        return _clamp(float(raw.strip()))
    except (ValueError, AttributeError):
        logger.warning("Could not parse score from LLM output %r — defaulting to 0.5", raw)
        return 0.5


def _extract_trace_fields(trace) -> dict:
    """Extract input, output, tool_outputs, and tool_calls from a Langfuse trace object."""
    input_text = getattr(trace, "input", "") or ""
    output_text = getattr(trace, "output", "") or ""

    # Attempt to pull tool call data from trace observations/spans
    tool_calls_parts: list[str] = []
    tool_outputs_parts: list[str] = []
    for obs in getattr(trace, "observations", []) or []:
        obs_name = getattr(obs, "name", "")
        if "tool" in obs_name.lower():
            tool_input = str(getattr(obs, "input", ""))
            tool_output = str(getattr(obs, "output", ""))
            tool_calls_parts.append(f"{obs_name}({tool_input})")
            if tool_output:
                tool_outputs_parts.append(f"{obs_name}: {tool_output}")

    return {
        "input": str(input_text)[:2000],
        "output": str(output_text)[:2000],
        "tool_calls": "; ".join(tool_calls_parts) if tool_calls_parts else "none",
        "tool_outputs": "; ".join(tool_outputs_parts) if tool_outputs_parts else "none",
    }


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def _score_trace(trace, model, run_id: str, langfuse_client) -> dict:
    """
    Run all scoring criteria against one trace and post scores to Langfuse.

    Returns a dict mapping criterion name → score value.
    """
    fields = _extract_trace_fields(trace)
    results: dict[str, float] = {}

    for criterion in CRITERIA:
        prompt = criterion["prompt_template"].format(**fields)
        try:
            raw_output = model(prompt)
            # LiteLLMModel returns a string when called directly
            score = _parse_score(str(raw_output))
        except Exception as exc:
            logger.warning(
                "LLM judge failed for criterion '%s' on trace %s: %s",
                criterion["name"],
                trace.id,
                exc,
            )
            score = 0.5

        score_id = _deterministic_score_id(trace.id, criterion["name"], run_id)
        try:
            langfuse_client.score(
                trace_id=trace.id,
                name=criterion["name"],
                value=score,
                data_type="NUMERIC",
                id=score_id,
                comment=f"Automated evaluation run: {run_id}",
            )
        except Exception as exc:
            logger.warning(
                "Failed to post score for trace %s criterion '%s': %s",
                trace.id,
                criterion["name"],
                exc,
            )

        results[criterion["name"]] = score

    return results


def run_evaluation(
    from_date: datetime,
    to_date: datetime,
    run_id: Optional[str] = None,
) -> list[dict]:
    """
    Fetch traces from Langfuse between *from_date* and *to_date*, score them,
    and return a list of result records.

    Parameters
    ----------
    from_date, to_date : datetime
        Inclusive date range (UTC).
    run_id : str, optional
        Evaluation run identifier used in idempotent score IDs.
        Auto-generated if not provided.
    """
    run_id = run_id or str(uuid.uuid4())[:8]
    langfuse = Langfuse()

    model_id = os.environ.get("MODEL_ID", "")
    model_api_key = os.environ.get("MODEL_API_KEY", "")
    if not model_id or not model_api_key:
        raise EnvironmentError(
            "MODEL_ID and MODEL_API_KEY must be set to run evaluations."
        )

    model = LiteLLMModel(model_id=model_id, api_key=model_api_key)

    # Fetch traces — Langfuse v3 API
    traces_page = langfuse.fetch_traces(
        from_timestamp=from_date,
        to_timestamp=to_date,
    )
    traces = list(getattr(traces_page, "data", traces_page) or [])

    if not traces:
        logger.info("No traces found in the specified date range.")
        return []

    logger.info("Evaluating %d trace(s) with run_id=%r …", len(traces), run_id)
    records: list[dict] = []
    for trace in traces:
        scores = _score_trace(trace, model, run_id, langfuse)
        records.append({"trace_id": trace.id, "run_id": run_id, **scores})

    return records


# ---------------------------------------------------------------------------
# CLI entry point (T020)
# ---------------------------------------------------------------------------

def _print_table(records: list[dict]) -> None:
    if not records:
        print("No traces evaluated.")
        return

    header = f"{'Trace ID':<38} {'relevance':>10} {'correctness':>12} {'tool_efficiency':>16}"
    print("\n" + header)
    print("-" * len(header))
    for r in records:
        print(
            f"{r['trace_id']:<38} "
            f"{r.get('relevance', 'N/A'):>10.2f} "
            f"{r.get('correctness', 'N/A'):>12.2f} "
            f"{r.get('tool_efficiency', 'N/A'):>16.2f}"
        )
    print(f"\nTotal evaluated: {len(records)}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    parser = argparse.ArgumentParser(
        description="Run batch evaluation over Langfuse traces."
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        required=True,
        help="Start date (ISO 8601, e.g. 2026-03-01)",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        required=True,
        help="End date (ISO 8601, e.g. 2026-03-31)",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Optional run identifier for idempotent scores (default: auto-generated)",
    )
    args = parser.parse_args()

    from_dt = datetime.fromisoformat(args.from_date).replace(tzinfo=timezone.utc)
    to_dt = datetime.fromisoformat(args.to_date).replace(tzinfo=timezone.utc)

    records = run_evaluation(from_dt, to_dt, run_id=args.run_id)
    _print_table(records)


if __name__ == "__main__":
    main()
