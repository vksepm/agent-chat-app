"""
src/smolagents_adapter.py — Adapter that owns all structural assumptions about
smolagents internals.

Isolates every ``hasattr``/``getattr`` defensive chain, every private-symbol
import, and every type coercion into one place.  The rest of the codebase
interacts exclusively with the clean ``StepSummary`` dataclass and the two
public functions.

Public API
----------
* ``StepSummary``            — dataclass; one record per ``ActionStep``
* ``parse_action_steps()``   — converts a memory-steps slice into ``StepSummary`` list
* ``stream_with_tool_capture()`` — streams agent events, captures ``ToolOutput``
"""

import dataclasses
import logging
from typing import Any, Generator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StepSummary:
    """Parsed representation of one smolagents ``ActionStep``.

    Parameters
    ----------
    step_number:
        1-based index of the step within the current agent run.
    model_output:
        Raw LLM reasoning text emitted during this step.
    tool_invocations:
        All tool calls made in this step (``final_answer`` excluded).
        Each entry is a dict with keys ``id``, ``tool_name``, ``arguments``,
        and ``response``.
    observations:
        Combined observation string from all tool outputs.
    duration_seconds:
        Wall-clock duration derived from ``ActionStep.timing``.
        ``None`` when the attribute is absent (logged at WARNING level).
    input_tokens:
        LLM input token count from ``ActionStep.token_usage``.
        ``None`` when absent.
    output_tokens:
        LLM output token count from ``ActionStep.token_usage``.
        ``None`` when absent.
    total_tokens:
        Total token count from ``ActionStep.token_usage``.
        ``None`` when absent.
    error:
        Error string when the step raised an ``AgentError``, otherwise ``None``.
    """

    step_number: int
    model_output: str
    tool_invocations: list[dict]
    observations: str
    duration_seconds: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None


# ---------------------------------------------------------------------------
# parse_action_steps
# ---------------------------------------------------------------------------


def parse_action_steps(
    steps: list[Any],
    tool_responses: dict[str, str] | None = None,
) -> list[StepSummary]:
    """Convert a slice of ``agent.memory.steps`` into ``StepSummary`` records.

    Only ``ActionStep`` objects (those with both ``tool_calls`` and
    ``step_number`` attributes) are processed.  ``TaskStep``,
    ``FinalAnswerStep``, and ``PlanningStep`` are silently skipped.

    A WARNING is logged for each expected field that is absent on a step so
    that degradation is observable in logs rather than silent.

    Parameters
    ----------
    steps:
        A list of memory step objects from ``agent.memory.steps``.
    tool_responses:
        Optional ``{tool_call_id: observation_str}`` dict captured during
        streaming from ``ToolOutput`` events.  When provided, each
        ``tool_invocation["response"]`` is populated with the individual
        tool output.

    Returns
    -------
    list[StepSummary]
        One entry per ``ActionStep`` found in *steps*, in original order.
    """
    responses = tool_responses or {}
    result: list[StepSummary] = []

    for step in steps:
        # Filter: only ActionStep objects have both tool_calls and step_number
        if not hasattr(step, "tool_calls") or not hasattr(step, "step_number"):
            continue

        # --- Tool invocations ---
        tool_invocations: list[dict] = []
        for tc in getattr(step, "tool_calls", None) or []:
            name = getattr(tc, "name", "unknown") or "unknown"
            if name == "final_answer":
                continue
            args = getattr(tc, "arguments", {})
            tc_id = getattr(tc, "id", "") or ""
            tool_invocations.append(
                {
                    "id": tc_id,
                    "tool_name": name,
                    "arguments": args if isinstance(args, dict) else str(args),
                    "response": responses.get(tc_id, ""),
                }
            )

        # --- Timing ---
        timing = getattr(step, "timing", None)
        if timing is None:
            logger.warning(
                "smolagents_adapter: step %s has no 'timing' attribute — "
                "duration_seconds will be None",
                getattr(step, "step_number", "?"),
            )
            duration_seconds = None
        else:
            start_time = getattr(timing, "start_time", None)
            end_time = getattr(timing, "end_time", None)
            if start_time is None or end_time is None:
                logger.warning(
                    "smolagents_adapter: step %s timing is missing start_time or "
                    "end_time — duration_seconds will be None",
                    getattr(step, "step_number", "?"),
                )
                duration_seconds = None
            else:
                try:
                    duration_seconds = float(end_time - start_time)
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "smolagents_adapter: step %s could not compute duration: %s",
                        getattr(step, "step_number", "?"),
                        exc,
                    )
                    duration_seconds = None

        # --- Token usage ---
        token_usage = getattr(step, "token_usage", None)
        if token_usage is None:
            logger.warning(
                "smolagents_adapter: step %s has no 'token_usage' attribute — "
                "token counts will be None",
                getattr(step, "step_number", "?"),
            )
            input_tokens = output_tokens = total_tokens = None
        else:
            raw_input = getattr(token_usage, "input_tokens", None)
            raw_output = getattr(token_usage, "output_tokens", None)
            raw_total = getattr(token_usage, "total_tokens", None)
            try:
                input_tokens = int(raw_input) if raw_input is not None else None
            except (TypeError, ValueError):
                input_tokens = None
            try:
                output_tokens = int(raw_output) if raw_output is not None else None
            except (TypeError, ValueError):
                output_tokens = None
            try:
                total_tokens = int(raw_total) if raw_total is not None else None
            except (TypeError, ValueError):
                total_tokens = None

        # --- Error ---
        raw_error = getattr(step, "error", None)
        error = str(raw_error) if raw_error is not None else None

        result.append(
            StepSummary(
                step_number=int(getattr(step, "step_number", len(result) + 1)),
                model_output=str(getattr(step, "model_output", "") or ""),
                tool_invocations=tool_invocations,
                observations=str(getattr(step, "observations", "") or ""),
                duration_seconds=duration_seconds,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                error=error,
            )
        )

    return result


# ---------------------------------------------------------------------------
# stream_with_tool_capture
# ---------------------------------------------------------------------------


def stream_with_tool_capture(
    agent: Any,
    task: str,
    tool_responses: dict[str, str],
) -> Generator:
    """Stream ``agent.run()`` events, capturing ``ToolOutput`` observations.

    All smolagents private imports are confined here.  If a required private
    symbol cannot be imported, an ``ImportError`` is raised immediately with
    an actionable message instead of crashing mid-stream.

    Parameters
    ----------
    agent:
        A ``smolagents.ToolCallingAgent`` (or compatible) instance.
    task:
        The user message to run.
    tool_responses:
        Mutable dict populated in place: ``{tool_call_id: observation_str}``.
        Callers read this after the generator is exhausted.

    Yields
    ------
    ``gr.ChatMessage`` objects (tool-call panels) and plain ``str`` chunks
    for streaming final-answer text — identical to what ``stream_to_gradio``
    would yield.

    Raises
    ------
    ImportError
        If required smolagents private symbols cannot be imported.
    """
    try:
        from smolagents.agents import ChatMessageStreamDelta, ToolOutput
        from smolagents.gradio_ui import agglomerate_stream_deltas, pull_messages_from_step
        from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep
    except ImportError as exc:
        raise ImportError(
            f"smolagents_adapter: required smolagents private symbol could not be "
            f"imported ({exc}). Ensure smolagents is installed and up to date. "
            f"If smolagents was updated, the adapter may need updating to match the "
            f"new internal API."
        ) from exc

    accumulated_events: list = []
    for event in agent.run(task, stream=True, reset=False):
        if isinstance(event, (ActionStep, PlanningStep, FinalAnswerStep)):
            for message in pull_messages_from_step(
                event,
                skip_model_outputs=getattr(agent, "stream_outputs", False),
            ):
                yield message
            accumulated_events = []
        elif isinstance(event, ChatMessageStreamDelta):
            accumulated_events.append(event)
            text = agglomerate_stream_deltas(accumulated_events).render_as_markdown()
            yield text
        elif isinstance(event, ToolOutput):
            tc_id = getattr(event, "id", "") or ""
            observation = getattr(event, "observation", "") or ""
            if tc_id:
                tool_responses[tc_id] = observation
