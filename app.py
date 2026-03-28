"""
app.py — Gradio entry point for the AI Assistant.

Wires together:
  - Config loading (src/config.py)
  - MCP tool discovery (src/mcp_client.py)
  - Agent construction (src/agent.py)
  - Telemetry bootstrap (src/telemetry.py)
  - Gradio chat UI with streaming, session isolation, and Langfuse metadata panel
"""

import logging
import os
import uuid
from contextlib import ExitStack
from typing import Generator

import gradio as gr

from src.agent import build_agent
from src.config import load_config
from src.data_logger import init_logger, log_interaction
from src.mcp_client import MCPToolError, build_mcp_tools
from theme import theme

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup: config, MCP tools, agent, telemetry
# ---------------------------------------------------------------------------

cfg = load_config()

mcp_tools, _mcp_stack = build_mcp_tools(
    [cfg.mcp_server_url_1, cfg.mcp_server_url_2]
)

agent = build_agent(
    tools=mcp_tools,
    model_id=cfg.model_id,
    model_api_key=cfg.model_api_key,
)

# Telemetry — import deferred so startup works even if telemetry packages are
# absent; bootstrap_telemetry() handles its own graceful degradation.
try:
    from src.telemetry import bootstrap_telemetry

    langfuse = bootstrap_telemetry(cfg)
except ImportError:
    langfuse = None
    logger.warning("Telemetry module not available — continuing without tracing.")

# ---------------------------------------------------------------------------
# Data logger — initialise background JSONL logger + HF Hub sync
# ---------------------------------------------------------------------------

init_logger(
    log_dir=cfg.data_log_dir,
    repo_id=cfg.hf_dataset_repo_id,
    hf_token=cfg.hf_token,
    sync_interval=cfg.hf_sync_interval,
)


# ---------------------------------------------------------------------------
# Gradio chat handler
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 20  # context-window guard (edge case from spec)


def _make_langfuse_url(session_id: str) -> str:
    """Return a direct Langfuse trace URL for the current session, or empty string."""
    if (
        langfuse is None
        or not cfg.langfuse_base_url
        or not cfg.langfuse_project_id
    ):
        return ""
    base = cfg.langfuse_base_url.rstrip("/")
    return f"{base}/project/{cfg.langfuse_project_id}/traces?sessionId={session_id}"


def _has_tool_title(msg) -> bool:
    """Return True if *msg* is a tool-call ChatMessage (has a metadata title)."""
    meta = getattr(msg, "metadata", None)
    return isinstance(meta, dict) and bool(meta.get("title"))


def _extract_assistant_response(messages: list) -> str:
    """Return the final plain-text assistant response from new_messages."""
    for msg in reversed(messages):
        if not _has_tool_title(msg):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                return content
    return ""


def _parse_memory_steps(steps_slice: list, tool_responses: dict[str, str] | None = None) -> list[dict]:
    """
    Convert a slice of ``agent.memory.steps`` into a list of dicts
    matching the ``ActionStepLog`` schema.

    Only ``ActionStep`` objects are converted — ``TaskStep``,
    ``FinalAnswerStep``, and ``PlanningStep`` are skipped.
    The ``final_answer`` pseudo-tool is excluded from ``tool_invocations``
    (it is captured separately as ``final_answer`` on ``ConversationTurnLog``).

    Parameters
    ----------
    steps_slice:
        A list of memory step objects from ``agent.memory.steps``.
    tool_responses:
        Optional dict ``{tool_call_id: observation_str}`` captured during
        streaming from ``ToolOutput`` events.  When provided, each
        ``ToolInvocation.response`` is populated with the individual tool's
        output rather than leaving it empty.
    """
    responses = tool_responses or {}
    result = []
    for step in steps_slice:
        # Only process ActionStep objects
        if not hasattr(step, "tool_calls") or not hasattr(step, "step_number"):
            continue

        # Build structured tool invocations (ALL tool calls, not just [0])
        tool_invocations = []
        for tc in getattr(step, "tool_calls", None) or []:
            name = getattr(tc, "name", "unknown") or "unknown"
            if name == "final_answer":
                # Skip the pseudo-tool; its output is stored in final_answer
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

        timing = getattr(step, "timing", None)
        if timing is not None:
            start_time = getattr(timing, "start_time", None)
            end_time = getattr(timing, "end_time", None)
            duration = (end_time - start_time) if (start_time is not None and end_time is not None) else None
        else:
            duration = None

        token_usage = getattr(step, "token_usage", None)
        input_tokens = getattr(token_usage, "input_tokens", None) if token_usage else None
        output_tokens = getattr(token_usage, "output_tokens", None) if token_usage else None
        total_tokens = getattr(token_usage, "total_tokens", None) if token_usage else None

        error = getattr(step, "error", None)

        result.append(
            {
                "step_number": int(getattr(step, "step_number", len(result) + 1)),
                "model_output": str(getattr(step, "model_output", "") or ""),
                "tool_invocations": tool_invocations,
                "observations": str(getattr(step, "observations", "") or ""),
                "duration_seconds": float(duration) if duration is not None else None,
                "input_tokens": int(input_tokens) if input_tokens is not None else None,
                "output_tokens": int(output_tokens) if output_tokens is not None else None,
                "total_tokens": int(total_tokens) if total_tokens is not None else None,
                "error": str(error) if error is not None else None,
            }
        )
    return result


def _supplement_tool_panels(new_messages: list, steps_slice: list) -> list:
    """
    Inject missing tool-call panels into *new_messages* for any tool call
    in *steps_slice* whose name is not already represented.

    smolagents' ``_process_action_step`` only creates a panel for
    ``tool_calls[0]``; this function adds panels for ``tool_calls[1:]``
    so every parallel tool call is visible in the Gradio chatbot.

    Parameters
    ----------
    new_messages:
        The accumulated list of ``gr.ChatMessage`` objects from streaming.
    steps_slice:
        A list of memory step objects from ``agent.memory.steps``.
    """
    # Collect tool names already present in new_messages
    present_tools: set[str] = set()
    for msg in new_messages:
        if _has_tool_title(msg):
            title: str = msg.metadata.get("title", "")
            # smolagents title format: "🛠️ Used tool <name>"
            if "Used tool" in title:
                tool_name = title.split("Used tool", 1)[-1].strip()
                present_tools.add(tool_name)

    supplemented = list(new_messages)
    for step in steps_slice:
        for tc in getattr(step, "tool_calls", None) or []:
            name = getattr(tc, "name", "") or ""
            if name in ("final_answer", "") or name in present_tools:
                continue
            # Build a done-state panel for this missing tool call
            args = getattr(tc, "arguments", {})
            content = str(args) if not isinstance(args, dict) else str(args)
            supplemented.append(
                gr.ChatMessage(
                    role="assistant",
                    content=content,
                    metadata={"title": f"🛠️ Used tool {name}", "status": "done"},
                )
            )
            present_tools.add(name)
            logger.debug("data_logger: supplemented UI panel for tool '%s'", name)

    return supplemented


def _stream_with_tool_capture(
    task: str,
    tool_responses: dict[str, str],
):
    """
    Custom streaming generator that mirrors ``stream_to_gradio`` but also
    intercepts ``ToolOutput`` events yielded by ``agent.run(stream=True)``.

    ``stream_to_gradio`` silently drops ``ToolOutput`` events; this wrapper
    captures them so each ``ToolInvocation`` can store its individual
    response string rather than relying on the combined ``observations``.

    Parameters
    ----------
    task:
        The user message to run.
    tool_responses:
        Mutable dict populated in place: ``{tool_call_id: observation_str}``.
        Callers read this after the generator is exhausted.

    Yields
    ------
    ``gr.ChatMessage`` objects (tool-call panels and text chunks) — identical
    to what ``stream_to_gradio`` would yield.
    """
    from smolagents.agents import ChatMessageStreamDelta, ToolOutput
    from smolagents.gradio_ui import agglomerate_stream_deltas, pull_messages_from_step
    from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep

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
            # Capture individual tool response — not forwarded to UI
            tc_id = getattr(event, "id", "") or ""
            observation = getattr(event, "observation", "") or ""
            if tc_id:
                tool_responses[tc_id] = observation


def chat(
    message: str,
    history: list,
    session_id: str,
) -> Generator:
    """
    Gradio streaming chat handler.

    Yields incremental assistant tokens and, once complete, posts session
    metadata to the active Langfuse trace.

    Tool calls produced by the smolagents ToolCallingAgent are surfaced as
    expandable gr.ChatMessage panels (metadata title) in the chatbot, so the
    user can inspect every MCP tool invocation and its result inline.
    """
    # --- Empty-message guard (FR-001 / edge case) ---
    if not message or not message.strip():
        yield history, session_id, "Please enter a message before submitting."
        return

    # --- Context-window truncation guard (edge case from spec) ---
    truncation_notice = ""
    if len(history) >= MAX_HISTORY_TURNS * 2:
        history = history[-(MAX_HISTORY_TURNS * 2):]
        truncation_notice = (
            "\n\n_(Earlier messages were summarised to fit context limits.)_"
        )
        logger.info("Session %s: history trimmed to %d turns.", session_id, MAX_HISTORY_TURNS)

    # --- Langfuse v4: propagate session metadata to all child observations ---
    # We call __enter__() manually and intentionally skip __exit__() to avoid
    # "Failed to detach context" errors from OpenTelemetry.  These errors occur
    # because Gradio's streaming event loop resumes the generator in a different
    # async task context than the one where __enter__ (and the ContextVar tokens)
    # was called, making the reset() call in __exit__ raise ValueError.
    # It is safe to skip __exit__() here because Gradio creates a fresh context
    # copy (copy_context()) for every request, so the session attributes stay
    # scoped to this one request's task and are cleaned up when the task ends.
    if langfuse is not None:
        try:
            from langfuse import propagate_attributes
            propagate_attributes(
                session_id=session_id,
                metadata={"app_version": cfg.app_version},
            ).__enter__()
        except Exception as exc:
            logger.warning("Could not propagate session attributes: %s", exc)

    # --- Agent call with tool-call tracking ---
    try:
        # Snapshot the step count BEFORE this turn so we can extract only
        # the new ActionSteps produced during this turn after streaming ends.
        _steps_before = len(getattr(agent.memory, "steps", []) or [])

        # Per-tool responses captured from ToolOutput events during streaming.
        # Populated in place by _stream_with_tool_capture().
        _tool_responses: dict[str, str] = {}

        # new_messages accumulates every chunk yielded for this turn:
        #   - gr.ChatMessage with metadata.title  → tool-call panels (expandable)
        #   - gr.ChatMessage without metadata      → streaming final-answer text
        # Tool-call panels are updated in-place (pending → done) by matching title.
        new_messages: list = []

        for chunk in _stream_with_tool_capture(message, _tool_responses):
            if not hasattr(chunk, "content") or chunk.content is None:
                continue

            if _has_tool_title(chunk):
                # Tool-call chunk: update the matching existing panel or append.
                title = chunk.metadata["title"]
                idx = next(
                    (
                        i for i, m in enumerate(new_messages)
                        if _has_tool_title(m) and m.metadata["title"] == title
                    ),
                    None,
                )
                if idx is not None:
                    new_messages[idx] = chunk
                else:
                    new_messages.append(chunk)
            else:
                # Final-answer text chunk: replace the last non-tool message
                # (streaming update) or append a new one.
                last_text_idx = next(
                    (
                        i for i in range(len(new_messages) - 1, -1, -1)
                        if not _has_tool_title(new_messages[i])
                    ),
                    None,
                )
                if last_text_idx is not None:
                    new_messages[last_text_idx] = chunk
                else:
                    new_messages.append(chunk)

            yield (
                history + [{"role": "user", "content": message}] + new_messages,
                session_id,
                _make_langfuse_url(session_id),
            )

        if not new_messages:
            # Non-streaming fallback
            result = agent.run(message, reset=False)
            new_messages = [
                gr.ChatMessage(role="assistant", content=str(result) + truncation_notice)
            ]
            yield (
                history + [{"role": "user", "content": message}] + new_messages,
                session_id,
                _make_langfuse_url(session_id),
            )
        elif truncation_notice:
            # Append truncation notice to the last plain-text message.
            last_text_idx = next(
                (
                    i for i in range(len(new_messages) - 1, -1, -1)
                    if not _has_tool_title(new_messages[i])
                ),
                None,
            )
            if last_text_idx is not None:
                last_content = getattr(new_messages[last_text_idx], "content", "") or ""
                new_messages[last_text_idx] = gr.ChatMessage(
                    role="assistant", content=last_content + truncation_notice
                )
                yield (
                    history + [{"role": "user", "content": message}] + new_messages,
                    session_id,
                    _make_langfuse_url(session_id),
                )

        # --- Post-streaming: extract structured data from agent memory ---
        # Get only the steps produced during THIS turn (not accumulated history).
        _all_steps = list(getattr(agent.memory, "steps", []) or [])
        _new_steps = _all_steps[_steps_before:]

        # Supplement the UI with any tool-call panels that smolagents dropped
        # (it only shows tool_calls[0] per step; parallel calls are invisible).
        new_messages = _supplement_tool_panels(new_messages, _new_steps)
        # Emit the final supplemented state so the UI shows all tool calls.
        yield (
            history + [{"role": "user", "content": message}] + new_messages,
            session_id,
            _make_langfuse_url(session_id),
        )

        # --- Data logging: capture completed turn asynchronously ---
        log_interaction(
            conversation_id=session_id,
            model_id=cfg.model_id,
            app_version=cfg.app_version,
            user_input=message,
            final_answer=_extract_assistant_response(new_messages),
            agent_steps=_parse_memory_steps(_new_steps, _tool_responses),
            turn_number=len(history) // 2,  # each full turn = user + assistant
            extra={"langfuse_url": _make_langfuse_url(session_id)},
        )

    except MCPToolError:
        # MCPToolError is already converted to a safe "[Tool unavailable: ...]"
        # string by the tool wrapper in mcp_client.py and fed into agent reasoning.
        # If it somehow escapes, surface a generic message and re-raise to the log.
        logger.exception("MCPToolError escaped agent reasoning context.")
        yield (
            history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "A tool call failed. The assistant will try to answer without it."},
            ],
            session_id,
            "",
        )
    except Exception:
        # Top-level catch-all for non-MCP exceptions (model API errors, etc.) — T013.
        # Full traceback is logged; user receives a safe message.
        logger.exception("Unhandled error during agent.run() for session %s", session_id)
        yield (
            history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Something went wrong. Please try again."},
            ],
            session_id,
            "",
        )


# ---------------------------------------------------------------------------
# Load stylesheet
# ---------------------------------------------------------------------------

_css_path = os.path.join(os.path.dirname(__file__), "styles.css")
_css = open(_css_path).read() if os.path.exists(_css_path) else ""


# ---------------------------------------------------------------------------
# Gradio layout (gr.Blocks)
# ---------------------------------------------------------------------------

def _new_session_id() -> str:
    return str(uuid.uuid4())


_HEADER_HTML = """
<div class="app-header">
  <div class="app-header-icon">⚡</div>
  <div>
    <p class="app-header-title">AI Assistant</p>
    <p class="app-header-subtitle">smolagents &nbsp;·&nbsp; MCP tools &nbsp;·&nbsp; Langfuse observability</p>
  </div>
</div>
"""

with gr.Blocks(title="AI Assistant", theme=theme, css=_css) as demo:
    # Per-session state — each browser tab gets an independent session_id (T018)
    session_state = gr.State(value=_new_session_id)

    gr.HTML(_HEADER_HTML)

    chatbot = gr.Chatbot(
        label="Conversation",
        height=520,
        show_label=False,
        render_markdown=True,
        elem_classes=["chatbot"],
        avatar_images=(
            None,
            "https://huggingface.co/front/assets/huggingface_logo-noborder.svg",
        ),
    )

    with gr.Row(elem_classes=["input-row"]):
        msg_input = gr.Textbox(
            placeholder="Ask me anything…",
            lines=2,
            autofocus=True,
            show_label=False,
            container=False,
            scale=9,
        )
        submit_btn = gr.Button(
            "Send ↵",
            variant="primary",
            scale=1,
            min_width=100,
            elem_classes=["send-btn"],
        )

    with gr.Row():
        clear_btn = gr.Button(
            "🗑️  Clear conversation",
            variant="secondary",
            size="sm",
            elem_classes=["clear-btn"],
        )

    # Collapsible metadata panel for SC-006: session ID + Langfuse trace URL
    with gr.Accordion(
        "📊  Session info",
        open=False,
        elem_classes=["session-accordion"],
    ):
        session_info_md = gr.Markdown(
            "_Send a message to see session details._",
            elem_classes=["session-info-md"],
        )

    def _update_info(trace_url: str, session_id: str) -> str:
        lines = [f"**Session ID**: `{session_id}`"]
        if trace_url:
            lines.append(f"**Langfuse traces**: [{trace_url}]({trace_url})")
        else:
            lines.append("_Langfuse tracing not configured or in degraded mode._")
        return "\n\n".join(lines)

    def _submit(message: str, history: list, session_id: str):
        final_history, final_session_id, trace_url = history, session_id, ""
        for final_history, final_session_id, trace_url in chat(message, history, session_id):
            info = _update_info(trace_url, final_session_id)
            yield final_history, "", final_session_id, info

    (
        submit_btn.click(
            _submit,
            inputs=[msg_input, chatbot, session_state],
            outputs=[chatbot, msg_input, session_state, session_info_md],
            concurrency_limit=1,  # prevents race conditions (edge case from spec)
        )
    )
    (
        msg_input.submit(
            _submit,
            inputs=[msg_input, chatbot, session_state],
            outputs=[chatbot, msg_input, session_state, session_info_md],
            concurrency_limit=1,
        )
    )

    def _clear_history():
        agent.memory.reset()
        new_session_id = _new_session_id()
        return [], new_session_id, "_Send a message to see session details._"

    clear_btn.click(
        _clear_history,
        outputs=[chatbot, session_state, session_info_md],
    )


if __name__ == "__main__":
    demo.launch()
