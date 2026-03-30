"""
app.py — Gradio entry point for the AI Assistant.

Wires together:
  - Config loading (src/config.py)
  - MCP tool discovery (src/mcp_client.py)
  - Agent construction (src/agent.py)
  - Telemetry bootstrap (src/telemetry.py)
  - Gradio chat UI with streaming, session isolation, and Langfuse metadata panel
"""

import atexit
import dataclasses
import logging
import os
import uuid
from contextlib import ExitStack
from typing import Generator

import gradio as gr

from src.agent import build_agent
from src.config import load_config
from src.data_logger import DataLogger
from src.mcp_client import MCPToolError, build_mcp_tools
from src.smolagents_adapter import parse_action_steps, stream_with_tool_capture
from src.tool_panel_manager import ToolPanelManager
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
atexit.register(_mcp_stack.close)

agent = build_agent(
    tools=mcp_tools,
    config=cfg,
)

# Telemetry — TelemetrySession lives in our own module (no optional deps at
# module level) so it is always importable.  bootstrap_telemetry() is deferred
# inside a try/except so startup works even when Langfuse isn't installed.
from src.telemetry import TelemetrySession

try:
    from src.telemetry import bootstrap_telemetry, shutdown_telemetry

    langfuse = bootstrap_telemetry(cfg)
    atexit.register(shutdown_telemetry, langfuse)
except ImportError:
    langfuse = None
    logger.warning("Telemetry module not available — continuing without tracing.")

# ---------------------------------------------------------------------------
# Data logger — initialise background JSONL logger + HF Hub sync
# ---------------------------------------------------------------------------

_data_logger = DataLogger(
    log_dir=cfg.data_log_dir,
    repo_id=cfg.hf_dataset_repo_id,
    hf_token=cfg.hf_token,
    sync_interval=cfg.hf_sync_interval,
).start()

# Drain the queue and optionally push a final HF Hub upload on process exit
# (Ctrl-C, demo.close(), or SIGTERM).
atexit.register(_data_logger.shutdown, timeout=10, final_sync=True)


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
    TelemetrySession(langfuse, session_id, cfg.app_version).attach()

    # --- Agent call with tool-call tracking ---
    try:
        # Snapshot the step count BEFORE this turn so we can extract only
        # the new ActionSteps produced during this turn after streaming ends.
        _steps_before = len(getattr(agent.memory, "steps", []) or [])

        # Per-tool responses captured from ToolOutput events during streaming.
        # Populated in place by stream_with_tool_capture().
        _tool_responses: dict[str, str] = {}

        # ToolPanelManager accumulates, deduplicates, and supplements
        # gr.ChatMessage objects for this turn.
        panel_mgr = ToolPanelManager()

        for chunk in stream_with_tool_capture(agent, message, _tool_responses):
            panel_mgr.ingest(chunk)
            yield (
                history + [{"role": "user", "content": message}] + panel_mgr.messages,
                session_id,
                _make_langfuse_url(session_id),
            )

        if not panel_mgr.messages:
            # Non-streaming fallback
            result = agent.run(message, reset=False)
            panel_mgr.ingest(gr.ChatMessage(role="assistant", content=str(result)))
            if truncation_notice:
                panel_mgr.append_to_last_text(truncation_notice)
            yield (
                history + [{"role": "user", "content": message}] + panel_mgr.messages,
                session_id,
                _make_langfuse_url(session_id),
            )
        elif truncation_notice:
            panel_mgr.append_to_last_text(truncation_notice)
            yield (
                history + [{"role": "user", "content": message}] + panel_mgr.messages,
                session_id,
                _make_langfuse_url(session_id),
            )

        # --- Post-streaming: extract structured data from agent memory ---
        # Get only the steps produced during THIS turn (not accumulated history).
        _all_steps = list(getattr(agent.memory, "steps", []) or [])
        _new_steps = _all_steps[_steps_before:]

        # Supplement the UI with any tool-call panels that smolagents dropped
        # (it only shows tool_calls[0] per step; parallel calls are invisible).
        panel_mgr.supplement(_new_steps)
        # Emit the final supplemented state so the UI shows all tool calls.
        yield (
            history + [{"role": "user", "content": message}] + panel_mgr.messages,
            session_id,
            _make_langfuse_url(session_id),
        )

        # --- Data logging: capture completed turn asynchronously ---
        _data_logger.log(
            conversation_id=session_id,
            model_id=cfg.model_id,
            app_version=cfg.app_version,
            user_input=message,
            final_answer=panel_mgr.final_answer(),
            agent_steps=[dataclasses.asdict(s) for s in parse_action_steps(_new_steps, _tool_responses)],
            turn_number=len(history) // 2,  # each full turn = user + assistant
            extra={"langfuse_url": _make_langfuse_url(session_id)},
        )

    except MCPToolError as exc:
        # MCPToolError is already converted to a safe "[Tool unavailable: ...]"
        # string by the tool wrapper in mcp_client.py and fed into agent reasoning.
        # If it somehow escapes, surface a generic message and re-raise to the log.
        logger.exception(
            "MCPToolError escaped agent reasoning context — tool='%s' category='%s'.",
            exc.tool_name,
            exc.category,
        )
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

with gr.Blocks(title="AI Assistant", theme=theme, css=_css, fill_height=True) as demo:
    # Per-session state — each browser tab gets an independent session_id (T018)
    session_state = gr.State(value=_new_session_id())

    gr.HTML(_HEADER_HTML)

    chatbot = gr.Chatbot(
        label="Conversation",
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
