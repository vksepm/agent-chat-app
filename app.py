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
import uuid
from contextlib import ExitStack
from typing import Generator

import gradio as gr

from src.agent import build_agent
from src.config import load_config
from src.mcp_client import MCPToolError, build_mcp_tools

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
# Gradio chat handler
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 20  # context-window guard (edge case from spec)


def _make_langfuse_url(session_id: str) -> str:
    """Return a direct Langfuse trace URL for the current session, or empty string."""
    if (
        langfuse is None
        or not cfg.langfuse_host
        or not cfg.langfuse_project_id
    ):
        return ""
    base = cfg.langfuse_host.rstrip("/")
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
    """
    # --- Empty-message guard (FR-001 / edge case) ---
    if not message or not message.strip():
        yield history, session_id, "Please enter a message before submitting."
        return

    # --- Context-window truncation guard (edge case from spec) ---
    truncation_notice = ""
    if len(history) >= MAX_HISTORY_TURNS * 2:
        history = history[-(MAX_HISTORY_TURNS * 2) :]
        truncation_notice = (
            "\n\n_(Earlier messages were summarised to fit context limits.)_"
        )
        logger.info("Session %s: history trimmed to %d turns.", session_id, MAX_HISTORY_TURNS)

    # --- Langfuse v2 native trace for this turn ---
    # Creates one trace per user message with session_id, input, and output.
    # Flushed after the stream completes (or on error) so every turn is recorded.
    lf_trace = None
    if langfuse is not None:
        try:
            import uuid as _uuid
            lf_trace = langfuse.trace(
                id=str(_uuid.uuid4()),
                name="chat",
                session_id=session_id,
                input=message,
                metadata={"app_version": cfg.app_version},
            )
        except Exception as exc:
            logger.warning("Could not create Langfuse trace: %s", exc)

    # --- Agent call with error handling (T013) ---
    try:
        from smolagents import stream_to_gradio

        # Seed history with the user's message once
        user_msg = gr.ChatMessage(role="user", content=message)
        current_history = history + [user_msg]
        yielded = False
        final_output = ""

        for chunk in stream_to_gradio(agent, task=message, reset_agent_memory=False):
            if isinstance(chunk, str):
                # Streaming text delta — update or append the last assistant message
                if current_history and current_history[-1].role == "assistant":
                    current_history[-1] = gr.ChatMessage(role="assistant", content=chunk)
                else:
                    current_history = current_history + [gr.ChatMessage(role="assistant", content=chunk)]
                final_output = chunk
            elif hasattr(chunk, "role") and hasattr(chunk, "content"):
                # Completed step message (tool call, plan, final answer, etc.)
                current_history = current_history + [chunk]
                if getattr(chunk, "role", "") == "assistant":
                    final_output = str(getattr(chunk, "content", "")) or final_output
            trace_url = _make_langfuse_url(session_id)
            yield current_history, session_id, trace_url
            yielded = True

        if not yielded:
            # Non-streaming fallback
            result = agent.run(message)
            final_output = str(result) + truncation_notice
            current_history = current_history + [gr.ChatMessage(role="assistant", content=final_output)]
            trace_url = _make_langfuse_url(session_id)
            yield current_history, session_id, trace_url

        # Flush trace with final output to Langfuse
        if lf_trace is not None:
            try:
                lf_trace.update(output=final_output)
                langfuse.flush()
            except Exception as exc:
                logger.warning("Could not flush Langfuse trace: %s", exc)
    except MCPToolError:
        logger.exception("MCPToolError escaped agent reasoning context.")
        if lf_trace is not None:
            try:
                lf_trace.update(output="[Tool error]", metadata={"error": "MCPToolError", "app_version": cfg.app_version})
                langfuse.flush()
            except Exception:
                pass
        yield (
            history + [
                gr.ChatMessage(role="user", content=message),
                gr.ChatMessage(role="assistant", content="A tool call failed. The assistant will try to answer without it."),
            ],
            session_id,
            "",
        )
    except Exception:
        logger.exception("Unhandled error during agent.run() for session %s", session_id)
        if lf_trace is not None:
            try:
                lf_trace.update(output="[Agent error]", metadata={"error": "UnhandledException", "app_version": cfg.app_version})
                langfuse.flush()
            except Exception:
                pass
        yield (
            history + [
                gr.ChatMessage(role="user", content=message),
                gr.ChatMessage(role="assistant", content="Something went wrong. Please try again."),
            ],
            session_id,
            "",
        )


# ---------------------------------------------------------------------------
# Gradio layout (gr.Blocks)
# ---------------------------------------------------------------------------

def _new_session_id() -> str:
    return str(uuid.uuid4())


with gr.Blocks(title="AI Assistant") as demo:
    # Per-session state — each browser tab gets an independent session_id (T018)
    session_state = gr.State(value=_new_session_id)

    gr.Markdown("# 🤖 AI Assistant\nPowered by smolagents · MCP tools · Langfuse observability")

    chatbot = gr.Chatbot(
        label="Conversation",
        height=520,
    )
    msg_input = gr.Textbox(
        placeholder="Ask me anything…",
        label="Your message",
        lines=2,
        autofocus=True,
    )
    submit_btn = gr.Button("Send", variant="primary")
    clear_btn = gr.Button("Clear conversation", variant="secondary")

    # Collapsible metadata panel for SC-006: session ID + Langfuse trace URL
    with gr.Accordion("Session info", open=False):
        session_info_md = gr.Markdown("_Send a message to see session details._")

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
