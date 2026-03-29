"""
src/data_logger.py — Interaction data logger.

Captures the full smolagents conversation: user inputs, agent reasoning steps,
all tool invocations (name + structured arguments), tool observations,
per-step timing and token counts, and the final answer.

Entries are validated with Pydantic, written to a local JSON Lines file per
conversation turn, and periodically synced to a HuggingFace Hub dataset.

Architecture
------------
* ``DataLogger``  — owns its own queue, threads, and lock.  Instantiate once;
                    call ``start()``, then ``log()``, then ``shutdown()``.
                    Also usable as a context manager.

JSONL structure: one line per turn, all turns sharing the same
``conversation_id`` form a complete conversation (group + order by
``turn_number``).

The app MUST NOT crash if HF Hub is unreachable — all upload errors are
swallowed and logged at WARNING level.
"""

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator

from src.timer import Timer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """Model and deployment metadata included in every log entry."""

    model_id: str = Field(
        ..., description="LiteLLM model identifier, e.g. 'openai/gpt-4o'"
    )
    app_version: str = Field("dev", description="APP_VERSION env-var value")

    model_config = {"extra": "forbid"}


class ToolInvocation(BaseModel):
    """One tool call within an agent step.

    Maps directly to smolagents ``ToolCall`` dataclass fields, with the
    addition of ``response`` which is captured from the corresponding
    ``ToolOutput.observation`` yielded during streaming.
    """

    id: str = Field("", description="smolagents ToolCall.id")
    tool_name: str = Field(..., description="Name of the MCP tool invoked")
    arguments: dict[str, Any] | str = Field(
        default_factory=dict,
        description="Structured arguments dict (as returned by the LLM); "
        "falls back to raw string if the value is not a dict",
    )
    response: str = Field(
        "",
        description="Individual tool response / observation captured from "
        "ToolOutput.observation during streaming",
    )

    model_config = {"extra": "forbid"}


class ActionStepLog(BaseModel):
    """Full record of one smolagents ActionStep (one LLM call + tool executions).

    Fields map 1-to-1 with ``smolagents.memory.ActionStep`` attributes:
    ``step_number``, ``model_output``, ``tool_calls``, ``observations``,
    ``timing``, ``token_usage``, and ``error``.
    """

    step_number: int = Field(..., ge=1, description="1-based step index")
    model_output: str = Field(
        "", description="Raw LLM output / reasoning for this step"
    )
    tool_invocations: list[ToolInvocation] = Field(
        default_factory=list,
        description="All tool calls made in this step (not just the first)",
    )
    observations: str = Field(
        "", description="Combined tool output / observation string for the step"
    )
    duration_seconds: float | None = Field(
        None, description="Wall-clock duration from ActionStep.timing.duration"
    )
    input_tokens: int | None = Field(
        None, description="LLM input tokens from ActionStep.token_usage"
    )
    output_tokens: int | None = Field(
        None, description="LLM output tokens from ActionStep.token_usage"
    )
    total_tokens: int | None = Field(
        None, description="Total tokens (input + output)"
    )
    error: str | None = Field(
        None, description="Error string if the step raised an AgentError"
    )

    model_config = {"extra": "forbid"}


class ConversationTurnLog(BaseModel):
    """Complete record of one user↔agent interaction turn.

    Each turn is an independent JSONL line.  Turns belonging to the same
    multi-turn conversation share the same ``conversation_id`` and are
    ordered by ``turn_number``.
    """

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique log entry ID (UUID4)",
    )
    conversation_id: str = Field(
        ...,
        description="Gradio per-tab session UUID — groups all turns of one conversation",
    )
    turn_number: int = Field(
        0, ge=0, description="0-based position of this turn in the conversation"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp when this turn was logged",
    )
    model: ModelInfo = Field(..., description="Model and application metadata")
    user_input: str = Field(..., description="Verbatim user message for this turn")
    agent_steps: list[ActionStepLog] = Field(
        default_factory=list,
        description="All smolagents ActionSteps executed for this turn, in order",
    )
    final_answer: str = Field(
        "", description="Agent's final answer text for this turn"
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary extra metadata (e.g. Langfuse URL)"
    )

    model_config = {"extra": "forbid"}

    @field_validator("user_input")
    @classmethod
    def _input_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_input must not be empty")
        return v


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, entry: ConversationTurnLog) -> None:
    """Append a validated entry to the local JSONL file.

    Callers are responsible for acquiring any necessary write lock before
    calling this function.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = entry.model_dump_json() + "\n"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line)


def _upload_to_hub(
    local_path: Path,
    repo_id: str,
    hf_token: str,
    repo_filename: str = "interactions.jsonl",
) -> bool:
    """
    Upload *local_path* to the HuggingFace Hub dataset *repo_id*.

    Returns True on success, False on any error (errors are logged, not raised).
    """
    try:
        from huggingface_hub import HfApi

        t = Timer("hf_upload")
        t.start()

        api = HfApi(token=hf_token)

        # Ensure the dataset repo exists; create private repo if not found
        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception:
            logger.info("data_logger: creating new dataset repo '%s'", repo_id)
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True)

        t.add_step("repo check")

        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_filename,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Update interactions log {datetime.now(timezone.utc).isoformat()}",
        )
        t.add_step("upload")
        t.end()
        logger.info("data_logger: synced to HF Hub. %s", t.formatted_result())
        return True

    except Exception as exc:  # never crash the app
        logger.warning("data_logger: HF Hub upload failed — %s", exc)
        return False


# ---------------------------------------------------------------------------
# DataLogger
# ---------------------------------------------------------------------------

#: Sentinel value placed in the log queue to signal the worker to exit.
_STOP_SENTINEL = None


class DataLogger:
    """Non-blocking interaction logger with optional HuggingFace Hub sync.

    Owns its own queue, write lock, and daemon threads so multiple independent
    instances can coexist in the same process (e.g. one per integration test).

    Usage
    -----
    .. code-block:: python

        dl = DataLogger(log_dir="logs", repo_id="user/ds", hf_token="hf_...")
        dl.start()           # spawn worker threads; returns self for chaining

        dl.log(...)          # non-blocking enqueue

        dl.shutdown(         # drain queue, stop sync, optional final upload
            timeout=10,
            final_sync=True,
        )

    Or as a context manager::

        with DataLogger(...) as dl:
            dl.log(...)

    Parameters
    ----------
    log_dir:
        Directory for the local JSONL file.  Created automatically.
    repo_id:
        HuggingFace Hub dataset repo ID, e.g. ``"username/chat-logs"``.
        If *None* or empty, HF sync is disabled.
    hf_token:
        HuggingFace write token.  Required when *repo_id* is set.
    sync_interval:
        How often (seconds) to push updates to HF Hub.  Default 300 (5 min).
        Floored at 30 seconds.
    _upload_fn:
        Injectable upload callable used in tests.  Defaults to the real
        ``_upload_to_hub`` function.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        repo_id: str | None = None,
        hf_token: str | None = None,
        sync_interval: int = 300,
        _upload_fn: Callable | None = None,
    ) -> None:
        self._jsonl_path = Path(log_dir) / "interactions.jsonl"
        self._repo_id = repo_id or None
        self._hf_token = hf_token or None
        self._sync_interval = max(30, sync_interval)
        self._upload_fn: Callable = _upload_fn or _upload_to_hub

        self._queue: Queue = Queue()
        self._write_lock = threading.Lock()
        self._sync_stop = threading.Event()
        self._started = False
        self._log_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "DataLogger":
        """Spawn worker threads and begin accepting log entries.

        Returns *self* so calls can be chained: ``DataLogger(...).start()``.

        Raises
        ------
        RuntimeError
            If ``start()`` is called a second time on the same instance.
        """
        if self._started:
            raise RuntimeError(
                "DataLogger.start() called twice on the same instance. "
                "Create a new DataLogger instance instead."
            )
        self._started = True

        self._log_thread = threading.Thread(
            target=self._log_worker, name="data-log-worker", daemon=True
        )
        self._log_thread.start()

        if self._repo_id and self._hf_token:
            self._sync_thread = threading.Thread(
                target=self._sync_worker, name="data-sync-worker", daemon=True
            )
            self._sync_thread.start()
            logger.info(
                "data_logger: HF Hub sync enabled — repo=%s interval=%ds",
                self._repo_id,
                self._sync_interval,
            )
        else:
            logger.info(
                "data_logger: HF Hub sync disabled "
                "(HF_TOKEN / HF_DATASET_REPO_ID not set)"
            )

        logger.info("data_logger: logging to %s", self._jsonl_path)
        return self

    def shutdown(self, timeout: float = 10.0, final_sync: bool = False) -> None:
        """Drain the log queue, stop worker threads, and optionally upload.

        Blocks until the log worker exits (up to *timeout* seconds) and the
        sync worker is signalled to stop.  Safe to call multiple times; second
        and subsequent calls are no-ops.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait for the log worker to drain and exit.
        final_sync:
            When ``True`` and HF Hub credentials are configured, perform one
            final upload after the queue drains.
        """
        if not self._started:
            return

        # Signal log worker to exit after draining all pending entries.
        self._queue.put(_STOP_SENTINEL)
        if self._log_thread is not None:
            self._log_thread.join(timeout=timeout)

        # Signal sync worker to stop its timer loop.
        self._sync_stop.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=2.0)

        if (
            final_sync
            and self._jsonl_path.exists()
            and self._repo_id
            and self._hf_token
        ):
            self._upload_fn(self._jsonl_path, self._repo_id, self._hf_token)

        # Mark as stopped so log() calls after shutdown are silently dropped.
        self._started = False

    def __enter__(self) -> "DataLogger":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Public logging API
    # ------------------------------------------------------------------

    def log(
        self,
        conversation_id: str,
        model_id: str,
        app_version: str,
        user_input: str,
        final_answer: str,
        agent_steps: list[dict],
        turn_number: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Enqueue a conversation turn for async logging.  Non-blocking.

        Silently drops the entry (with a WARNING) if called before ``start()``
        or after ``shutdown()``, or if Pydantic validation fails.

        Parameters
        ----------
        conversation_id:
            Gradio per-tab session UUID (constant for the whole conversation).
        model_id:
            LiteLLM model identifier used for this turn.
        app_version:
            Application version string from config.
        user_input:
            The raw user message for this turn.
        final_answer:
            The agent's final plain-text answer.
        agent_steps:
            List of dicts matching ``ActionStepLog`` schema.
        turn_number:
            0-based position of this turn in the conversation.
        extra:
            Arbitrary extra key/value metadata to attach to the log entry.
        """
        if not self._started:
            logger.warning(
                "data_logger: log() called before start() or after shutdown() "
                "— entry discarded"
            )
            return
        try:
            parsed_steps = [ActionStepLog(**s) for s in agent_steps]
            entry = ConversationTurnLog(
                conversation_id=conversation_id,
                model=ModelInfo(model_id=model_id, app_version=app_version),
                user_input=user_input,
                final_answer=final_answer,
                agent_steps=parsed_steps,
                turn_number=turn_number,
                extra=extra or {},
            )
            self._queue.put(entry)
        except Exception as exc:
            logger.warning("data_logger: failed to enqueue entry — %s", exc)

    # ------------------------------------------------------------------
    # Internal worker methods
    # ------------------------------------------------------------------

    def _log_worker(self) -> None:
        """Drain the queue: validate each entry and write it to the JSONL file."""
        while True:
            entry = self._queue.get()
            if entry is _STOP_SENTINEL:
                self._queue.task_done()
                break
            try:
                if isinstance(entry, ConversationTurnLog):
                    with self._write_lock:
                        _write_jsonl(self._jsonl_path, entry)
                    logger.debug(
                        "data_logger: wrote conversation_id=%s turn=%d log_id=%s",
                        entry.conversation_id,
                        entry.turn_number,
                        entry.log_id,
                    )
            except Exception as exc:
                logger.error("data_logger: write failed — %s", exc)
            finally:
                self._queue.task_done()

    def _sync_worker(self) -> None:
        """Periodically push the local JSONL to HuggingFace Hub."""
        while not self._sync_stop.wait(timeout=self._sync_interval):
            if (
                self._jsonl_path.exists()
                and self._repo_id
                and self._hf_token
            ):
                self._upload_fn(self._jsonl_path, self._repo_id, self._hf_token)

