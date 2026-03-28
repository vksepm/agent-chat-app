# Data Model: Interactive AI Assistant (smolagents + Gradio + MCP + Langfuse)

**Phase**: 1 — Design & Contracts  
**Branch**: `001-smolagents-gradio-mcp`  
**Date**: 2026-03-27

---

## Overview

This application is stateless at the persistence layer (no database). All runtime state lives
in-memory within a Gradio session. The entities below describe the logical domain model — they
map to Python dataclasses/dicts and smolagents built-in types, not database tables.

---

## Entity: ConversationSession

**Description**: One user's browser session from first message to tab close. Holds the ordered
history of user/assistant turns. Managed implicitly by smolagents `agent.memory` and Gradio
`gr.State`.

**Fields**:

| Field | Type | Notes |
|-------|------|-------|
| `session_id` | `str` (UUID4) | Generated on first message; used as Langfuse `session_id` |
| `history` | `list[ChatMessage]` | Gradio `gr.ChatInterface` message list; auto-managed by `GradioUI` |
| `agent_memory` | `smolagents.AgentMemory` | Managed internally by `ToolCallingAgent`; not exposed directly |
| `created_at` | `datetime` | Timestamp of first turn in this session |

**Lifecycle**: Created on first message; destroyed when Gradio session ends (tab close or timeout). Not persisted.

**Validation rules**:
- `session_id` MUST be a valid UUID4 string.
- Session MUST NOT be shared across different browser tabs (Gradio `gr.State` enforces this).

---

## Entity: AgentTurn

**Description**: One complete round-trip: a user message, zero or more tool calls, and the final
assistant response. Corresponds 1:1 with a Langfuse `Trace`.

**Fields**:

| Field | Type | Notes |
|-------|------|-------|
| `turn_id` | `str` (UUID4) | Maps to Langfuse `trace_id`; generated before agent.run() |
| `session_id` | `str` | FK → `ConversationSession.session_id` |
| `user_message` | `str` | Raw user input text; MUST NOT be empty |
| `assistant_response` | `str` | Final agent output; may be empty only on hard error |
| `tool_calls` | `list[ToolCall]` | Zero or more tool invocations within this turn |
| `started_at` | `datetime` | Timestamp when agent.run() was invoked |
| `completed_at` | `datetime` | Timestamp when agent.run() returned |
| `latency_ms` | `int` | `(completed_at - started_at)` in milliseconds |
| `error` | `str \| None` | Human-readable error message if turn failed; None on success |

**Validation rules**:
- `user_message` MUST NOT be empty or whitespace-only (enforced at UI layer before agent.run()).
- `assistant_response` MUST be a non-empty string when `error` is `None`.
- `latency_ms` MUST be ≥ 0.

**State transitions**:
```
[PENDING] → agent.run() starts → [IN_PROGRESS] → response received → [COMPLETED]
                                                → exception raised   → [FAILED]
```

---

## Entity: ToolCall

**Description**: A single invocation of one MCP-exposed tool within an `AgentTurn`. Captured by
the OTEL auto-instrumentation as a child span of the `Trace`.

**Fields**:

| Field | Type | Notes |
|-------|------|-------|
| `tool_name` | `str` | Name of the MCP tool as declared by the MCP server |
| `tool_input` | `dict` | JSON-serialisable arguments passed to the tool |
| `tool_output` | `str \| dict \| None` | Raw tool result; `None` if tool call failed |
| `error` | `str \| None` | Error message if tool call failed; `None` on success |
| `latency_ms` | `int` | Duration of the tool call in milliseconds |

**Validation rules**:
- `tool_name` MUST match a tool advertised by the connected MCP server.
- `tool_input` MUST be JSON-serialisable.
- Either `tool_output` or `error` MUST be non-None (not both None simultaneously).

---

## Entity: MCPTool

**Description**: A capability exposed by a connected MCP server. Discovered at agent startup via
`MCPClient.get_tools()`. Represented internally as a `smolagents.Tool` instance.

**Fields** (as seen from smolagents `Tool`):

| Field | Type | Notes |
|-------|------|-------|
| `name` | `str` | Unique tool identifier within the agent's tool list |
| `description` | `str` | Natural-language description used by the LLM for tool selection |
| `inputs` | `dict` | JSON Schema for input parameters |
| `output_type` | `str` | smolagents output type descriptor (e.g., `"string"`, `"any"`) |

**Source**: Populated at startup from `MCPClient.__enter__()`. Refreshed only on app restart (no hot-reload of MCP tool catalogue).

---

## Entity: Trace

**Description**: The Langfuse record of one `AgentTurn`. Contains spans for each reasoning step
and tool call, with timing and payload data. Created and managed automatically by
`SmolagentsInstrumentor` via OTEL.

**Fields** (Langfuse trace object):

| Field | Type | Notes |
|-------|------|-------|
| `trace_id` | `str` | Langfuse trace ID; correlates with `AgentTurn.turn_id` via OTEL context |
| `session_id` | `str` | Set via `langfuse.update_current_trace(session_id=...)` |
| `user_id` | `str \| None` | Optional; not used in v1 (no auth) |
| `input` | `str` | User message |
| `output` | `str` | Final assistant response |
| `metadata` | `dict` | Arbitrary key-value bag: model ID, MCP server URL, app version |
| `spans` | `list[Span]` | Child spans: one per reasoning step + one per tool call |
| `scores` | `list[EvaluationRecord]` | Attached after evaluation workflow runs |

---

## Entity: EvaluationRecord

**Description**: A scored assessment of an `AgentTurn`/`Trace`. Produced by the batch evaluation
workflow (`src/evaluation.py`). Stored in Langfuse via `langfuse.create_score()`.

**Fields**:

| Field | Type | Notes |
|-------|------|-------|
| `score_id` | `str` (UUID4) | Idempotency key; prevents duplicate scores on re-run |
| `trace_id` | `str` | FK → `Trace.trace_id` |
| `name` | `str` | Quality criterion label, e.g., `"relevance"`, `"correctness"`, `"tool_efficiency"` |
| `value` | `float` | Numeric score in range [0.0, 1.0] |
| `data_type` | `str` | Always `"NUMERIC"` for automated scores |
| `comment` | `str \| None` | Optional explanation of the score |
| `evaluator` | `str` | `"automated"` for script-run scores; `"human"` for manual |
| `evaluated_at` | `datetime` | Timestamp when score was created |

**Validation rules**:
- `value` MUST be in [0.0, 1.0] for NUMERIC scores.
- `name` MUST be one of the registered quality criteria (initially: `relevance`, `correctness`, `tool_efficiency`).
- `score_id` MUST be deterministic per `(trace_id, name, run_id)` to ensure idempotent re-runs.

---

## Environment Configuration (not an entity — operational context)

All sensitive values come from environment variables or HF Spaces Secrets. No defaults for
secret values.

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_ID` | Yes | LiteLLM model ID, e.g., `"openai/gpt-4o"` |
| `MODEL_API_KEY` | Yes | API key for the model provider |
| `MCP_SERVER_URL_1` | Yes | HTTP Streamable MCP server base URL (first server) |
| `MCP_SERVER_URL_2` | Yes | HTTP Streamable MCP server base URL (second server) |
| `LANGFUSE_PUBLIC_KEY` | Yes (P2+) | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | Yes (P2+) | Langfuse project secret key |
| `LANGFUSE_HOST` | No | Defaults to `https://cloud.langfuse.com` |
| `LANGFUSE_PROJECT_ID` | No | Optional; Langfuse v3 derives project from public key but can be set explicitly for multi-project deployments |
| `APP_VERSION` | No | Injected into Langfuse trace `metadata`; defaults to `"dev"` |

---

## Relationships Summary

```
ConversationSession (1) ──── (N) AgentTurn
AgentTurn           (1) ──── (N) ToolCall
AgentTurn           (1) ──── (1) Trace         [via OTEL trace_id correlation]
Trace               (1) ──── (N) EvaluationRecord
MCPTool             (N) ──── (N) AgentTurn      [agent selects tools per turn]
```
