# Research: Interactive AI Assistant (smolagents + Gradio + MCP + Langfuse)

**Phase**: 0 — Outline & Research  
**Branch**: `001-smolagents-gradio-mcp`  
**Date**: 2026-03-27

---

## R-001: smolagents + MCP HTTP Streamable Transport

**Question**: Does smolagents support MCP HTTP Streamable transport natively, and what API is used?

**Decision**: Use `MCPClient` from `smolagents` with `transport: "streamable-http"`.

**Rationale**:
- `smolagents==1.24.0` ships native MCP support. No additional adapter library required.
- `MCPClient` is used as a context manager and returns a list of `Tool` objects passed directly to the agent constructor.
- HTTP Streamable (`"streamable-http"`) is the current non-deprecated transport. SSE (`"sse"`) is legacy.
- Multiple MCP servers can be combined by initialising separate `MCPClient` instances and flattening their tool lists.

**Integration pattern**:
```python
from smolagents import MCPClient, ToolCallingAgent

with MCPClient({"url": MCP_SERVER_URL, "transport": "streamable-http"}) as mcp_tools:
    agent = ToolCallingAgent(tools=list(mcp_tools), model=model)
```

**Alternatives considered**:
- `ToolCollection.from_mcp()` — rejected because it requires `trust_remote_code=True`, which violates the security posture for a public-facing app.
- Building a custom HTTP client for MCP — rejected (over-engineering; violates Principle III).

**Known constraints**:
- MCP server must be reachable via HTTPS from HF Spaces runtime (confirmed as an assumption in spec).
- `structured_output=True` is opt-in in v1.24.0; leave at default `False` for now.

---

## R-002: Langfuse Integration with smolagents

**Question**: How do you instrument smolagents runs to produce Langfuse traces?

**Decision**: Use OpenTelemetry auto-instrumentation via `openinference-instrumentation-smolagents` + Langfuse v3 OTEL exporter.

**Rationale**:
- smolagents exposes an OTEL-compatible telemetry layer (available via `smolagents[telemetry]` extra). This is the officially supported path.
- `SmolagentsInstrumentor().instrument()` patches all agent `run()` calls globally — zero per-agent code changes required.
- Langfuse v3 accepts OTLP traces natively; no separate Langfuse Python SDK wrapper around each agent call is needed.
- Session and user metadata is injected via `langfuse.update_current_trace()` within the Gradio request context.

**Integration pattern**:
```python
# telemetry.py — called once at app startup
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from langfuse import get_client

def bootstrap_telemetry():
    SmolagentsInstrumentor().instrument()
    langfuse = get_client()
    langfuse.auth_check()   # raises if env vars are missing
    return langfuse
```

**Required env vars**: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (defaults to `https://cloud.langfuse.com`).

**Graceful degradation**: If `langfuse.auth_check()` raises (e.g., Langfuse unreachable), the app catches the exception, logs locally, and continues without telemetry. Agent runs are never blocked by telemetry failures (FR requirement).

**Alternatives considered**:
- `step_callbacks` on `MultiStepAgent` — rejected because the Langfuse integration does not use this hook; OTEL auto-instrumentation is the documented approach.
- Langfuse SDK v2 `Langfuse()` constructor — rejected in favour of v3 `get_client()` pattern (current API).

---

## R-003: Gradio Chat UI with Streaming

**Question**: What is the lowest-effort way to build a streaming, multi-turn chat UI in Gradio backed by a smolagents agent?

**Decision**: Use `smolagents.GradioUI` (built-in, part of `smolagents[gradio]`).

**Rationale**:
- `GradioUI` wraps `gradio.ChatInterface` and uses `agent.run(task, stream=True)` internally — streaming is automatic.
- Multi-turn history, loading states, and file upload are handled by the component.
- No custom Gradio component code required — satisfies "no gold-plating" (Principle II/III).
- Session isolation is achieved via `reset_agent_memory=False` (default) combined with `gr.State` managed by `GradioUI`.

**Integration pattern**:
```python
from smolagents import GradioUI
GradioUI(agent).launch()
```

**Alternatives considered**:
- `stream_to_gradio()` with a custom `gr.ChatInterface` — rejected for MVP; adds complexity with no additional user value at P1.
- Raw `gradio.Blocks` with manual streaming — rejected (over-engineering).

**Edge case handling**:
- Empty message: `GradioUI` does not submit empty strings; the Gradio `ChatInterface` component prevents empty submission at the UI level. A guard in `app.py` is added as a belt-and-suspenders check.
- Rapid-fire messages: `GradioUI` disables the submit button during agent execution (Gradio `ChatInterface` default behaviour).

---

## R-004: HuggingFace Spaces Deployment

**Question**: What files and configuration are required for a Gradio app on HuggingFace Spaces?

**Decision**: Standard HF Spaces layout: `app.py` + `requirements.txt` + `README.md` with YAML frontmatter.

**Rationale**:
- HF Spaces automatically installs `requirements.txt` and runs `app.py` at startup.
- The YAML frontmatter in `README.md` configures the Space (SDK, Python version, visibility).
- Environment variables (MCP URLs, Langfuse keys, model API keys) are set via the HF Spaces "Secrets" panel — never committed to source.

**README.md frontmatter**:
```yaml
---
title: AI Assistant (smolagents + MCP)
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0"
python_version: "3.11"
app_file: app.py
pinned: false
---
```

**Alternatives considered**:
- `agent.push_to_hub()` — useful for model sharing but not required for a Gradio app deployment; standard HF Spaces git push is sufficient.
- Docker SDK — adds complexity without benefit for a pure Python Gradio app.

---

## R-005: Langfuse Evaluation Scoring

**Question**: How do evaluation scores get attached to Langfuse traces?

**Decision**: Use `langfuse.create_score()` (v3 SDK) in a standalone batch evaluation script.

**Rationale**:
- `create_score()` accepts `trace_id` + `name` + `value` (numeric or categorical) — maps directly to spec requirement FR-007.
- The evaluation script (`src/evaluation.py`) fetches traces from Langfuse, runs scoring logic, and posts scores back.
- Scores are visible in the Langfuse UI alongside their traces (satisfies SC-004, SC-006).

**Integration pattern**:
```python
langfuse = get_client()
langfuse.create_score(
    trace_id=trace.id,
    name="relevance",
    value=0.85,
    data_type="NUMERIC",
    comment="Response addressed all parts of the user query."
)
```

**Alternatives considered**:
- Inline real-time scoring during agent run — rejected (spec explicitly states P3 is batch, not real-time).
- Manual score entry in Langfuse UI — acceptable for ad hoc review but not automatable; rejected as the primary path.

---

## R-006: Model Provider Configuration

**Question**: Which model provider is used, and how is it configured?

**Decision**: Model provider is fully env-configured, with no hard-coded default. `LiteLLMModel` from smolagents is used as the model adapter.

**Rationale**:
- Spec assumption: "A single AI model provider (configured via environment variable)".
- `LiteLLMModel` supports OpenAI, Anthropic, Mistral, and others via a single `model_id` string — allows switching provider without code changes.
- `MODEL_ID` and `MODEL_API_KEY` env vars control the model; `LiteLLMModel` reads these at startup.

**Alternatives considered**:
- `InferenceClientModel` (HF Inference API) — valid for HF-hosted models but less flexible; provider switching requires code changes.
- Hard-coded `gpt-4o` default — violates FR-008 / "configurable via environment variables".

---

## Summary of Resolved Unknowns

| ID | Unknown | Resolution |
|----|---------|------------|
| R-001 | MCP HTTP Streamable integration API | `MCPClient` with `transport: "streamable-http"` |
| R-002 | Langfuse tracing integration mechanism | OTEL auto-instrumentation (`SmolagentsInstrumentor`) |
| R-003 | Streaming Gradio chat implementation | `smolagents.GradioUI` (built-in) |
| R-004 | HF Spaces deployment requirements | `app.py` + `requirements.txt` + `README.md` YAML |
| R-005 | Evaluation scoring API | `langfuse.create_score()` (v3 SDK) |
| R-006 | Model provider configuration | `LiteLLMModel` + env vars |

All NEEDS CLARIFICATION items resolved. Phase 1 may proceed.
