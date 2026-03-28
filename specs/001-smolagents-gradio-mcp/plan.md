# Implementation Plan: Interactive AI Assistant (smolagents + Gradio + MCP + Langfuse)

**Branch**: `001-smolagents-gradio-mcp` | **Date**: 2026-03-27 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-smolagents-gradio-mcp/spec.md`

## Summary

Build a publicly accessible AI assistant chat interface on HuggingFace Spaces that uses the
`smolagents` framework for agent orchestration, `Gradio` for the browser UI, MCP servers
(HTTP Streamable transport) for external tool access, and `Langfuse` (via OpenTelemetry
auto-instrumentation) for full interaction tracing and evaluation scoring.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**:
- `smolagents[gradio,telemetry]==1.24.0` — agent orchestration, built-in Gradio UI, OTEL telemetry layer
- `gradio>=5.0` — chat interface (bundled via `smolagents[gradio]`)
- `mcp` — MCP HTTP Streamable client (required by smolagents MCP features)
- `langfuse>=3.0` — tracing dashboard + evaluation scoring API
- `openinference-instrumentation-smolagents` — OTEL auto-instrumentation bridge for smolagents → Langfuse
- `opentelemetry-sdk`, `opentelemetry-exporter-otlp` — OTEL pipeline

**Storage**: In-memory per session only; no database. Conversation history held in smolagents' built-in `agent.memory`.  
**Testing**: `pytest` with unit tests for evaluation logic; integration test for agent+MCP round-trip.  
**Target Platform**: HuggingFace Spaces (Linux container, Gradio SDK).  
**Project Type**: web-service (Gradio app hosted on HF Spaces).  
**Performance Goals**: <30s p95 agent response time per turn (per SC-001).  
**Constraints**: No user auth, single model provider (env-configured), no persistent storage across sessions, two MCP servers supported (FR-003), Langfuse failures MUST NOT crash the app (graceful degradation).  
**Scale/Scope**: Single public HuggingFace Space; concurrent sessions handled by Gradio's built-in session isolation via `gr.State`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Check | Status |
|-----------|-------|--------|
| **I. PoC First** | P1 (chat + MCP tool call) is the PoC — demonstrates core mechanic end-to-end before P2/P3 investment. Research phase validates MCP + smolagents integration before coding. | ✅ PASS |
| **II. MVP Delivery** | P1 stories (FR-001 through FR-004, FR-006, FR-008) constitute a deployable HF Space with working agent chat. P2/P3 (Langfuse tracing, evaluation) are explicitly deferred. | ✅ PASS |
| **III. Speed to Market** | Single-project structure, `smolagents[gradio]` built-in UI, OTEL auto-instrumentation (no hand-rolled tracing). No abstractions added beyond what requirements demand. | ✅ PASS |
| **IV. Decent UX** | `GradioUI` provides loading/streaming states built-in. Error messages are human-readable (FR-011). Empty message guard (edge case) handled at UI layer. | ✅ PASS |
| **No over-engineering** | No custom agent loop, no custom UI components, no repository pattern. `smolagents` handles orchestration; `GradioUI` handles streaming. | ✅ PASS |
| **Dependency hygiene** | All chosen packages are well-maintained HuggingFace and Arize/OpenInference packages with active release histories. Licenses: Apache-2.0 (`smolagents`, `gradio`, `langfuse`), Apache-2.0 (`openinference`). | ✅ PASS |

**Gate Result**: ✅ ALL CHECKS PASS — proceed to Phase 0 research.

*Post-Phase 1 re-check*: No new abstractions or patterns introduced beyond the above. Gate remains ✅ PASS.

## Project Structure

### Documentation (this feature)

```text
specs/001-smolagents-gradio-mcp/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── mcp-tool-interface.md
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
app.py                   # Gradio entry point — initialises agent, wires OTEL, launches GradioUI
requirements.txt         # Pinned dependencies for HF Spaces build
README.md                # HF Spaces config frontmatter + project description
.env.example             # Documentation of required environment variables (never committed with secrets)

src/
├── config.py            # Env config loader — reads all vars via python-dotenv, raises on missing required
├── agent.py             # Agent factory: builds ToolCallingAgent with MCP tools
├── mcp_client.py        # MCPClient setup — reads MCP_SERVER_URL_1/2 from env
├── telemetry.py         # OTEL + Langfuse bootstrap (SmolagentsInstrumentor + get_client)
└── evaluation.py        # Batch evaluation runner: reads traces, scores via langfuse.create_score()

tests/
├── unit/
│   └── test_evaluation.py   # Unit tests for scoring logic
└── integration/
    └── test_agent_mcp.py    # Integration test: agent produces response with MCP tool call
```

**Structure Decision**: Single-project layout (Option 1 variant). No frontend/backend split because `smolagents[gradio]` provides the entire browser UI from Python. `src/` separates concerns (agent, MCP, telemetry, evaluation) without introducing abstraction layers.

## Complexity Tracking

*No constitution violations identified — table not required.*
