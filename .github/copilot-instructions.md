# speckit-chat-app — Copilot Instructions

## What This App Does

Interactive AI chat agent deployed on HuggingFace Spaces. Users chat with a `smolagents` `ToolCallingAgent` that can call external tools via MCP (Model Context Protocol) servers over HTTP Streamable transport. All interactions are traced via Langfuse + OpenTelemetry. A separate batch evaluation workflow scores past traces using LLM-as-judge.

## Commands

```bash
# Run app locally
python app.py                   # Gradio UI at http://localhost:7860

# Tests
pytest tests/unit/              # Fast, no external deps
pytest tests/integration/       # Requires live MCP + model credentials

# Single test
pytest tests/unit/test_evaluation.py::TestClamp::test_clamp_in_range -v

# Lint
ruff check .
```

Integration tests use `@pytest.mark.skipif` guards when credentials are absent — they can be run by providing env vars inline:
```bash
MCP_SERVER_URL_1=https://... MODEL_API_KEY=sk-... pytest tests/integration/ -v
```

Evaluation CLI (batch LLM-as-judge scoring against Langfuse traces):
```bash
python -m src.evaluation --from 2026-03-01 --to 2026-03-31 [--run-id my-eval-v2]
```

## Architecture

```
app.py  (entry point)
├── src/config.py       → Config dataclass, loads .env, raises EnvironmentError if required vars missing
├── src/mcp_client.py   → Opens MCPClient connections; wraps all tool errors as MCPToolError strings
├── src/agent.py        → build_agent() → ToolCallingAgent with LiteLLMModel + MCP tools injected
├── src/telemetry.py    → bootstrap_telemetry() → SmolagentsInstrumentor (OTEL) + Langfuse client
└── src/evaluation.py   → Batch scoring: fetch Langfuse traces → LLM judge → post scores back
```

**Key design decisions:**
- `reset_memory_between_tasks=False` — multi-turn context is preserved within a browser session
- Per-session UUID in `gr.State` ties Gradio history to Langfuse trace metadata
- `ExitStack` manages MCP client lifetimes; one server failing at startup doesn't prevent other tools from loading
- Telemetry failures are swallowed silently — the app MUST NOT crash if Langfuse is unreachable
- History capped at 20 turns (`MAX_HISTORY_TURNS`) to guard LLM context window

**Data flow for one chat turn:**
1. Gradio `chat()` handler validates message (empty → early return)
2. Langfuse metadata injected (`session_id`, `app_version`)
3. `agent.run(message)` streamed via `stream_to_gradio()`
4. MCP tool errors surface to agent as `"[Tool unavailable: ...]"` strings, not exceptions
5. OTEL auto-instrumentation posts the completed trace to Langfuse

## Environment Variables

| Variable | Required | Notes |
|---|---|---|
| `MODEL_ID` | ✅ | LiteLLM model string, e.g. `openai/gpt-4o` |
| `MODEL_API_KEY` | ✅ | API key for model provider |
| `MCP_SERVER_URL_1` | ✅ | First MCP server (HTTPS, HTTP Streamable) |
| `MCP_SERVER_URL_2` | ✅ | Second MCP server |
| `LANGFUSE_PUBLIC_KEY` | ✅ | Langfuse auth |
| `LANGFUSE_SECRET_KEY` | ✅ | Langfuse auth |
| `LANGFUSE_HOST` | ❌ | Defaults to `https://cloud.langfuse.com` |
| `LANGFUSE_PROJECT_ID` | ❌ | Enables deep-link URLs in session info panel |
| `APP_VERSION` | ❌ | Injected into trace metadata; defaults to `dev` |

Copy `.env.example` → `.env` for local dev. On HuggingFace Spaces, set these as Space Secrets.

## Key Conventions

- **Config validation**: `src/config.py` uses a `_require()` helper — any missing required env var raises `EnvironmentError` at startup, not at first use
- **Private helpers**: prefix with `_` (e.g. `_make_langfuse_url()`, `_require()`)
- **Docstrings**: numpy-style throughout `src/`
- **MCP errors**: always converted to safe strings before reaching the agent or user; never let raw exceptions propagate to the UI
- **Evaluation idempotency**: score IDs are UUID5 (deterministic) — re-running evaluation on the same traces won't duplicate scores
- **No database**: everything is in-memory; `agent.memory` is the only state store

## Project Principles (from constitution.md)

- **PoC First**: validate the core mechanic end-to-end before production-quality build
- **MVP Delivery**: P1 stories alone must constitute a shippable product; no scope creep before P1 ships
- **Speed to Market**: bias toward action; YAGNI applies; technical debt incurred for speed must be logged as a backlog item
- **Decent UX**: loading/error/empty states required on every P1 story; functional clarity required at MVP

## specs/ Directory

`specs/001-smolagents-gradio-mcp/` contains the feature specification, plan, data model, task breakdown, and contracts for the current feature. These are living documents — consult them for design intent and accepted contracts (e.g. `contracts/mcp-tool-interface.md`).
