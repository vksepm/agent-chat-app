# Tasks: Interactive AI Assistant (smolagents + Gradio + MCP + Langfuse)

**Input**: Design documents from `/specs/001-smolagents-gradio-mcp/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/mcp-tool-interface.md ✅, quickstart.md ✅

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create repository structure, dependency manifest, and HuggingFace Spaces config so the project is immediately deployable (even before any agent logic exists).

- [X] T001 Create project file layout: `app.py`, `src/__init__.py`, `src/agent.py`, `src/mcp_client.py`, `src/telemetry.py`, `src/evaluation.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/integration/__init__.py`
- [X] T002 Create `requirements.txt` with pinned dependencies: `smolagents[gradio,telemetry]==1.24.0`, `mcp`, `langfuse>=3.0`, `openinference-instrumentation-smolagents`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `python-dotenv`, `pytest`
- [X] T003 [P] Create `.env.example` documenting all required env vars: `MODEL_ID`, `MODEL_API_KEY`, `MCP_SERVER_URL_1`, `MCP_SERVER_URL_2`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `LANGFUSE_PROJECT_ID` (optional), `APP_VERSION`
- [X] T004 [P] Create `README.md` with HuggingFace Spaces YAML frontmatter: `sdk: gradio`, `sdk_version: "5.0"`, `python_version: "3.11"`, `app_file: app.py`, `pinned: false`, plus project description matching spec

**Checkpoint**: Repo structure exists; `pip install -r requirements.txt` succeeds; HF Spaces can parse `README.md` frontmatter.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure consumed by all three user stories — env config loading, MCP connection, model wiring. Must be complete before any user story begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T005 Implement env config loader in `src/config.py`: read all vars from `.env.example` via `python-dotenv`; raise `EnvironmentError` with a clear message for any missing required var at startup; `LANGFUSE_PROJECT_ID` is optional — include it in the Langfuse client init if present
- [X] T006 [P] Implement `src/mcp_client.py`: `build_mcp_tools(urls: list[str])` function that opens an `MCPClient` for each URL in `[MCP_SERVER_URL_1, MCP_SERVER_URL_2]` using `{"url": url, "transport": "streamable-http"}`, merges the returned tool lists, and returns the combined list; validate every URL starts with `https://`; handle `ConnectionError` per server by logging a warning and continuing — a failure on one server MUST NOT prevent tools from the other from loading
- [X] T007 [P] Implement `src/agent.py`: `build_agent(tools, model_id, model_api_key)` factory that creates a `LiteLLMModel` and a `ToolCallingAgent`; accepts a pre-built tool list from `build_mcp_tools`
- [X] T008 Create stub `app.py` that loads config, calls `build_mcp_tools`, calls `build_agent`, and calls `GradioUI(agent).launch()` — enough to verify the wiring compiles and launches without errors

**Checkpoint**: `python app.py` starts Gradio on localhost; agent is present (even if MCP tools list is empty in local dev); no import errors.

---

## Phase 3: User Story 1 — Chat with AI Agent via Tool-Enabled Interface (Priority: P1) 🎯 MVP

**Goal**: A user opens the app, types a question, the agent calls an MCP tool if needed, and returns a grounded multi-turn response. Errors are user-friendly; empty messages are rejected; rapid submissions are queued.

**Independent Test**: Open the deployed app URL, type "What is the current weather in Paris?", verify the assistant returns a tool-informed response within 30 seconds, and that a follow-up question receives a contextually relevant reply.

### Implementation for User Story 1

- [X] T009 [P] [US1] Harden `src/mcp_client.py` for FR-003/FR-004: raise a custom `MCPToolError(Exception)` exception class (defined in `src/mcp_client.py`) when `tools/call` returns `isError: true` or when the MCP transport layer raises; catch only `MCPToolError` at the call site and return a formatted `"[Tool unavailable: {reason}]"` string into the agent's reasoning context (not exposed raw to the user); all other exception types propagate upward to the T013 handler
- [X] T010 [P] [US1] Harden `src/agent.py` for FR-002 (multi-turn context): confirm `ToolCallingAgent` is initialised with `reset_memory_between_tasks=False` so conversation history is maintained within a Gradio session
- [X] T011 [US1] Implement empty-message guard in `app.py`: before passing user input to the agent, strip whitespace and return a user-friendly prompt ("Please enter a message.") if the result is empty; use `gr.Warning` or a chat reply, not an exception
- [X] T012 [US1] Wire `GradioUI` in `app.py` with `reset_agent_memory=False` (multi-turn) and verify the Gradio `ChatInterface` disables the submit button during agent execution (default behaviour — add an explicit `concurrency_limit=1` kwarg if needed to prevent race conditions per edge-case requirement)
- [X] T013 [US1] Implement human-readable error wrapper in `app.py`: catch `Exception` (excluding `MCPToolError` which is handled inside `src/mcp_client.py` before reaching this layer) from `agent.run()`, log the full traceback locally via `logging.exception()`, and return a user-friendly message ("Something went wrong. Please try again.") — never expose raw stack traces (FR-011)
- [X] T014 [US1] Write integration test `tests/integration/test_agent_mcp.py`: sends one message that requires a tool call, asserts the response is non-empty and contains tool-informed content; skipped if `MCP_SERVER_URL` or `MODEL_API_KEY` env vars are not set

**Checkpoint**: `python app.py` launches; sending a tool-requiring question returns a grounded response; follow-up messages use prior context; empty messages show a prompt; MCP errors produce a readable message, not a crash. US1 acceptance scenarios 1–4 all pass manually.

---

## Phase 4: User Story 2 — Observe Agent Interactions via Telemetry Dashboard (Priority: P2)

**Goal**: Every agent turn produces a Langfuse trace with full reasoning + tool call spans. Telemetry failures degrade gracefully — they never crash the chat app.

**Independent Test**: After sending a message through the P1 chat interface, verify a new trace appears in the Langfuse dashboard within 10 seconds with at least one tool call span visible.

### Implementation for User Story 2

- [X] T015 [US2] Implement `src/telemetry.py`: `bootstrap_telemetry()` function that calls `SmolagentsInstrumentor().instrument()` then `langfuse = get_client()` then `langfuse.auth_check()`; wrap `auth_check()` in a try/except — on failure, log `"[Telemetry] Langfuse unreachable, continuing without tracing"` and return `None` (graceful degradation per edge-case requirement)
- [X] T016 [US2] Integrate `bootstrap_telemetry()` into `app.py` at startup (before `GradioUI.launch()`); if it returns `None`, continue without telemetry — do not block app start
- [X] T017 [US2] Inject `session_id` and `app_version` into each Langfuse trace: inside the Gradio chat handler, call `langfuse.update_current_trace(session_id=session_id, metadata={"app_version": APP_VERSION})` using the `ConversationSession.session_id` stored in `gr.State`; guard with `if langfuse is not None` to handle degraded mode
- [X] T018 [P] [US2] Initialise `session_id` (UUID4) in `gr.State` on app load so each browser tab gets a unique session identifier for Langfuse trace grouping
- [X] T028 [US2] Implement collapsible metadata panel in `app.py` for SC-006: after each assistant turn, append a `gr.Accordion` (collapsed by default, labelled "Session info") below the chat message containing the current `session_id` and a direct Langfuse trace URL (`{LANGFUSE_HOST}/project/{LANGFUSE_PROJECT_ID}/traces/{trace_id}`); guard the URL generation with `if langfuse is not None` to handle degraded mode

**Checkpoint**: Send a message; open Langfuse dashboard; verify trace appears within 10 seconds with user message, tool call span (if tool was invoked), and final response. Set `LANGFUSE_HOST` to an invalid URL, restart app — verify app still starts and chat still works (graceful degradation). US2 acceptance scenarios 1–3 pass.

---

## Phase 5: User Story 3 — Evaluate Agent Output Quality (Priority: P3)

**Goal**: A developer runs a batch evaluation that scores Langfuse traces on defined quality criteria and posts numeric scores back, visible in the Langfuse UI.

**Independent Test**: Run `python -m src.evaluation --from <date> --to <date>` against a set of recorded traces; verify each trace receives a `relevance`, `correctness`, and `tool_efficiency` score in the Langfuse UI.

### Implementation for User Story 3

- [X] T019 [US3] Implement `src/evaluation.py` with a `run_evaluation(from_date, to_date)` function: use `get_client()` to fetch traces in the date range; for each trace, compute scores using LLM-as-judge via `LiteLLMModel` with the following 1-shot prompt templates per criterion:
  - **relevance** (0–1): Prompt — *"You are an evaluator. Given the user query and the assistant response below, rate how relevant the response is to the query on a scale of 0.0 to 1.0, where 1.0 = fully relevant and addresses all aspects, 0.0 = completely irrelevant. Return only a floating-point number. Query: {input} Response: {output}"*
  - **correctness** (0–1): Prompt — *"You are an evaluator. Given the user query, the assistant response, and any tool outputs used, rate factual correctness on a scale of 0.0 to 1.0, where 1.0 = fully accurate, 0.0 = factually wrong. Return only a floating-point number. Query: {input} Tool outputs: {tool_outputs} Response: {output}"*
  - **tool_efficiency** (0–1): Prompt — *"You are an evaluator. Given the user query and the list of tool calls made, rate how efficiently the agent used tools (0.0 = unnecessary/excessive tool calls, 1.0 = exactly the right tools called once each). Return only a floating-point number. Query: {input} Tool calls: {tool_calls}"*
  - Parse the LLM response as a float; clamp to [0.0, 1.0] on parse error; post via `langfuse.create_score(trace_id=..., name=..., value=..., data_type="NUMERIC", score_id=<deterministic-uuid>)`
- [X] T020 [US3] Add CLI entry point to `src/evaluation.py`: parse `--from` and `--to` date args (ISO 8601 format) using `argparse`; print a summary table of scored traces to stdout on completion
- [X] T021 [P] [US3] Write unit tests `tests/unit/test_evaluation.py`: test score value clamping (value always in [0.0, 1.0]), test deterministic `score_id` generation for idempotent re-runs, test CLI arg parsing; mock all Langfuse and LLM calls
- [X] T022 [US3] Document evaluation workflow in `quickstart.md` under "Running the Evaluation Workflow" — the section stub already exists; fill in concrete example command, expected output, and how to view scores in Langfuse UI

**Checkpoint**: `python -m src.evaluation --from 2026-03-01 --to 2026-03-31` completes without error; at least one score per quality dimension appears on each processed trace in the Langfuse UI. Re-running the same command does not create duplicate scores (idempotency via `score_id`). US3 acceptance scenarios 1–3 pass. `pytest tests/unit/` passes.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final hardening, deployment smoke-test, and documentation review that spans all stories.

- [X] T023 [P] Add `.gitignore` entries for `.env`, `__pycache__/`, `*.pyc`, `.venv/`, `.pytest_cache/`
- [X] T024 [P] Review all user-facing error messages across `app.py`, `src/mcp_client.py`, `src/telemetry.py` — confirm no raw exception text or stack traces are ever returned to the user (FR-011 audit)
- [X] T025 Validate context-window truncation edge case in `app.py`: if `agent.memory` exceeds a configurable threshold (default 20 turns), trim the oldest turns and surface a notice to the user ("Earlier messages were summarised to fit context limits.")
- [ ] T026 Run full quickstart.md validation: follow every step from "Local Development" through "Smoke test" in a clean environment; fix any discrepancy between docs and actual behaviour
- [ ] T027 [P] Final HF Spaces deployment: push to the Space repository, verify the Space builds successfully, verify the public URL is accessible without login (SC-005), and confirm a trace appears in Langfuse after sending a test message (SC-002)

**Checkpoint**: All 6 success criteria (SC-001 through SC-006) verified in the live HF Space. `pytest tests/` passes in CI.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **BLOCKS all user stories**
- **Phase 3 (US1)**: Depends on Phase 2 — no dependency on US2/US3
- **Phase 4 (US2)**: Depends on Phase 2 — additionally requires Phase 3 app running (to generate traces to observe); T028 depends on T017 (trace_id must be available before building the Langfuse URL)
- **Phase 5 (US3)**: Depends on Phase 2 and Phase 4 (traces must exist in Langfuse to evaluate)
- **Phase 6 (Polish)**: Depends on all chosen stories being complete

### User Story Dependencies

| Story | Depends on | Independently testable? |
|-------|-----------|------------------------|
| US1 (P1) | Phase 2 only | ✅ Yes — no Langfuse required |
| US2 (P2) | Phase 2 + US1 running | ✅ Yes — verified against live traces from US1 |
| US3 (P3) | Phase 2 + US2 traces in Langfuse | ✅ Yes — batch script, no chat UI changes |

### Within Each User Story

- `src/config.py` (T005) must exist before any `src/*.py` file that reads env vars
- `src/mcp_client.py` (T006) and `src/agent.py` (T007) can be developed in parallel
- `app.py` stub (T008) must be complete before US1 UI tasks (T011–T013) begin
- `src/telemetry.py` (T015) must be complete before session injection (T017–T018)
- Evaluation fetch logic (T019) must be complete before CLI entry point (T020)

---

## Parallel Opportunities

### Phase 1 — can all run in parallel after T001

```
T001 → T002, T003 [P], T004 [P]   (T003 and T004 both depend only on T001 structure)
```

### Phase 2 — T006 and T007 in parallel after T005

```
T005 → T006 [P], T007 [P] → T008
```

### Phase 3 — T009 and T010 in parallel, then serial chain

```
T009 [P], T010 [P] → T011 → T012 → T013 → T014
```

### Phase 4 — T018 in parallel with T015

```
T015 → T016 → T017 → T028
T018 [P]   (can start as soon as Phase 2 is complete)
```

### Phase 5 — T021 in parallel with T019

```
T019 → T020
T021 [P]   (unit tests can be written alongside T019 logic)
T019, T020 → T022
```

### Phase 6 — T023 and T024 in parallel

```
T023 [P], T024 [P]  →  T025 → T026 → T027
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete **Phase 1**: Setup (T001–T004)
2. Complete **Phase 2**: Foundational (T005–T008) — critical gate
3. Complete **Phase 3**: User Story 1 (T009–T014)
4. **STOP and VALIDATE**: open app locally, test tool call, test follow-up, test error cases
5. Deploy to HF Spaces and run smoke test (SC-001, SC-003, SC-005)

### Incremental Delivery

1. Phases 1–3 → **MVP deployed** (US1 functional, publicly accessible)
2. Phase 4 → Add telemetry → **traces visible in Langfuse** (US2)
3. Phase 5 → Add evaluation → **scores attached to traces** (US3)
4. Phase 6 → Polish & harden → **all success criteria verified**

### Suggested MVP Scope

**Stop after Phase 3** for the MVP checkpoint. US1 alone satisfies FR-001 through FR-004, FR-006, FR-008, FR-010, FR-011 and success criteria SC-001, SC-003, SC-005. Telemetry (US2) and evaluation (US3) can ship in subsequent iterations.
