# Quickstart: Interactive AI Assistant (smolagents + Gradio + MCP + Langfuse)

**Branch**: `001-smolagents-gradio-mcp`  
**Date**: 2026-03-27

This guide covers local development setup and HuggingFace Spaces deployment.

---

## Prerequisites

- Python 3.11+
- A running MCP server accessible over HTTPS (HTTP Streamable transport) — two server URLs are required
- A model provider API key (OpenAI, Anthropic, Mistral, or any LiteLLM-supported provider)
- A Langfuse account (cloud or self-hosted) — required for P2/P3; optional for P1 local testing

---

## Local Development

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd speckit-chat-app
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Required for all stories
MODEL_ID=openai/gpt-4o
MODEL_API_KEY=sk-...

# Required: MCP servers (both must be set; both must use https://)
MCP_SERVER_URL_1=https://your-first-mcp-server.example.com/mcp
MCP_SERVER_URL_2=https://your-second-mcp-server.example.com/mcp

# Required for P2/P3 telemetry (optional for P1 local testing)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # default; omit if using cloud

# Optional
APP_VERSION=dev
```

### 3. Run the app locally

```bash
python app.py
```

The Gradio interface opens at `http://localhost:7860`. Type a question to test the agent.

### 4. Verify telemetry (P2)

After sending a message, open your Langfuse project dashboard. A new trace should appear
within ~10 seconds, showing the full agent reasoning chain.

### 5. Run tests

```bash
pytest tests/unit/          # fast, no external dependencies
pytest tests/integration/   # requires MCP_SERVER_URL_1, MCP_SERVER_URL_2, MODEL_API_KEY to be set
```

---

## Running the Evaluation Workflow (P3)

The evaluation script scores a batch of existing Langfuse traces using an LLM-as-judge.

### Basic usage

```bash
python -m src.evaluation --from 2026-03-01 --to 2026-03-31
```

### With an explicit run ID (for comparing eval runs)

```bash
python -m src.evaluation --from 2026-03-01 --to 2026-03-31 --run-id my-run-v2
```

### Expected output

```
INFO src.evaluation — Evaluating 12 trace(s) with run_id='a3f1c2d0' …

Trace ID                               relevance  correctness  tool_efficiency
-----------------------------------------------------------------------
trace_01abc...                              0.90         0.85             0.95
trace_02def...                              0.75         0.80             1.00
…

Total evaluated: 12
```

### How to view scores in Langfuse

1. Open your [Langfuse project dashboard](https://cloud.langfuse.com).
2. Go to **Traces** and open any trace from the evaluation date range.
3. Scroll to the **Scores** section — you will see three entries:
   `relevance`, `correctness`, and `tool_efficiency`, each with a value in [0.0, 1.0].
4. Use the **Scores** filter in the Traces list view to compare across runs
   (filter by `run-id` tag stored in the score comment).

### Re-running is safe (idempotent)

Running the same command twice does not create duplicate scores. Each score has a
deterministic ID derived from `(trace_id, criterion, run_id)`, so Langfuse ignores
duplicates silently.

---

## HuggingFace Spaces Deployment

### 1. Create a new Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. Set **SDK** to `Gradio`, **Python version** to `3.11`, **Visibility** to `Public`.
3. Clone the Space repository:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
   ```

### 2. Copy project files into the Space repo

```bash
cp app.py requirements.txt README.md <space-repo>/
cp -r src/ <space-repo>/src/
```

The `README.md` in this repo already contains the required HF Spaces YAML frontmatter:

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

### 3. Set Secrets in HF Spaces

In the Space settings → **Secrets** panel, add each variable from `.env.example`:

| Secret name | Description |
|-------------|-------------|
| `MODEL_ID` | LiteLLM model ID (e.g., `openai/gpt-4o`) |
| `MODEL_API_KEY` | Model provider API key |
| `MCP_SERVER_URL_1` | Full HTTPS URL of your first MCP server |
| `MCP_SERVER_URL_2` | Full HTTPS URL of your second MCP server |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key |
| `LANGFUSE_HOST` | Langfuse host (omit for cloud default) |
| `LANGFUSE_PROJECT_ID` | Project ID for per-trace deep-links in session panel (optional) |

**Never commit secret values to git.**

### 4. Push and verify

```bash
cd <space-repo>
git add .
git commit -m "Deploy AI assistant v1"
git push
```

HF Spaces builds the container automatically. The Space URL (format: `https://huggingface.co/spaces/<user>/<name>`)
becomes publicly accessible once the build succeeds (typically 2–5 minutes).

### 5. Smoke test

Open the Space URL in a browser, type a message that requires a tool call, and verify:
- The assistant responds within 30 seconds (SC-001).
- A trace appears in Langfuse within 10 seconds (SC-002).
- The interface remains usable after the response (SC-003).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: smolagents` | Missing install | `pip install -r requirements.txt` |
| Agent responds but no Langfuse trace | Missing env vars | Check `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` |
| `ConnectionError` on startup | MCP server unreachable | Verify `MCP_SERVER_URL` is correct and server is running |
| Space build fails | Missing `requirements.txt` entries | Check build logs; add missing packages |
| Empty response from agent | Model API key invalid | Check `MODEL_API_KEY` and `MODEL_ID` format |
