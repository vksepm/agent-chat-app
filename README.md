---
title: AI Assistant (smolagents + MCP)
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.10.0
python_version: '3.13'
app_file: app.py
pinned: false
---

# AI Assistant — smolagents + Gradio + MCP + Langfuse

An interactive AI assistant built with [smolagents](https://github.com/huggingface/smolagents),
[Gradio](https://gradio.app), and [MCP](https://modelcontextprotocol.io) (Model Context Protocol).


![architecture](./application%20architecture.drawio.svg)

## Features

- **Tool-enabled chat**: the agent autonomously calls external tools exposed via MCP servers
  (HTTP Streamable transport) to answer questions grounded in real data.
- **Multi-turn conversations**: conversation history is maintained within each browser session.
- **Full observability**: every agent interaction is traced to [Langfuse](https://langfuse.com)
  with spans for each reasoning step and tool call.
- **Evaluation support**: a batch evaluation workflow scores agent outputs on `relevance`,
  `correctness`, and `tool_efficiency` and posts scores back to Langfuse traces.

## Deployment

Hosted on HuggingFace Spaces. Set the following Secrets in the Space settings:

| Secret                | Required | Description                                     |
| --------------------- | -------- | ----------------------------------------------- |
| `MODEL_ID`            | Yes      | LiteLLM model ID (e.g. `openai/gpt-4o`)         |
| `MODEL_API_KEY`       | Yes      | Model provider API key                          |
| `MCP_SERVER_URL_1`    | Yes      | First MCP server HTTPS URL                      |
| `MCP_SERVER_URL_2`    | Yes      | Second MCP server HTTPS URL                     |
| `LANGFUSE_PUBLIC_KEY` | P2+      | Langfuse public key                             |
| `LANGFUSE_SECRET_KEY` | P2+      | Langfuse secret key                             |
| `LANGFUSE_HOST`       | No       | Defaults to `https://cloud.langfuse.com`        |
| `LANGFUSE_PROJECT_ID` | No       | Enables per-trace Langfuse deep-links in the UI |
| `APP_VERSION`         | No       | Injected into trace metadata; defaults to `dev` |

See [quickstart.md](specs/001-smolagents-gradio-mcp/quickstart.md) for local development setup.