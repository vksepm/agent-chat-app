# Contract: MCP Tool Interface

**Phase**: 1 — Design & Contracts  
**Branch**: `001-smolagents-gradio-mcp`  
**Date**: 2026-03-27  
**Scope**: The interface between the AI assistant application and externally hosted MCP servers.

---

## Overview

This application consumes tools exposed by one or more MCP (Model Context Protocol) servers via
HTTP Streamable transport. The application is a **client only** — it does not host an MCP server.
This document defines:

1. The transport and protocol version requirements the MCP server MUST satisfy.
2. The tool discovery contract (what the app reads from the server).
3. The tool invocation schema (what the app sends / expects in response).
4. Error handling obligations on both sides.

---

## Transport Contract

| Property | Requirement |
|----------|-------------|
| Protocol | MCP (Model Context Protocol) |
| Transport | HTTP Streamable (`transport: "streamable-http"`) |
| Base URL | Configured via `MCP_SERVER_URL_1` and `MCP_SERVER_URL_2` env vars; both MUST be reachable over HTTPS from HF Spaces |
| Auth | Optional; if required by the server, credentials are passed via headers configured in `MCPClient` init (env-var driven) |
| TLS | REQUIRED; plain HTTP is not permitted for any MCP connection |

---

## Tool Discovery Contract

The application calls `MCPClient.__enter__()` at startup for each of the two configured server URLs (`MCP_SERVER_URL_1`, `MCP_SERVER_URL_2`) to retrieve their tool catalogues. Each MCP
server MUST respond to the standard MCP `tools/list` request.

**Expected response shape per tool**:
```json
{
  "name": "<string: unique tool name>",
  "description": "<string: natural-language description for LLM>",
  "inputSchema": {
    "type": "object",
    "properties": {
      "<param>": {
        "type": "<json-schema-type>",
        "description": "<param description>"
      }
    },
    "required": ["<required-params>"]
  }
}
```

**App obligations**:
- The app MUST NOT hard-code tool names. All tools are discovered dynamically.
- If `tools/list` returns zero tools, the app MUST log a warning and continue (agent operates without tools).
- If `tools/list` fails (connection error, timeout), the app MUST surface a startup warning and continue without MCP tooling rather than crashing.

---

## Tool Invocation Contract

The agent invokes tools via the MCP `tools/call` method. smolagents handles serialisation and
deserialisation; this section documents the schema for integration verification.

**Request** (sent by smolagents `MCPClient`):
```json
{
  "method": "tools/call",
  "params": {
    "name": "<tool-name>",
    "arguments": {
      "<param>": "<value>"
    }
  }
}
```

**Success response** (expected from MCP server):
```json
{
  "content": [
    {
      "type": "text",
      "text": "<tool output as string>"
    }
  ],
  "isError": false
}
```

**Error response** (MCP server MUST return this form on tool failure, not HTTP 5xx):
```json
{
  "content": [
    {
      "type": "text",
      "text": "<human-readable error description>"
    }
  ],
  "isError": true
}
```

**App obligations on error response**:
- If `isError: true`, the agent MUST receive the error text as the tool result and incorporate it into its reasoning.
- The app MUST NOT display raw MCP error text to the user; it MUST translate it to a human-readable message (FR-011).
- The agent MUST continue operating and attempt to answer without the tool rather than propagating the error to a crash (FR-004 acceptance scenario 4).

---

## Session Lifecycle

| Event | App Action | MCP Server Expectation |
|-------|-----------|------------------------|
| App startup | Opens one `MCPClient` connection per server URL, calls `tools/list` on each | Must respond within 10s |
| Agent turn starts | Reuses existing `MCPClient` connections | Connections are long-lived (context managers open) |
| Tool call | Sends `tools/call` within active connection | Must respond within 25s (leaving 5s margin vs. 30s p95 SLA) |
| App shutdown | `MCPClient.__exit__()` closes both connections | Servers may close sessions |

---

## Security Obligations

- MCP server URLs and any bearer tokens MUST be stored in HF Spaces Secrets (env vars), never in source code.
- The app MUST validate that both `MCP_SERVER_URL_1` and `MCP_SERVER_URL_2` are `https://` URLs before establishing connections (guard implemented in `src/mcp_client.py`).
- Tool outputs are treated as untrusted input to the LLM context window — no execution of tool output as code on the client side.

---

## Known Limitations (v1)

- Exactly two MCP server URLs are supported in v1 (`MCP_SERVER_URL_1`, `MCP_SERVER_URL_2`). Support for more than two servers is deferred to v2.
- Tool catalogue is loaded once at startup. Dynamic tool registration/deregistration during a session is not supported.
- `structured_output=True` (MCP spec 2025-06-18+) is not enabled in v1; tool outputs are treated as plain text.
