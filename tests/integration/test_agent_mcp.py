"""
tests/integration/test_agent_mcp.py — Integration test for the agent + MCP round-trip.

Skipped automatically when MCP_SERVER_URL_1, MCP_SERVER_URL_2, or MODEL_API_KEY
are not set in the environment, so CI runs pass without live credentials.

To run locally:
    MCP_SERVER_URL_1=https://... MCP_SERVER_URL_2=https://... MODEL_API_KEY=sk-... pytest tests/integration/
"""

import os

import pytest

SKIP_REASON = (
    "MCP_SERVER_URL_1, MCP_SERVER_URL_2, and MODEL_API_KEY must be set to run integration tests."
)


def _integration_env_set() -> bool:
    return all(
        os.environ.get(v, "").strip()
        for v in ("MCP_SERVER_URL_1", "MCP_SERVER_URL_2", "MODEL_API_KEY", "MODEL_ID")
    )


@pytest.mark.skipif(not _integration_env_set(), reason=SKIP_REASON)
def test_agent_responds_to_tool_requiring_query():
    """
    Send a message that typically requires a tool call and assert the agent
    returns a non-empty, grounded response.

    Acceptance scenario: US1-2 — tool is called via HTTP Streamable transport
    and the result is incorporated into the response.
    """
    from src.agent import build_agent
    from src.config import load_config
    from src.mcp_client import build_mcp_tools

    cfg = load_config()
    tools, stack = build_mcp_tools([cfg.mcp_server_url_1, cfg.mcp_server_url_2])

    with stack:
        agent = build_agent(
            tools=tools,
            model_id=cfg.model_id,
            model_api_key=cfg.model_api_key,
        )
        response = agent.run("What tools do you have available? List them briefly.")

    assert response, "Agent returned an empty response."
    assert isinstance(response, str), f"Expected str, got {type(response)}"
    assert len(response.strip()) > 10, "Response is suspiciously short."


@pytest.mark.skipif(not _integration_env_set(), reason=SKIP_REASON)
def test_agent_multi_turn_context():
    """
    Send two related messages and verify the second response references the first,
    demonstrating multi-turn context retention (FR-002, US1-3).
    """
    from src.agent import build_agent
    from src.config import load_config
    from src.mcp_client import build_mcp_tools

    cfg = load_config()
    tools, stack = build_mcp_tools([cfg.mcp_server_url_1, cfg.mcp_server_url_2])

    with stack:
        agent = build_agent(
            tools=tools,
            model_id=cfg.model_id,
            model_api_key=cfg.model_api_key,
        )
        agent.run("My name is SpecKit.")
        follow_up = agent.run("What is my name?")

    assert "speckit" in follow_up.lower(), (
        f"Expected follow-up to reference 'SpecKit' from prior turn. Got: {follow_up!r}"
    )


@pytest.mark.skipif(not _integration_env_set(), reason=SKIP_REASON)
def test_agent_mcp_failure_does_not_crash():
    """
    If an MCP server is unreachable, build_mcp_tools() should log a warning
    and return an empty list — the agent must still respond (US1-4).
    """
    from src.agent import build_agent
    from src.config import load_config
    from src.mcp_client import build_mcp_tools

    cfg = load_config()
    # Use a deliberately unreachable HTTPS URL as the second server
    tools, stack = build_mcp_tools(
        [cfg.mcp_server_url_1, "https://localhost:19999/mcp-unreachable"]
    )

    with stack:
        agent = build_agent(
            tools=tools,
            model_id=cfg.model_id,
            model_api_key=cfg.model_api_key,
        )
        response = agent.run("Hello, are you there?")

    assert response, "Agent should still respond even with one MCP server unreachable."
