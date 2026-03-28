"""
src/agent.py — Agent factory.

Builds a smolagents ToolCallingAgent backed by a LiteLLMModel.
The agent is configured for multi-turn conversations (reset_memory_between_tasks=False)
so Gradio session history is preserved across turns within a single browser session.
"""

import logging
from typing import Any

from smolagents import LiteLLMModel, ToolCallingAgent

logger = logging.getLogger(__name__)


def build_agent(
    tools: list[Any],
    model_id: str,
    model_api_key: str,
) -> ToolCallingAgent:
    """
    Create and return a ToolCallingAgent.

    Parameters
    ----------
    tools:
        List of smolagents Tool objects (typically from build_mcp_tools()).
        May be empty — the agent will answer from its own knowledge only.
    model_id:
        LiteLLM model identifier, e.g. "openai/gpt-4o".
    model_api_key:
        API key for the model provider.
    """
    model = LiteLLMModel(model_id=model_id, api_key=model_api_key)

    agent = ToolCallingAgent(
        tools=tools,
        model=model,
    )

    logger.info(
        "Agent initialised with model=%r and %d tool(s).", model_id, len(tools)
    )
    return agent
