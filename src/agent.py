"""
src/agent.py — Agent factory.

Builds a smolagents ToolCallingAgent backed by a configurable model.
Multi-turn conversation memory is preserved by passing reset=False to
agent.run() or reset_agent_memory=False to stream_to_gradio() at call time
(both default to resetting memory, so callers must opt-in to multi-turn mode).
"""

import logging
from typing import Any

from smolagents import (
    AzureOpenAIServerModel,
    InferenceClientModel,
    LiteLLMModel,
    OpenAIServerModel,
    ToolCallingAgent,
)

from .config import Config

logger = logging.getLogger(__name__)


def _build_model(config: Config) -> Any:
    """
    Create and return a model instance based on configuration.

    Parameters
    ----------
    config:
        Configuration object containing model_type and model-specific settings.

    Returns
    -------
        A smolagents model instance (InferenceClientModel, LiteLLMModel, etc.)

    Raises
    ------
    ValueError:
        If model_type is invalid or required credentials are missing.
    """
    if config.model_type == "hf_api":
        logger.info("Creating InferenceClientModel with model_id=%r", config.model_id)
        return InferenceClientModel(
            model_id=config.model_id,
            token=config.hf_token or None,
        )

    elif config.model_type == "litellm":
        logger.info("Creating LiteLLMModel with model_id=%r", config.model_id)
        return LiteLLMModel(model_id=config.model_id, api_key=config.model_api_key)

    elif config.model_type == "openai_server":
        logger.info(
            "Creating OpenAIServerModel with model_id=%r and api_base=%r",
            config.model_id,
            config.openai_api_base,
        )
        return OpenAIServerModel(
            model_id=config.model_id,
            api_base=config.openai_api_base or None,
            api_key=config.model_api_key,
        )

    elif config.model_type == "azure_openai_server":
        logger.info(
            "Creating AzureOpenAIServerModel with model_id=%r and endpoint=%r",
            config.model_id,
            config.azure_endpoint,
        )
        return AzureOpenAIServerModel(
            model_id=config.model_id,
            azure_endpoint=config.azure_endpoint,
            api_key=config.model_api_key,
            api_version=config.azure_api_version,
        )

    else:
        raise ValueError(
            f"Unknown model_type '{config.model_type}'. "
            "Valid options: hf_api, litellm, openai_server, azure_openai_server"
        )


def build_agent(
    tools: list[Any],
    config: Config,
) -> ToolCallingAgent:
    """
    Create and return a ToolCallingAgent.

    Parameters
    ----------
    tools:
        List of smolagents Tool objects (typically from build_mcp_tools()).
        May be empty — the agent will answer from its own knowledge only.
    config:
        Configuration object containing model_type and model-specific settings.
    """
    model = _build_model(config)

    # Custom instructions to guide the agent's behavior
    instructions = (
        "You are a helpful AI assistant. When you have all the information needed "
        "to answer the user's question, provide the final answer immediately. "
        "Do not call the same tool multiple times with identical arguments. "
        "After receiving a tool response, analyze it carefully and decide: "
        "(1) Do you have enough information to answer? If yes, provide the final_answer. "
        "(2) Do you need more information? If yes, call a different tool or ask for clarification. "
        "(3) Is this already resolved? If yes, provide the final_answer without calling the tool again."
    )

    agent = ToolCallingAgent(
        tools=tools,
        model=model,
        verbosity_level=config.agent_verbosity_level,
        max_steps=config.agent_max_steps,
        instructions=instructions,
    )

    logger.info(
        "Agent initialised with model_type=%s, model_id=%s, verbosity_level=%d, max_steps=%d and %d tool(s).",
        config.model_type,
        config.model_id,
        config.agent_verbosity_level,
        config.agent_max_steps,
        len(tools),
    )
    return agent
