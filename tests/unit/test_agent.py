"""
tests/unit/test_agent.py — Unit tests for agent factory.

Tests model instantiation across all supported model types.
"""

import pytest

from src.config import Config


class TestBuildModel:
    """Tests for _build_model() factory function."""

    def test_build_hf_api_model(self):
        """InferenceClientModel should be instantiated with model_id and token."""
        from src.agent import _build_model

        config = Config(
            model_type="hf_api",
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            model_api_key="",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            hf_token="hf_token123",
        )

        model = _build_model(config)
        assert model.__class__.__name__ == "InferenceClientModel"
        assert model.model_id == "Qwen/Qwen2.5-Coder-32B-Instruct"

    def test_build_litellm_model(self):
        """LiteLLMModel should be instantiated with model_id and api_key."""
        from src.agent import _build_model

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )

        model = _build_model(config)
        assert model.__class__.__name__ == "LiteLLMModel"
        assert model.model_id == "openai/gpt-4o"

    def test_build_openai_server_model(self):
        """OpenAIServerModel should be instantiated with model_id, api_base, and api_key."""
        from src.agent import _build_model

        config = Config(
            model_type="openai_server",
            model_id="gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            openai_api_base="https://api.openai.com/v1",
        )

        model = _build_model(config)
        assert model.__class__.__name__ == "OpenAIModel"
        assert model.model_id == "gpt-4o"

    def test_build_azure_openai_server_model(self):
        """AzureOpenAIServerModel should be instantiated with required Azure params."""
        from src.agent import _build_model

        config = Config(
            model_type="azure_openai_server",
            model_id="gpt-4o-mini",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            azure_endpoint="https://myresource.openai.azure.com/",
            azure_api_version="2024-12-01-preview",
        )

        model = _build_model(config)
        assert model.__class__.__name__ == "AzureOpenAIModel"
        assert model.model_id == "gpt-4o-mini"

    def test_build_model_invalid_type_raises_error(self):
        """_build_model() should raise ValueError for invalid model_type."""
        from src.agent import _build_model

        config = Config(
            model_type="invalid_type",
            model_id="test",
            model_api_key="test",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )

        with pytest.raises(ValueError, match="Unknown model_type"):
            _build_model(config)


class TestBuildAgent:
    """Tests for build_agent() factory function."""

    def test_build_agent_with_litellm_config(self):
        """build_agent() should create ToolCallingAgent with LiteLLMModel."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
        )

        agent = build_agent(tools=[], config=config)
        assert agent.__class__.__name__ == "ToolCallingAgent"
        assert agent.model.__class__.__name__ == "LiteLLMModel"

    def test_build_agent_with_hf_api_config(self):
        """build_agent() should create ToolCallingAgent with InferenceClientModel."""
        from src.agent import build_agent

        config = Config(
            model_type="hf_api",
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            model_api_key="",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            hf_token="hf_test123",
        )

        agent = build_agent(tools=[], config=config)
        assert agent.__class__.__name__ == "ToolCallingAgent"
        assert agent.model.__class__.__name__ == "InferenceClientModel"


class TestBuildAgentVerbosityLevel:
    """Tests for agent verbosity level configuration."""

    def test_agent_accepts_verbosity_level_parameter(self):
        """build_agent() should accept verbosity_level parameter."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_verbosity_level=0,  # ERROR level
        )

        agent = build_agent(tools=[], config=config)
        assert agent.__class__.__name__ == "ToolCallingAgent"

    def test_agent_with_debug_verbosity_level(self):
        """Agent should accept DEBUG verbosity level (-1)."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_verbosity_level=-1,  # DEBUG level
        )

        agent = build_agent(tools=[], config=config)
        assert agent is not None

    def test_agent_with_info_verbosity_level(self):
        """Agent should accept INFO verbosity level (1, default)."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_verbosity_level=1,  # INFO level (default)
        )

        agent = build_agent(tools=[], config=config)
        assert agent is not None

    def test_agent_with_off_verbosity_level(self):
        """Agent should accept OFF verbosity level (2)."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_verbosity_level=2,  # OFF level
        )

        agent = build_agent(tools=[], config=config)
        assert agent is not None


class TestBuildAgentMaxSteps:
    """Tests for agent max_steps configuration."""

    def test_agent_accepts_max_steps_parameter(self):
        """build_agent() should accept max_steps parameter."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_max_steps=5,  # Default value
        )

        agent = build_agent(tools=[], config=config)
        assert agent.__class__.__name__ == "ToolCallingAgent"

    def test_agent_with_custom_max_steps(self):
        """Agent should accept custom max_steps values."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_max_steps=10,
        )

        agent = build_agent(tools=[], config=config)
        assert agent is not None

    def test_agent_with_single_step(self):
        """Agent should accept max_steps=1."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_max_steps=1,
        )

        agent = build_agent(tools=[], config=config)
        assert agent is not None

    def test_agent_with_large_max_steps(self):
        """Agent should accept large max_steps values."""
        from src.agent import build_agent

        config = Config(
            model_type="litellm",
            model_id="openai/gpt-4o",
            model_api_key="sk-test123",
            mcp_server_url_1="https://example.com/mcp1",
            mcp_server_url_2="https://example.com/mcp2",
            langfuse_public_key="pk",
            langfuse_secret_key="sk",
            agent_max_steps=100,
        )

        agent = build_agent(tools=[], config=config)
        assert agent is not None

