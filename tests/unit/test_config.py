"""
tests/unit/test_config.py — Unit tests for configuration loading.

Tests model_type selection and environment variable validation.
"""

import os
import pytest


class TestLoadConfigModelType:
    """Tests for load_config() model_type handling."""

    def _setup_base_env(self):
        """Set up minimal required environment variables."""
        os.environ["MODEL_ID"] = "test-model"
        os.environ["MCP_SERVER_URL_1"] = "https://example.com/mcp1"
        os.environ["MCP_SERVER_URL_2"] = "https://example.com/mcp2"
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk"

    def _cleanup_env(self):
        """Clean up test environment variables."""
        for key in [
            "MODEL_ID",
            "MODEL_TYPE",
            "MODEL_API_KEY",
            "MCP_SERVER_URL_1",
            "MCP_SERVER_URL_2",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "HF_TOKEN",
            "AZURE_ENDPOINT",
            "AZURE_API_VERSION",
            "OPENAI_API_BASE",
        ]:
            os.environ.pop(key, None)

    def test_default_model_type_is_litellm(self):
        """If MODEL_TYPE is not set, default to litellm."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_API_KEY"] = "sk-test"
        os.environ.pop("MODEL_TYPE", None)  # Ensure it's not set

        try:
            config = load_config()
            assert config.model_type == "litellm"
        finally:
            self._cleanup_env()

    def test_model_type_hf_api(self):
        """load_config() should parse MODEL_TYPE=hf_api."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "hf_api"
        os.environ["HF_TOKEN"] = "hf_test123"

        try:
            config = load_config()
            assert config.model_type == "hf_api"
            assert config.hf_token == "hf_test123"
        finally:
            self._cleanup_env()

    def test_model_type_litellm(self):
        """load_config() should parse MODEL_TYPE=litellm."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "litellm"
        os.environ["MODEL_API_KEY"] = "sk-test"

        try:
            config = load_config()
            assert config.model_type == "litellm"
            assert config.model_api_key == "sk-test"
        finally:
            self._cleanup_env()

    def test_model_type_openai_server(self):
        """load_config() should parse MODEL_TYPE=openai_server."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "openai_server"
        os.environ["MODEL_API_KEY"] = "sk-test"
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"

        try:
            config = load_config()
            assert config.model_type == "openai_server"
            assert config.openai_api_base == "https://api.openai.com/v1"
        finally:
            self._cleanup_env()

    def test_model_type_azure_openai_server(self):
        """load_config() should parse MODEL_TYPE=azure_openai_server."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "azure_openai_server"
        os.environ["MODEL_API_KEY"] = "sk-test"
        os.environ["AZURE_ENDPOINT"] = "https://myresource.openai.azure.com/"
        os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

        try:
            config = load_config()
            assert config.model_type == "azure_openai_server"
            assert config.azure_endpoint == "https://myresource.openai.azure.com/"
            assert config.azure_api_version == "2024-12-01-preview"
        finally:
            self._cleanup_env()

    def test_invalid_model_type_raises_error(self):
        """load_config() should raise EnvironmentError for invalid MODEL_TYPE."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "invalid_type"

        try:
            with pytest.raises(EnvironmentError, match="Invalid MODEL_TYPE"):
                load_config()
        finally:
            self._cleanup_env()

    def test_litellm_requires_model_api_key(self):
        """load_config() should require MODEL_API_KEY for litellm."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "litellm"
        os.environ.pop("MODEL_API_KEY", None)

        try:
            with pytest.raises(EnvironmentError, match="MODEL_API_KEY"):
                load_config()
        finally:
            self._cleanup_env()

    def test_azure_openai_requires_endpoint(self):
        """load_config() should require AZURE_ENDPOINT for azure_openai_server."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "azure_openai_server"
        os.environ["MODEL_API_KEY"] = "sk-test"
        os.environ.pop("AZURE_ENDPOINT", None)

        try:
            with pytest.raises(EnvironmentError, match="AZURE_ENDPOINT"):
                load_config()
        finally:
            self._cleanup_env()

    def test_azure_openai_requires_api_version(self):
        """load_config() should require AZURE_API_VERSION for azure_openai_server."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["MODEL_TYPE"] = "azure_openai_server"
        os.environ["MODEL_API_KEY"] = "sk-test"
        os.environ["AZURE_ENDPOINT"] = "https://myresource.openai.azure.com/"
        os.environ.pop("AZURE_API_VERSION", None)

        try:
            with pytest.raises(EnvironmentError, match="AZURE_API_VERSION"):
                load_config()
        finally:
            self._cleanup_env()


class TestLoadConfigVerbosityLevel:
    """Tests for agent verbosity level configuration."""

    def _setup_base_env(self):
        """Set up minimal required environment variables."""
        os.environ["MODEL_ID"] = "test-model"
        os.environ["MODEL_TYPE"] = "litellm"
        os.environ["MODEL_API_KEY"] = "sk-test"
        os.environ["MCP_SERVER_URL_1"] = "https://example.com/mcp1"
        os.environ["MCP_SERVER_URL_2"] = "https://example.com/mcp2"
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk"

    def _cleanup_env(self):
        """Clean up test environment variables."""
        for key in [
            "MODEL_ID",
            "MODEL_TYPE",
            "MODEL_API_KEY",
            "MCP_SERVER_URL_1",
            "MCP_SERVER_URL_2",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "AGENT_VERBOSITY_LEVEL",
        ]:
            os.environ.pop(key, None)

    def test_default_verbosity_level_is_info(self):
        """If AGENT_VERBOSITY_LEVEL is not set, default to 1 (INFO)."""
        from src.config import load_config

        self._setup_base_env()
        os.environ.pop("AGENT_VERBOSITY_LEVEL", None)

        try:
            config = load_config()
            assert config.agent_verbosity_level == 1
        finally:
            self._cleanup_env()

    def test_verbosity_level_debug(self):
        """load_config() should parse AGENT_VERBOSITY_LEVEL=-1 (DEBUG)."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["AGENT_VERBOSITY_LEVEL"] = "-1"

        try:
            config = load_config()
            assert config.agent_verbosity_level == -1
        finally:
            self._cleanup_env()

    def test_verbosity_level_error(self):
        """load_config() should parse AGENT_VERBOSITY_LEVEL=0 (ERROR)."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["AGENT_VERBOSITY_LEVEL"] = "0"

        try:
            config = load_config()
            assert config.agent_verbosity_level == 0
        finally:
            self._cleanup_env()

    def test_verbosity_level_off(self):
        """load_config() should parse AGENT_VERBOSITY_LEVEL=2 (OFF)."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["AGENT_VERBOSITY_LEVEL"] = "2"

        try:
            config = load_config()
            assert config.agent_verbosity_level == 2
        finally:
            self._cleanup_env()


class TestLoadConfigMaxSteps:
    """Tests for agent max_steps configuration."""

    def _setup_base_env(self):
        """Set up minimal required environment variables."""
        os.environ["MODEL_ID"] = "test-model"
        os.environ["MODEL_TYPE"] = "litellm"
        os.environ["MODEL_API_KEY"] = "sk-test"
        os.environ["MCP_SERVER_URL_1"] = "https://example.com/mcp1"
        os.environ["MCP_SERVER_URL_2"] = "https://example.com/mcp2"
        os.environ["LANGFUSE_PUBLIC_KEY"] = "pk"
        os.environ["LANGFUSE_SECRET_KEY"] = "sk"

    def _cleanup_env(self):
        """Clean up test environment variables."""
        for key in [
            "MODEL_ID",
            "MODEL_TYPE",
            "MODEL_API_KEY",
            "MCP_SERVER_URL_1",
            "MCP_SERVER_URL_2",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "AGENT_MAX_STEPS",
        ]:
            os.environ.pop(key, None)

    def test_default_max_steps_is_5(self):
        """If AGENT_MAX_STEPS is not set, default to 5."""
        from src.config import load_config

        self._setup_base_env()
        os.environ.pop("AGENT_MAX_STEPS", None)

        try:
            config = load_config()
            assert config.agent_max_steps == 5
        finally:
            self._cleanup_env()

    def test_max_steps_custom_value(self):
        """load_config() should parse AGENT_MAX_STEPS=10."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["AGENT_MAX_STEPS"] = "10"

        try:
            config = load_config()
            assert config.agent_max_steps == 10
        finally:
            self._cleanup_env()

    def test_max_steps_one(self):
        """load_config() should parse AGENT_MAX_STEPS=1."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["AGENT_MAX_STEPS"] = "1"

        try:
            config = load_config()
            assert config.agent_max_steps == 1
        finally:
            self._cleanup_env()

    def test_max_steps_large_value(self):
        """load_config() should parse large AGENT_MAX_STEPS values."""
        from src.config import load_config

        self._setup_base_env()
        os.environ["AGENT_MAX_STEPS"] = "100"

        try:
            config = load_config()
            assert config.agent_max_steps == 100
        finally:
            self._cleanup_env()
