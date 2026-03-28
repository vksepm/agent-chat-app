"""
src/config.py — Environment configuration loader.

Reads all application settings from environment variables (with .env file support
via python-dotenv). Raises EnvironmentError with a clear message for any missing
required variable so misconfiguration is caught immediately at startup.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class Config:
    # Model provider
    model_type: str  # hf_api, litellm, openai_server, azure_openai_server
    model_id: str

    # MCP servers
    mcp_server_url_1: str
    mcp_server_url_2: str

    # Langfuse (required for telemetry)
    langfuse_public_key: str
    langfuse_secret_key: str

    # Model-specific credentials (some required depending on model_type)
    model_api_key: str = ""  # For litellm and openai_server
    hf_token: str = ""  # For hf_api (optional if using default auth)
    azure_endpoint: str = ""  # For azure_openai_server
    azure_api_version: str = ""  # For azure_openai_server
    openai_api_base: str = ""  # For openai_server (optional)

    # Langfuse optional — LANGFUSE_BASE_URL is the v4 name; LANGFUSE_HOST is kept as fallback
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_project_id: str = ""

    # Application metadata
    app_version: str = "dev"

    # Agent configuration (optional)
    agent_verbosity_level: int = 1  # LogLevel: OFF=-1, ERROR=0, INFO=1 (default), DEBUG=2
    agent_max_steps: int = 5  # Maximum number of steps the agent can take to solve a task

    # ---------------------------------------------------------------------------
    # Data logging (all optional — logging degrades gracefully when unset)
    # ---------------------------------------------------------------------------
    # Directory for the local JSONL interaction log file.
    data_log_dir: str = "logs"
    # HuggingFace Hub dataset repo ID to sync logs to, e.g. "username/chat-logs".
    hf_dataset_repo_id: str = ""
    # How often (seconds) to push the local log file to HF Hub.  Default: 300.
    hf_sync_interval: int = 300


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill in all required values."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


def load_config() -> Config:
    model_type = _optional("MODEL_TYPE", "litellm").lower()
    model_id = _require("MODEL_ID")

    # Validate model_type
    valid_types = {"hf_api", "litellm", "openai_server", "azure_openai_server"}
    if model_type not in valid_types:
        raise EnvironmentError(
            f"Invalid MODEL_TYPE '{model_type}'. Must be one of: {', '.join(valid_types)}"
        )

    # Validate model-specific required variables
    if model_type == "litellm":
        model_api_key = _require("MODEL_API_KEY")
        hf_token = _optional("HF_TOKEN", "")
        azure_endpoint = ""
        azure_api_version = ""
        openai_api_base = ""
    elif model_type == "hf_api":
        model_api_key = ""
        hf_token = _optional("HF_TOKEN", "")  # Optional: uses huggingface_hub default if not set
        azure_endpoint = ""
        azure_api_version = ""
        openai_api_base = ""
    elif model_type == "openai_server":
        model_api_key = _require("MODEL_API_KEY")
        hf_token = ""
        azure_endpoint = ""
        azure_api_version = ""
        openai_api_base = _optional("OPENAI_API_BASE", "https://api.openai.com/v1")
    elif model_type == "azure_openai_server":
        model_api_key = _require("MODEL_API_KEY")
        hf_token = ""
        azure_endpoint = _require("AZURE_ENDPOINT")
        azure_api_version = _require("AZURE_API_VERSION")
        openai_api_base = ""

    return Config(
        model_type=model_type,
        model_id=model_id,
        model_api_key=model_api_key,
        mcp_server_url_1=_require("MCP_SERVER_URL_1"),
        mcp_server_url_2=_require("MCP_SERVER_URL_2"),
        langfuse_public_key=_require("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_require("LANGFUSE_SECRET_KEY"),
        # LANGFUSE_BASE_URL is the v4 env var; LANGFUSE_HOST is accepted as a fallback
        langfuse_base_url=_optional(
            "LANGFUSE_BASE_URL",
            _optional("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        ),
        langfuse_project_id=_optional("LANGFUSE_PROJECT_ID", ""),
        app_version=_optional("APP_VERSION", "dev"),
        agent_verbosity_level=int(_optional("AGENT_VERBOSITY_LEVEL", "1")),
        agent_max_steps=int(_optional("AGENT_MAX_STEPS", "5")),
        hf_token=hf_token,
        azure_endpoint=azure_endpoint,
        azure_api_version=azure_api_version,
        openai_api_base=openai_api_base,
        data_log_dir=_optional("DATA_LOG_DIR", "logs"),
        hf_dataset_repo_id=_optional("HF_DATASET_REPO_ID", ""),
        hf_sync_interval=int(_optional("HF_SYNC_INTERVAL", "300")),
    )
