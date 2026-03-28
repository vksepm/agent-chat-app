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
    model_id: str
    model_api_key: str

    # MCP servers
    mcp_server_url_1: str
    mcp_server_url_2: str

    # Langfuse (required for telemetry)
    langfuse_public_key: str
    langfuse_secret_key: str

    # Langfuse optional — LANGFUSE_BASE_URL is the v4 name; LANGFUSE_HOST is kept as fallback
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_project_id: str = ""

    # Application metadata
    app_version: str = "dev"


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
    return Config(
        model_id=_require("MODEL_ID"),
        model_api_key=_require("MODEL_API_KEY"),
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
    )
