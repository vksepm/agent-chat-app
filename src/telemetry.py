"""
src/telemetry.py — Langfuse v2 bootstrap.

Initialises the Langfuse v2 client using the native Python SDK (direct HTTP API).
OTEL/openinference instrumentation is intentionally omitted: Langfuse v2 uses its
own HTTP ingestion path, not OTLP, so SmolagentsInstrumentor is not compatible
(openinference 0.1.x crashes with NonRecordingSpan when no TracerProvider is set).

bootstrap_telemetry() is called once at app startup. If Langfuse is unreachable
or credentials are missing, it logs a warning and returns None — the calling code
MUST check the return value and skip all telemetry calls when None is returned.
Telemetry failures MUST NOT crash the chat application (edge case requirement).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def bootstrap_telemetry(cfg) -> Optional[object]:
    """
    Initialise the Langfuse v2 client.

    Parameters
    ----------
    cfg : src.config.Config
        Loaded application configuration (provides Langfuse credentials and host).

    Returns
    -------
    langfuse.Langfuse instance on success, or None on failure (degraded mode).
    """
    try:
        from langfuse import Langfuse

        # Explicitly pin env vars so the SDK's internal fallback path also sees
        # the correct credentials, regardless of what the shell environment holds.
        os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
        os.environ["LANGFUSE_HOST"] = cfg.langfuse_host

        langfuse = Langfuse(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key,
            host=cfg.langfuse_host,
        )
        if not langfuse.auth_check():
            raise RuntimeError(
                "Langfuse auth_check() returned False — verify credentials and host."
            )
        logger.info("Langfuse client authenticated successfully.")
        return langfuse
    except Exception as exc:
        logger.warning(
            "[Telemetry] Langfuse unreachable, continuing without tracing. Error: %s",
            exc,
        )
        return None
