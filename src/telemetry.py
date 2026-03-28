"""
src/telemetry.py — Langfuse v4 OTEL bootstrap.

Langfuse v4 is fully OpenTelemetry-native: the SDK auto-configures a
TracerProvider and OTLP exporter to the Langfuse cloud endpoint when
initialised with valid credentials. All that remains is to instrument
smolagents via SmolagentsInstrumentor (openinference) so every agent.run()
call generates spans that flow into Langfuse automatically.

bootstrap_telemetry() is called once at app startup.  Failures are logged
and return None — the application MUST NOT crash if Langfuse is unreachable
or credentials are missing.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Suppress a known non-fatal OpenTelemetry context-detach error that occurs
# when the openinference SmolagentsInstrumentor wraps stream_to_gradio().
# Root cause: the instrumentor creates a context token in the wrapper function,
# then calls context_api.detach(token) inside the generator's finally block.
# Gradio's async event loop may resume that generator in a different async-task
# context, making ContextVar.reset() raise ValueError.  OpenTelemetry already
# catches this internally and continues; the only effect is the ERROR log line.
# Traces are captured correctly.  Setting the logger to CRITICAL hides it.
# Upstream issue: https://github.com/Arize-ai/openinference/issues
# -----------------------------------------------------------------------
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)


def bootstrap_telemetry(cfg) -> Optional[object]:
    """
    Initialise Langfuse v4 client and instrument smolagents with OTEL.

    Langfuse v4 auto-configures the OpenTelemetry TracerProvider and OTLP
    exporter to the Langfuse cloud endpoint on client init, so no manual
    TracerProvider setup is required.

    Parameters
    ----------
    cfg : src.config.Config
        Loaded application configuration (provides Langfuse credentials).

    Returns
    -------
    langfuse.Langfuse instance on success, or None on failure (degraded mode).
    """
    try:
        # Pin credentials so both the SDK and any OTEL env-var paths see them.
        os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
        os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key
        os.environ["LANGFUSE_BASE_URL"] = cfg.langfuse_base_url

        from langfuse import Langfuse
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        # Langfuse v4: SDK init auto-registers the OTEL TracerProvider and
        # OTLP exporter → Langfuse cloud (base_url/api/public/otel).
        langfuse = Langfuse(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key,
            base_url=cfg.langfuse_base_url,
        )

        if not langfuse.auth_check():
            raise RuntimeError(
                "Langfuse auth_check() returned False — verify credentials and host."
            )

        # Instrument smolagents: every agent.run() / stream_to_gradio() call
        # now generates OTEL spans that are exported to Langfuse automatically.
        SmolagentsInstrumentor().instrument()

        logger.info(
            "Telemetry: Langfuse v4 OTEL bootstrap complete (%s); smolagents instrumented.",
            cfg.langfuse_base_url,
        )
        return langfuse

    except Exception as exc:
        logger.warning(
            "[Telemetry] Bootstrap failed, continuing without tracing. Error: %s",
            exc,
        )
        return None
