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


class TelemetrySession:
    """Scopes Langfuse session attributes to one Gradio request turn.

    Encapsulates the known OpenTelemetry context-var detach workaround:
    ``propagate_attributes().__enter__()`` is called without ``__exit__()``
    because Gradio's ``copy_context()`` isolation per request cleans up
    ContextVar tokens automatically when the request task ends.

    Background: Gradio's streaming event loop resumes the chat generator in a
    different async-task context than the one where ``__enter__()`` ran.
    Calling ``__exit__()`` in that resumed context causes
    ``ContextVar.reset()`` to raise ``ValueError`` ("<Token var=... at ...>
    was created in a different Context").  Skipping ``__exit__()`` is safe
    because Gradio calls ``copy_context()`` for every request — the context
    copy is isolated to that request's task and is discarded when the task
    ends, so context variables set inside it never leak to other requests.

    Upstream tracking issue:
    https://github.com/Arize-ai/openinference/issues

    Usage
    -----
    .. code-block:: python

        TelemetrySession(langfuse, session_id, cfg.app_version).attach()

    No teardown is required.
    """

    def __init__(
        self,
        langfuse: Optional[object],
        session_id: str,
        app_version: str,
    ) -> None:
        self._langfuse = langfuse
        self._session_id = session_id
        self._app_version = app_version

    def attach(self) -> None:
        """Propagate session attributes to all child OTEL spans for this request.

        No-op when the Langfuse client is ``None`` (degraded mode).
        Logs a warning and continues if propagation fails — the app must not
        crash because of a telemetry failure.

        ``__exit__()`` is intentionally *not* called on the returned context
        manager.  See the class docstring for the full rationale.
        """
        if self._langfuse is None:
            return
        try:
            from langfuse import propagate_attributes

            propagate_attributes(
                session_id=self._session_id,
                metadata={"app_version": self._app_version},
            ).__enter__()
            # __exit__() intentionally omitted — see class docstring.
        except Exception as exc:
            logger.warning("TelemetrySession.attach() failed: %s", exc)


def shutdown_telemetry(langfuse: Optional[object]) -> None:
    """Flush pending spans and close the Langfuse OTEL exporter.

    Call from an ``atexit`` handler or ``demo.close()`` callback to ensure
    in-flight traces are exported before the process exits.  Safe to call
    when *langfuse* is ``None``.

    Parameters
    ----------
    langfuse:
        The Langfuse client returned by ``bootstrap_telemetry()``, or
        ``None`` if telemetry is disabled.
    """
    if langfuse is None:
        return
    try:
        langfuse.flush()
    except Exception as exc:
        logger.warning("Telemetry flush failed: %s", exc)


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
