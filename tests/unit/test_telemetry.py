"""
tests/unit/test_telemetry.py — Boundary tests for src/telemetry.

No live Langfuse credentials or network access required.
All Langfuse types are mocked with unittest.mock.
langfuse is not installed in this environment; sys.modules is patched to
provide a mock module wherever `from langfuse import ...` is called.
"""

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.telemetry import TelemetrySession, shutdown_telemetry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_langfuse_module(propagate_fn=None):
    """Build a minimal mock langfuse module with a controllable propagate_attributes."""
    mock_mod = MagicMock()
    if propagate_fn is not None:
        mock_mod.propagate_attributes = propagate_fn
    return mock_mod


# ---------------------------------------------------------------------------
# TelemetrySession.attach()
# ---------------------------------------------------------------------------


class TestTelemetrySessionAttach:
    def test_attach_noop_when_langfuse_is_none(self):
        """attach() must not raise and must not call propagate_attributes."""
        mock_mod = _mock_langfuse_module()
        with patch.dict(sys.modules, {"langfuse": mock_mod}):
            TelemetrySession(None, "sess-123", "1.0").attach()
        mock_mod.propagate_attributes.assert_not_called()

    def test_attach_calls_propagate_enter(self):
        """attach() calls __enter__() on the context manager returned by propagate_attributes."""
        mock_cm = MagicMock()
        mock_prop = MagicMock(return_value=mock_cm)
        mock_mod = _mock_langfuse_module(propagate_fn=mock_prop)

        with patch.dict(sys.modules, {"langfuse": mock_mod}):
            TelemetrySession(MagicMock(), "sess-abc", "2.0").attach()

        mock_prop.assert_called_once_with(
            session_id="sess-abc",
            metadata={"app_version": "2.0"},
        )
        mock_cm.__enter__.assert_called_once()

    def test_attach_skips_exit(self):
        """__exit__() must NOT be called — the workaround must remain in place."""
        mock_cm = MagicMock()
        mock_prop = MagicMock(return_value=mock_cm)
        mock_mod = _mock_langfuse_module(propagate_fn=mock_prop)

        with patch.dict(sys.modules, {"langfuse": mock_mod}):
            TelemetrySession(MagicMock(), "sess-xyz", "dev").attach()

        mock_cm.__exit__.assert_not_called()

    def test_attach_swallows_propagation_error(self, caplog):
        """attach() must not propagate exceptions from propagate_attributes."""
        mock_prop = MagicMock(side_effect=RuntimeError("OTEL context failure"))
        mock_mod = _mock_langfuse_module(propagate_fn=mock_prop)

        with patch.dict(sys.modules, {"langfuse": mock_mod}):
            with caplog.at_level(logging.WARNING, logger="src.telemetry"):
                TelemetrySession(MagicMock(), "sess-1", "dev").attach()  # must not raise

        assert any("TelemetrySession.attach() failed" in r.message for r in caplog.records)

    def test_attach_swallows_import_error(self, caplog):
        """If from langfuse import propagate_attributes raises, attach() logs and continues."""
        with patch.dict(sys.modules, {"langfuse": None}):  # None causes ImportError
            with caplog.at_level(logging.WARNING, logger="src.telemetry"):
                TelemetrySession(MagicMock(), "s", "v").attach()  # must not raise


# ---------------------------------------------------------------------------
# shutdown_telemetry()
# ---------------------------------------------------------------------------


class TestShutdownTelemetry:
    def test_shutdown_calls_flush(self):
        mock_langfuse = MagicMock()
        shutdown_telemetry(mock_langfuse)
        mock_langfuse.flush.assert_called_once()

    def test_shutdown_noop_when_none(self):
        """shutdown_telemetry(None) must not raise."""
        shutdown_telemetry(None)  # no exception

    def test_shutdown_swallows_flush_error(self, caplog):
        """If flush() raises, shutdown_telemetry() logs and does not re-raise."""
        mock_langfuse = MagicMock()
        mock_langfuse.flush.side_effect = RuntimeError("connection lost")

        with caplog.at_level(logging.WARNING, logger="src.telemetry"):
            shutdown_telemetry(mock_langfuse)  # must not raise

        assert any("flush" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# bootstrap_telemetry() — degraded mode
# ---------------------------------------------------------------------------


class TestBootstrapTelemetry:
    def test_bootstrap_returns_none_on_import_error(self):
        """If langfuse cannot be imported, bootstrap_telemetry() returns None."""
        from src.telemetry import bootstrap_telemetry

        mock_cfg = MagicMock()
        mock_cfg.langfuse_public_key = "pk"
        mock_cfg.langfuse_secret_key = "sk"
        mock_cfg.langfuse_base_url = "https://cloud.langfuse.com"

        with patch.dict("sys.modules", {"langfuse": None}):
            result = bootstrap_telemetry(mock_cfg)

        assert result is None

    def test_bootstrap_returns_none_on_auth_failure(self):
        """If auth_check() returns False, bootstrap_telemetry() returns None."""
        from src.telemetry import bootstrap_telemetry

        mock_cfg = MagicMock()
        mock_cfg.langfuse_public_key = "pk"
        mock_cfg.langfuse_secret_key = "sk"
        mock_cfg.langfuse_base_url = "https://cloud.langfuse.com"

        mock_langfuse_instance = MagicMock()
        mock_langfuse_instance.auth_check.return_value = False

        mock_langfuse_class = MagicMock(return_value=mock_langfuse_instance)
        mock_instrumentor = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "langfuse": MagicMock(Langfuse=mock_langfuse_class),
                "openinference.instrumentation.smolagents": MagicMock(
                    SmolagentsInstrumentor=mock_instrumentor
                ),
            },
        ):
            result = bootstrap_telemetry(mock_cfg)

        assert result is None
