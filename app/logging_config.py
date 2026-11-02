"""Application-only logging configuration and request correlation context."""

import logging
import os
import re
from contextvars import ContextVar
from typing import Optional

_REQUEST_ID: ContextVar[Optional[str]] = ContextVar("request_id", default=None)

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def resolve_log_level(level_name: Optional[str] = None) -> int:
    """Return a supported logging level, defaulting safely to INFO."""
    configured_name = (level_name if level_name is not None else os.getenv("LOG_LEVEL", "INFO"))
    return _LEVELS.get(configured_name.strip().upper(), logging.INFO)


def configure_logging(level_name: Optional[str] = None) -> None:
    """Configure only research_assistant loggers; leave root and dependency levels alone."""
    level = resolve_log_level(level_name)
    app_logger = logging.getLogger("research_assistant")
    app_logger.setLevel(level)
    app_logger.propagate = False

    # A dedicated handler keeps application DEBUG records independent of the
    # root logger and third-party handlers configured by Uvicorn or libraries.
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ))
        app_logger.addHandler(handler)
    for handler in app_logger.handlers:
        handler.setLevel(level)


def set_request_id(request_id: str):
    """Set request-scoped correlation context and return its reset token."""
    return _REQUEST_ID.set(request_id)


def get_request_id() -> Optional[str]:
    """Return the current request ID, if a request context is active."""
    return _REQUEST_ID.get()


def reset_request_id(token) -> None:
    """Restore the previous request context."""
    _REQUEST_ID.reset(token)


class RequestLoggerAdapter(logging.LoggerAdapter):
    """Prefix each application log record with the current request correlation ID."""

    def process(self, msg, kwargs):
        request_id = _REQUEST_ID.get() or "unscoped"
        return f"request_id={request_id} | {msg}", kwargs


def get_logger(name: str) -> RequestLoggerAdapter:
    """Get a logger under the application's isolated logger namespace."""
    return RequestLoggerAdapter(logging.getLogger(f"research_assistant.{name}"), {})


def safe_log_value(value, max_length: int = 120) -> str:
    """Escape control characters and bound untrusted values before logging."""
    text = str(value)
    text = re.sub(
        r"[\x00-\x1f\x7f]",
        lambda match: f"\\x{ord(match.group(0)):02x}",
        text,
    )
    if len(text) > max_length:
        text = text[:max_length] + "..."
    return text
