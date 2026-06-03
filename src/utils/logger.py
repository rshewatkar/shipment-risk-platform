"""
Structured logger using structlog.
Every module in the project imports this instead of the standard logging module.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("model loaded", model_version="1.0.0", latency_ms=43)
"""

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for the entire application."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named structured logger.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        structlog.BoundLogger: Configured logger instance.
    """
    return structlog.get_logger(name)


# Run setup on import
setup_logging()
