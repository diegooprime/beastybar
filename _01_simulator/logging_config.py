"""Logging configuration for the Beasty Bar simulator."""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", format_json: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_json: Whether to output logs in JSON format (useful for production)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    if format_json:
        # Use structured logging for production
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","message":"%(message)s"}'
        )
        handler.setFormatter(formatter)
    else:
        # Use human-readable format for development
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    # Configure root logger
    logging.root.setLevel(log_level)
    logging.root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: The logger name (usually __name__)

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
