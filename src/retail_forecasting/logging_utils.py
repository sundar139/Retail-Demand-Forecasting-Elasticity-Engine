"""Logging helpers for the retail forecasting package."""

import logging

from retail_forecasting.config import get_settings


def configure_logging(log_level: str | None = None) -> None:
    """Configure root logging with a stable, readable format.

    Args:
        log_level: Desired logging level name; defaults to Settings.log_level.
    """
    configured_level_name = (log_level or get_settings().log_level).upper()
    resolved_level = getattr(logging, configured_level_name, logging.INFO)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        return

    logging.basicConfig(level=resolved_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
