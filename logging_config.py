"""
Application-wide logging setup. Logs to both console and a rotating file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(app_config):
    log_dir = os.path.dirname(app_config.LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(app_config.LOG_LEVEL)

    # Avoid duplicate handlers if setup_logging is called more than once
    if root_logger.handlers:
        return root_logger

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        app_config.LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    return root_logger
