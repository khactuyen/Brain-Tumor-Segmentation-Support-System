"""
Logging module for Brain Tumor Segmentation Support System.
Provides structured logging to both console and file.
"""

import sys
import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "BraTS_System",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup and configure a logger instance.

    Args:
        name: Logger name
        log_file: Path to output log file (optional)
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logging.Logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console formatter and handler
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger()
