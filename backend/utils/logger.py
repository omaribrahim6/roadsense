"""Logging configuration"""

import logging
import sys
from pathlib import Path

def setup_logger(name: str = "roadsense", level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger with console output.
    Args:
        name: Logger name
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    #Avoid duplicate
    if logger.handlers:
        return logger
    
    #console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger

def get_logger(name: str = "roadsense") -> logging.Logger:
    """Get an existing logger or create new one"""
    return logging.getLogger(name)
