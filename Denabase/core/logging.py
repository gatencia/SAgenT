import logging
import os
import sys
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with the given name.
    Avoids duplicate handlers and respects DENABASE_LOG_LEVEL.
    """
    logger = logging.getLogger(name)
    
    # Get log level from environment variable
    log_level_str = os.getenv("DENABASE_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    # Ensure the logger doesn't propagate to a root logger that might have different settings
    logger.propagate = False
    
    return logger

# Default library logger
logger = get_logger("denabase")
