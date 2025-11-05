import logging
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, file_path: Optional[str] = None) -> logging.Logger:
    """Set up a logger with console and optional file output.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).
        file_path: Optional file path for log output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Avoid duplicate handlers if called multiple times

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if file_path:
        try:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except IOError as e:
            logger.warning(f"Failed to set up file logging at {file_path}: {e}")

    return logger