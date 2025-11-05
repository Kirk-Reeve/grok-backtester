import logging
from typing import Optional

def setup_logger(name: str, level: int = logging.INFO, file_path: Optional[str] = None) -> logging.Logger:
    """Sets up a logger with console and optional file output.

    This function configures a logger that can write messages to both the
    console and a specified log file. It prevents the addition of duplicate
    handlers if it is called multiple times for the same logger.

    Args:
        name (str): The name of the logger.
        level (int): The logging level, e.g., logging.INFO, logging.DEBUG.
                     Defaults to logging.INFO.
        file_path (Optional[str]): The optional path to a file where logs
                                   should be saved. Defaults to None.

    Returns:
        logging.Logger: The configured logger instance.
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
