"""Tests for logger utility functions."""

from logging import DEBUG

from backtester.utils.logger import setup_logger


def test_setup_logger(caplog):
    """Test logger setup."""
    logger_name = "test_logger"
    logger = setup_logger(logger_name, level=DEBUG)
    with caplog.at_level(DEBUG, logger=logger_name):
        logger.debug("Test debug message")
    assert any("Test debug message" in record.message for record in caplog.records)
