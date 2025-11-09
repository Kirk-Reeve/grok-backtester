"""Helper functions and classes for the backtester application."""

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Configuration for data fetching.

    Attributes:
        source (str): The source of the data (e.g., 'yfinance').
        symbols (List[str]): A list of stock symbols to fetch.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        cache_dir (str): The directory to cache the data in.
    """

    source: str
    symbols: List[str]
    start_date: str
    end_date: str
    cache_dir: str

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validates that date strings are in 'YYYY-MM-DD' format.

        Args:
            v (str): The date string to validate.

        Returns:
            str: The validated date string.

        Raises:
            ValueError: If the date string is not in the correct format.
        """
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError as error:
            raise ValueError(
                f"Invalid date format: {v}. Must be YYYY-MM-DD."
            ) from error


class StrategyConfig(BaseModel):
    """Configuration for the trading strategy.

    Attributes:
        type (str): The type of the strategy to use (e.g., 'moving_average').
        params (Dict[str, Any]): A dictionary of parameters for the strategy.
    """

    type: str
    params: Dict[str, Any]


class BacktestConfig(BaseModel):
    """Configuration for the backtest engine.

    Attributes:
        initial_capital (float): The initial capital for the backtest.
        commission (float): The commission fee per trade.
        slippage (float): The slippage per trade.
        parallel (bool): Whether to run backtests in parallel.
    """

    initial_capital: float
    commission: float = Field(..., ge=0.0)
    slippage: float = Field(0.0, ge=0.0)
    parallel: bool


class AppConfig(BaseModel):
    """Root configuration for the backtester application.

    Attributes:
        data (DataConfig): The data fetching configuration.
        strategy (StrategyConfig): The trading strategy configuration.
        backtest (BacktestConfig): The backtest engine configuration.
    """

    data: DataConfig
    strategy: StrategyConfig
    backtest: BacktestConfig


class BacktestError(Exception):
    """Base exception for all custom errors in the backtester application."""


class DataError(BacktestError):
    """Exception raised for errors related to data fetching or processing."""


class StrategyError(BacktestError):
    """Exception raised for errors related to trading strategy logic."""


class EngineError(BacktestError):
    """Exception raised for errors within the backtesting engine."""


class MetricsError(BacktestError):
    """Exception raised for errors during performance metrics calculation."""


class VisualizationError(BacktestError):
    """Exception raised for errors during results visualization."""


def open_no_symlink(path: Path):
    """Safely open a file for reading without following symbolic links.

    This function opens the file located at the given path in read-only mode,
    explicitly preventing symlink traversal on systems that support the
    `O_NOFOLLOW` flag. It returns a text-mode file object with UTF-8 encoding.
    If the file cannot be opened safely (e.g., if it's a symlink or inaccessible),
    an exception is raised.

    Args:
        path (Path): The path to the file to open. Must point to a regular file
            that is not a symbolic link.

    Returns:
        TextIOWrapper: A file object opened in read-only text mode with UTF-8 encoding.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PermissionError: If the process lacks permission to read the file.
        OSError: If opening the file fails for any other OS-level reason, including
            attempting to open a symlink when `O_NOFOLLOW` is enforced.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(str(path), flags)
    try:
        # wrap fd with text mode and explicit encoding
        return os.fdopen(fd, "r", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Get the absolute path to the project root directory.

    This resolves from the current module's location, assuming standard structure:
    grok-backtester/
    ├── src/
    │   └── backtester/
    │       └── utils/
    │           └── helpers.py
    Adjust .parent count if your structure differs.

    Returns:
        Path: Absolute path to project root.

    Raises:
        ValueError: If resolution fails (e.g., frozen executable).
    """
    try:
        # __file__ is the current file; resolve to absolute,
        # go up 3 levels (utils -> backtester -> src -> root)
        root = Path(__file__).resolve().parent.parent.parent.parent
        if not root.exists() or not (root / "README.md").exists():
            raise ValueError("Project root detection failed; adjust parent levels.")
        return root
    except NameError:
        return Path(os.getcwd()).resolve()
