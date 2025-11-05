from typing import List, Dict, Any
from pydantic import BaseModel, field_validator, Field
from datetime import datetime
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

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

    @field_validator('start_date', 'end_date', mode='before')
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
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Must be YYYY-MM-DD.")

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
    pass

class DataError(BacktestError):
    """Exception raised for errors related to data fetching or processing."""
    pass

class StrategyError(BacktestError):
    """Exception raised for errors related to trading strategy logic."""
    pass

class EngineError(BacktestError):
    """Exception raised for errors within the backtesting engine."""
    pass

class MetricsError(BacktestError):
    """Exception raised for errors during performance metrics calculation."""
    pass

class VisualizationError(BacktestError):
    """Exception raised for errors during results visualization."""
    pass
