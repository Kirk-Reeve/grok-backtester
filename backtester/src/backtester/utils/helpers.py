from typing import List, Dict, Any
from pydantic import BaseModel, field_validator, Field
from datetime import datetime
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class DataConfig(BaseModel):
    """Configuration for data fetching."""
    source: str
    symbols: List[str]
    start_date: str
    end_date: str
    cache_dir: str

    @field_validator('start_date', 'end_date', mode='before')  # V2 style
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date strings are in YYYY-MM-DD format."""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Must be YYYY-MM-DD.")

class StrategyConfig(BaseModel):
    """Configuration for the trading strategy."""
    type: str
    params: Dict[str, Any]  # Flexible; strategy-specific validation in strategy init

class BacktestConfig(BaseModel):
    """Configuration for the backtest engine."""
    initial_capital: float
    commission: float = Field(..., ge=0.0)  # >=0
    slippage: float = Field(0.0, ge=0.0)  # Default 0, >=0
    parallel: bool

class AppConfig(BaseModel):
    """Root configuration for the backtester application."""
    data: DataConfig
    strategy: StrategyConfig
    backtest: BacktestConfig
    
class BacktestError(Exception):
    """Base exception for backtester errors."""
    pass

class DataError(BacktestError):
    """Exception for data-related errors."""
    pass

class StrategyError(BacktestError):
    """Exception for strategy-related errors."""
    pass

class EngineError(BacktestError):
    """Exception for backtesting engine errors."""
    pass

class MetricsError(BacktestError):
    """Exception for metrics calculation errors."""
    pass

class VisualizationError(BacktestError):
    """Exception for visualization errors."""
    pass