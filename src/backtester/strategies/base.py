"""Base class for trading strategies."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from pandas import DataFrame, Series

from ..utils.helpers import StrategyError
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    This class provides a common interface for all trading strategies.
    Subclasses are required to implement the `generate_signals` method,
    which defines the core logic of the strategy.

    Attributes:
        params (Dict[str, Any]): A dictionary of parameters for the strategy.
    """

    def __init__(self, params: Dict[str, Any]):
        """Initializes the BaseStrategy.

        Args:
            params (Dict[str, Any]): A dictionary of parameters for configuring
                                     the strategy.

        Raises:
            StrategyError: If there is an error during strategy initialization.
        """
        try:
            self.params = params
            logger.debug(
                "Initialized %s{self.__class__.__name__} with params: %s{params}",
                self.__class__.__name__,
                params,
            )
        except Exception as error:
            logger.error("Error initializing strategy: %s", error)
            raise StrategyError(f"Strategy initialization failed: {error}") from error

    @abstractmethod
    def generate_signals(self, data: DataFrame) -> Series:
        """Generates target position signals based on market data.

        This method must be implemented by all subclasses. It should take a
        DataFrame of market data and return a Series of signals, where the
        signal represents the target position (e.g., 1 for long, -1 for short,
        0 for neutral).

        Args:
            data (DataFrame): A pandas DataFrame containing the historical
                                 market data for an asset.

        Returns:
            Series: A pandas Series with the same index as the input data,
                       containing the trading signals.
        """
