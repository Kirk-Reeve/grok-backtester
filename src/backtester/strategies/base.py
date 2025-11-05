from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict
from ..utils.logger import setup_logger
from ..utils.helpers import StrategyError

logger = setup_logger(__name__)

class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    This class provides a common interface for all trading strategies.
    Subclasses are required to implement the `generate_signals` method,
    which defines the core logic of the strategy.

    Attributes:
        params (Dict[str, any]): A dictionary of parameters for the strategy.
    """

    def __init__(self, params: Dict[str, any]):
        """Initializes the BaseStrategy.

        Args:
            params (Dict[str, any]): A dictionary of parameters for configuring
                                     the strategy.

        Raises:
            StrategyError: If there is an error during strategy initialization.
        """
        try:
            self.params = params
            logger.debug(f"Initialized {self.__class__.__name__} with params: {params}")
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            raise StrategyError(f"Strategy initialization failed: {e}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generates target position signals based on market data.

        This method must be implemented by all subclasses. It should take a
        DataFrame of market data and return a Series of signals, where the
        signal represents the target position (e.g., 1 for long, -1 for short,
        0 for neutral).

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the historical
                                 market data for an asset.

        Returns:
            pd.Series: A pandas Series with the same index as the input data,
                       containing the trading signals.
        """
        pass
