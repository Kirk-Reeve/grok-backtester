from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict
from ..utils.logger import setup_logger
from ..utils.helpers import StrategyError

logger = setup_logger(__name__)

class BaseStrategy(ABC):
    """Base class for trading strategies."""

    def __init__(self, params: Dict[str, any]):
        """Initialize the strategy.

        Args:
            params: Strategy parameters.
        """
        try:
            self.params = params
            logger.debug(f"Initialized {self.__class__.__name__} with params: {params}")
        except Exception as e:
            logger.error(f"Error initializing strategy: {e}")
            raise StrategyError(f"Strategy initialization failed: {e}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate target position signals."""
        pass