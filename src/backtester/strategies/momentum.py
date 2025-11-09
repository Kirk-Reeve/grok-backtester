"""Momentum trading strategy implementation."""

from typing import Any, Dict, Optional

from numpy import where
from pandas import DataFrame, Series

from ..utils.logger import setup_logger
from .base import BaseStrategy

logger = setup_logger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy.

    This strategy calculates the rate of change (momentum) over a lookback period.
    It generates buy signals when momentum > buy_threshold, sell when < sell_threshold,
    and hold otherwise. Fully vectorized for efficiency.

    Parameters (via config):
    - lookback: int - Period for momentum calculation (default: 20)
    - buy_threshold: float - Positive threshold for buy (default: 0.05)
    - sell_threshold: float - Negative threshold for sell (default: -0.05)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with parameters.

        :param params: Dictionary of strategy parameters.
        """
        super().__init__(params or {})
        self.lookback: int = self.params.get("lookback", 20)
        self.buy_threshold: float = self.params.get("buy_threshold", 0.05)
        self.sell_threshold: float = self.params.get("sell_threshold", -0.05)
        if self.lookback <= 0:
            raise ValueError("Lookback period must be positive.")
        if self.sell_threshold >= self.buy_threshold:
            raise ValueError("Sell threshold must be less than buy threshold.")
        logger.info(
            "Initialized Momentum with lookback=%s, buy_threshold=%s, sell_threshold=%s",
            self.lookback,
            self.buy_threshold,
            self.sell_threshold,
        )

    def generate_signals(self, data: DataFrame) -> Series:
        """
        Generate trading signals based on momentum thresholds.

        :param data: DataFrame with 'Close' column for prices.
        :return: Series of signals (1: buy, -1: sell, 0: hold).
        """
        if "Close" not in data.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

        # Calculate momentum using vectorized pct_change
        momentum = data["Close"].pct_change(periods=self.lookback).fillna(0)

        # Generate signals vectorized with np.where
        signals = where(momentum > self.buy_threshold, 1, 0)
        signals = where(momentum < self.sell_threshold, -1, signals)

        signals_series = Series(signals, index=data.index)

        logger.debug(
            "Generated %s signals for %s to %s",
            len(signals_series),
            data.index[0],
            data.index[-1],
        )
        return signals_series
