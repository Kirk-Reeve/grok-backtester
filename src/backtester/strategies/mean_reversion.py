"""Mean-Reversion trading strategy implementation."""

from typing import Any, Dict, Optional

from numpy import nan, where
from pandas import DataFrame, Series

from ..utils.logger import setup_logger
from .base import BaseStrategy

logger = setup_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean-Reversion Strategy.

    This strategy uses z-score on rolling mean and std to detect deviations.
    Buys when z-score < -threshold (oversold), sells when > threshold (overbought),
    holds otherwise. Vectorized with rolling operations.

    Parameters (via config):
    - window: int - Rolling window for mean and std (default: 20)
    - threshold: float - Z-score deviation threshold (default: 2.0)
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with parameters.

        :param params: Dictionary of strategy parameters.
        """
        super().__init__(params or {})
        self.window: int = self.params.get("window", 20)
        self.threshold: float = self.params.get("threshold", 2.0)
        if self.window <= 1:
            raise ValueError("Window must be greater than 1.")
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive.")
        logger.info(
            "Initialized MeanReversion with window=%s, threshold=%s",
            self.window,
            self.threshold,
        )

    def generate_signals(self, data: DataFrame) -> Series:
        """
        Generate trading signals based on z-score thresholds.

        :param data: DataFrame with 'Close' column for prices.
        :return: Series of signals (1: buy, -1: sell, 0: hold).
        """
        if "Close" not in data.columns:
            raise ValueError("DataFrame must contain 'Close' column.")

        # Calculate rolling mean and std vectorized
        rolling_mean = data["Close"].rolling(window=self.window).mean()
        rolling_std = data["Close"].rolling(window=self.window).std()

        # Z-score calculation (handle division by zero)
        z_score = (data["Close"] - rolling_mean) / rolling_std.replace(0, nan).fillna(
            1e-10
        )

        # Generate signals vectorized with np.where
        signals = where(z_score < -self.threshold, 1, 0)  # Buy on oversold
        signals = where(z_score > self.threshold, -1, signals)  # Sell on overbought

        signals_series = Series(signals, index=data.index).fillna(0)

        logger.debug(
            "Generated %s signals for %s to %s",
            len(signals_series),
            data.index[0],
            data.index[-1],
        )
        return signals_series
