"""Moving Average Crossover trading strategy implementation."""

from typing import Any, Dict

from numpy import where
from pandas import DataFrame, Series

from ..utils.helpers import StrategyError
from ..utils.logger import setup_logger
from .base import BaseStrategy

logger = setup_logger(__name__)


class MovingAverageStrategy(BaseStrategy):
    """A simple moving average (SMA) crossover trading strategy.

    This strategy generates trading signals based on the crossover of two SMAs
    with different time windows: a short-term SMA and a long-term SMA. A long
    signal (1.0) is generated when the short-term SMA crosses above the
    long-term SMA, and a short signal (-1.0) is generated when it crosses below.

    The parameters for this strategy are:
        short_window (int): The number of periods for the short-term SMA.
                            Defaults to 50.
        long_window (int): The number of periods for the long-term SMA.
                           Defaults to 200.
    """

    def __init__(self, params: Dict[str, Any]):
        """Initializes the MovingAverageStrategy.

        Args:
            params (Dict[str, any]): A dictionary of parameters for configuring
                                     the strategy.

        Raises:
            ValueError: If the provided parameters are invalid (e.g., short_window
                        >= long_window).
        """
        super().__init__(params or {})
        self.short_window: int = params.get("short_window", 50)
        self.long_window: int = params.get("long_window", 200)

        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")

        logger.info(
            "Moving Average Strategy initialized: short_window=%s, long_window=%s",
            self.short_window,
            self.long_window,
        )

    def generate_signals(self, data: DataFrame) -> Series:
        """Generates trading signals based on the SMA crossover logic.

        Args:
            data (DataFrame): A pandas DataFrame containing the historical
                                 market data for an asset, including an
                                 'Adj Close' column.

        Returns:
            Series: A pandas Series with the same index as the input data,
                       containing the trading signals (1.0 for long, -1.0 for
                       short).

        Raises:
            StrategyError: If the input data is missing the 'Adj Close' column
                           or if an unexpected error occurs during signal
                           generation.
        """
        try:
            if "Adj Close" not in data.columns:
                raise StrategyError("Data missing 'Adj Close' column")

            if self.short_window >= self.long_window:
                logger.warning(
                    "Short window >= long window; may lead to invalid signals"
                )

            logger.debug(
                "Generating signals for %s rows with windows %s/%s",
                len(data),
                self.short_window,
                self.long_window,
            )

            short_mavg = (
                data["Adj Close"]
                .rolling(window=self.short_window, min_periods=1)
                .mean()
            )
            long_mavg = (
                data["Adj Close"].rolling(window=self.long_window, min_periods=1).mean()
            )

            signals = Series(where(short_mavg > long_mavg, 1.0, -1.0), index=data.index)

            logger.debug("Generated %s long signals", signals.sum())
            return signals
        except StrategyError as error:
            logger.error("Strategy error: %s", error)
            raise
        except Exception as error:
            logger.error("Unexpected error in signal generation: %s", error)
            raise StrategyError(f"Signal generation failed: {error}") from error
