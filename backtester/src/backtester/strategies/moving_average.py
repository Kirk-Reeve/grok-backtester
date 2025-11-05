import pandas as pd
import numpy as np
from typing import Dict
from .base import BaseStrategy
from ..utils.logger import setup_logger
from ..utils.helpers import StrategyError

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

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generates trading signals based on the SMA crossover logic.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the historical
                                 market data for an asset, including an
                                 'Adj Close' column.

        Returns:
            pd.Series: A pandas Series with the same index as the input data,
                       containing the trading signals (1.0 for long, -1.0 for
                       short).

        Raises:
            StrategyError: If the input data is missing the 'Adj Close' column
                           or if an unexpected error occurs during signal
                           generation.
        """
        try:
            if 'Adj Close' not in data.columns:
                raise StrategyError("Data missing 'Adj Close' column")

            short_window = self.params.get('short_window', 50)
            long_window = self.params.get('long_window', 200)
            if short_window >= long_window:
                logger.warning("Short window >= long window; may lead to invalid signals")

            logger.debug(f"Generating signals for {len(data)} rows with windows {short_window}/{long_window}")

            short_mavg = data['Adj Close'].rolling(window=short_window, min_periods=1).mean()
            long_mavg = data['Adj Close'].rolling(window=long_window, min_periods=1).mean()

            signals = pd.Series(np.where(short_mavg > long_mavg, 1.0, -1.0), index=data.index)

            logger.debug(f"Generated {signals.sum()} long signals")
            return signals
        except StrategyError as e:
            logger.error(f"Strategy error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in signal generation: {e}")
            raise StrategyError(f"Signal generation failed: {e}")
