import pandas as pd
import numpy as np
from typing import Dict
from .base import BaseStrategy
from ..utils.logger import setup_logger
from ..utils.helpers import StrategyError

logger = setup_logger(__name__)

class MovingAverageStrategy(BaseStrategy):
    """Simple moving average crossover strategy."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate signals based on SMA crossover."""
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