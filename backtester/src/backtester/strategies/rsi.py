# File: backtester/strategies/rsi.py
from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class RSIStrategy(BaseStrategy):
    """A Relative Strength Index (RSI) trading strategy.

    This strategy generates trading signals based on the RSI indicator. A long
    signal (1.0) is generated when the RSI falls below an "oversold" threshold,
    and a short signal (-1.0) is generated when the RSI rises above an
    "overbought" threshold.

    The parameters for this strategy are:
        period (int): The number of periods for the RSI calculation.
                      Defaults to 14.
        overbought (int): The RSI level that is considered overbought.
                           Defaults to 70.
        oversold (int): The RSI level that is considered oversold.
                         Defaults to 30.
    """

    def __init__(self, params: Dict[str, any]):
        """Initializes the RSIStrategy.

        Args:
            params (Dict[str, any]): A dictionary of parameters for configuring
                                     the strategy.

        Raises:
            ValueError: If the provided parameters are invalid (e.g., period <= 0).
        """
        self.period = params.get('period', 14)
        self.overbought = params.get('overbought', 70)
        self.oversold = params.get('oversold', 30)

        if self.period <= 0:
            raise ValueError("RSI period must be positive")
        if self.oversold >= self.overbought:
            raise ValueError("Oversold threshold must be less than overbought")
        if not (0 <= self.oversold <= 100) or not (0 <= self.overbought <= 100):
            raise ValueError("RSI thresholds must be between 0 and 100")

        logger.info(f"RSI strategy initialized: period={self.period}, "
                    f"overbought={self.overbought}, oversold={self.oversold}")

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generates trading signals based on the RSI indicator.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the historical
                                 market data for an asset, including an
                                 'Adj Close' column.

        Returns:
            pd.Series: A pandas Series with the same index as the input data,
                       containing the trading signals (1.0 for long, -1.0 for
                       short, and 0.0 for neutral).

        Raises:
            ValueError: If the input data is missing the 'Adj Close' column,
                        if the data is insufficient for the RSI calculation, or
                        if an unexpected error occurs during signal generation.
        """
        if 'Adj Close' not in data.columns:
            logger.error("DataFrame missing 'Adj Close' column")
            raise ValueError("DataFrame must contain 'Adj Close' column")

        if len(data) < self.period + 1:
            logger.warning("Insufficient data for RSI calculation; returning neutral signals")
            return pd.Series(0, index=data.index)

        try:
            # Vectorized RSI calculation
            delta = data['Adj Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.period).mean()

            # Avoid division by zero
            rs = gain / loss
            rs = rs.replace([float('inf'), -float('inf')], 0).fillna(0)

            rsi = 100 - (100 / (1 + rs))

            # Generate signals: 1 if RSI < oversold, -1 if RSI > overbought, 0 otherwise
            signals = pd.Series(0, index=data.index)
            signals[rsi < self.oversold] = 1
            signals[rsi > self.overbought] = -1

            logger.debug(f"Generated {len(signals)} signals for RSI strategy")
            return signals
        except Exception as e:
            logger.error(f"Error generating RSI signals: {e}")
            raise ValueError(f"Signal generation failed: {e}")
