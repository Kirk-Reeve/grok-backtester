# File: backtester/strategies/rsi.py
from typing import Dict
import pandas as pd
from .base import BaseStrategy
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class RSIStrategy(BaseStrategy):
    """
    Relative Strength Index (RSI) trading strategy.
    
    This strategy generates buy signals when RSI falls below the oversold threshold
    and sell signals when RSI rises above the overbought threshold. It supports
    customizable parameters for period, overbought, and oversold levels.
    
    Args:
        params: Dictionary of strategy parameters.
            - period: RSI calculation period (default: 14).
            - overbought: RSI level for overbought (sell) signal (default: 70).
            - oversold: RSI level for oversold (buy) signal (default: 30).
    
    Raises:
        ValueError: If parameters are invalid (e.g., period <= 0).
    """
    
    def __init__(self, params: Dict[str, any]):
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
        """
        Generate trading signals based on RSI.
        
        Args:
            data: Historical data DataFrame with 'Adj Close' column.
        
        Returns:
            Series of signals: 1 (buy/long), -1 (sell/short), 0 (hold).
        
        Raises:
            ValueError: If required columns are missing or data is insufficient.
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