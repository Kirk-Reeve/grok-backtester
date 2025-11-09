"""Initialization for strategies module."""

from typing import Dict

from .base import BaseStrategy
from .bollinger_bands import BollingerBandsStrategy
from .macd import MACDStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .moving_average import MovingAverageStrategy
from .rsi import RSIStrategy

STRATEGY_REGISTRY: Dict[str, type[BaseStrategy]] = {
    "moving_average": MovingAverageStrategy,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "bollinger_bands": BollingerBandsStrategy,
}
