"""Initialization for strategies module."""

from typing import Dict

from .base import BaseStrategy
from .bollinger_bands import BollingerBandsStrategy
from .commodity_channel_index import CCIStrategy
from .macd import MACDStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .moving_average import MovingAverageStrategy
from .parabolic_sar import ParabolicSARStrategy
from .rsi import RSIStrategy
from .rsi_v2 import EnhancedRSIStrategy
from .stochastic import StochasticStrategy

STRATEGY_REGISTRY: Dict[str, type[BaseStrategy]] = {
    "moving_average": MovingAverageStrategy,
    "rsi": RSIStrategy,
    "macd": MACDStrategy,
    "momentum": MomentumStrategy,
    "mean_reversion": MeanReversionStrategy,
    "bollinger_bands": BollingerBandsStrategy,
    "commodity_channel_index": CCIStrategy,
    "parabolic_sar": ParabolicSARStrategy,
    "stochastic": StochasticStrategy,
    "enhanced_rsi": EnhancedRSIStrategy,
}
