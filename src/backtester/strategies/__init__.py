"""Initialization for strategies module."""

from typing import Dict
from .base import BaseStrategy
from .moving_average import MovingAverageStrategy
from .rsi import RSIStrategy

STRATEGY_REGISTRY: Dict[str, type[BaseStrategy]] = {
    "moving_average": MovingAverageStrategy,
    "rsi": RSIStrategy,
}