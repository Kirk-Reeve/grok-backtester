"""Base class for trading strategies."""

from abc import ABC, abstractmethod
from types import MappingProxyType  # For immutable dict proxy (stdlib, secure)
from typing import Any, Dict, List, Optional

from pandas import DataFrame, Series

from ..utils.logger import setup_logger


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    This class provides a common interface for all trading strategies.
    Subclasses are required to implement the `generate_signals` method,
    which defines the core logic of the strategy.

    Args:
        params (Optional[Dict[str, Any]]): A dictionary of parameters for the strategy.
            Defaults to an empty dictionary if not provided.

    Attributes:
        params (MappingProxyType): An immutable dictionary of parameters for the strategy.
        logger (Logger): Instance-specific logger for the strategy.
    """

    # Subclasses can override with their defaults (e.g., {'fast_period': 12})
    default_params: Dict[str, Any] = {}

    # Subclasses override with required DataFrame columns
    # (e.g., ['Close'] or ['High', 'Low', 'Close'])
    required_columns: List[str] = []

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """Initialize the base strategy with parameters.

        Merges user params with class defaults, validates, sets up immutable params and logger.
        Logs initialization and handles errors gracefully.

        Args:
            params (Optional[Dict[str, Any]]): Dictionary of strategy parameters.

        Raises:
            ValueError: If params is not a dictionary or validation fails.
        """
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary.")

        # Merge defaults with user params (user overrides defaults)
        effective_params = self.default_params.copy()
        effective_params.update(params)
        self._validate_params(effective_params)  # Hook for subclass validation

        # Make params immutable for security
        self.params = MappingProxyType(effective_params)

        # Setup instance logger using class name for better traceability
        self.logger = setup_logger(
            self.__class__.__name__, file_path=f"{self.__class__.__name__.lower()}.log"
        )
        self.logger.debug(
            "Initialized %s with params: %s",
            self.__class__.__name__,
            effective_params,
        )

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validate strategy parameters.

        Subclasses can override to add specific checks (e.g., param types/values).
        Base implementation is a no-op.

        Args:
            params (Dict[str, Any]): The parameters to validate.

        Raises:
            ValueError: If parameters are invalid.
        """

    @abstractmethod
    def generate_signals(self, data: DataFrame) -> Series:
        """Generate target position signals based on market data.

        This method must be implemented by all subclasses. It should take a
        DataFrame of market data and return a Series of signals, where the
        signal represents the target position (e.g., 1 for buy, -1 for sell,
        0 for hold). Handle NaNs gracefully (e.g., 0 for insufficient data).

        Args:
            data (DataFrame): Historical market data for the asset(s).

        Returns:
            Series: Trading signals aligned with the data index.

        Raises:
            ValueError: If data is missing required columns or insufficient.
        """
        # Common validation: Check required columns
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(
                f"DataFrame missing required columns: {', '.join(missing_cols)}"
            )
        return Series(0, index=data.index, dtype="int8")  # Default neutral signals

    def get_name(self) -> str:
        """Get the name of the strategy.

        Returns:
            str: The class name of the strategy.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        """Representation for debugging.

        Returns:
            str: String representation including class and params.
        """
        return f"{self.__class__.__name__}(params={dict(self.params)})"
