"""Unit tests for strategies."""

from numpy import isin
from pandas import DataFrame, Series, date_range
from pytest import fixture, raises

from backtester.strategies.moving_average import MovingAverageStrategy
from backtester.utils.helpers import StrategyError


@fixture
def strategy():
    """Fixture that provides a configured MovingAverageStrategy instance for testing.

    This fixture creates and returns a MovingAverageStrategy object initialized
    with a short moving average window of 2 and a long moving average window of 3.
    It can be used by tests that require a ready-to-use trading strategy instance.

    Returns:
        MovingAverageStrategy: An instance of the MovingAverageStrategy class
        configured with predefined short and long window parameters.
    """
    return MovingAverageStrategy({"short_window": 2, "long_window": 3})


def test_generate_signals(strategy, sample_data):
    """Test signal generation."""
    signals = strategy.generate_signals(sample_data)
    assert isinstance(signals, Series)
    assert len(signals) == len(sample_data)
    assert isin(signals, [1.0, -1.0]).all()


def test_generate_signals_invalid_data(strategy):
    """Test invalid data raises error."""
    invalid_data = DataFrame({"Close": [1, 2]})
    with raises(StrategyError, match="missing 'Adj Close'"):
        strategy.generate_signals(invalid_data)


def test_generate_signals_edge_case(strategy):
    """Test small data set."""
    small_data = DataFrame(
        {"Adj Close": [100]}, index=date_range("2020-01-01", periods=1)
    )
    signals = strategy.generate_signals(small_data)
    assert len(signals) == 1
    assert signals.iloc[0] == -1.0  # Default to -1 if mavgs equal
