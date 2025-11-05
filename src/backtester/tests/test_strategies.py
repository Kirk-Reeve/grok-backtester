import pytest
import pandas as pd
import numpy as np
from backtester.strategies.moving_average import MovingAverageStrategy
from backtester.utils.helpers import StrategyError

@pytest.fixture
def strategy():
    return MovingAverageStrategy({'short_window': 2, 'long_window': 3})

def test_generate_signals(strategy, sample_data):
    """Test signal generation."""
    signals = strategy.generate_signals(sample_data)
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(sample_data)
    assert np.isin(signals, [1.0, -1.0]).all()

def test_generate_signals_invalid_data(strategy):
    """Test invalid data raises error."""
    invalid_data = pd.DataFrame({'Close': [1, 2]})
    with pytest.raises(StrategyError, match="missing 'Adj Close'"):
        strategy.generate_signals(invalid_data)

def test_generate_signals_edge_case(strategy):
    """Test small data set."""
    small_data = pd.DataFrame({'Adj Close': [100]}, index=pd.date_range('2020-01-01', periods=1))
    signals = strategy.generate_signals(small_data)
    assert len(signals) == 1
    assert signals.iloc[0] == -1.0  # Default to -1 if mavgs equal