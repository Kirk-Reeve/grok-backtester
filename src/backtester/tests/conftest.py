import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Fixture for sample historical data DataFrame."""
    dates = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [105, 106, 107, 108, 109],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Adj Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=dates,
    )
    return data


@pytest.fixture
def sample_portfolio():
    """Fixture for sample portfolio DataFrame."""
    dates = pd.date_range("2020-01-01", periods=5)
    portfolio = pd.DataFrame(
        {
            "returns": [0.01, -0.005, 0.02, 0.015, -0.01],
            "total": [100000, 100995, 100490, 102499, 104036],  # Fixed: 5 values
        },
        index=dates,
    )
    return portfolio
