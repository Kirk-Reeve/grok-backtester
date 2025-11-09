import pandas as pd
import pytest

from backtester.engine.backtest import run_backtest, run_parallel_backtests
from backtester.strategies.moving_average import MovingAverageStrategy
from backtester.utils.helpers import EngineError


@pytest.fixture
def strategy():
    return MovingAverageStrategy({"short_window": 2, "long_window": 3})


def test_run_backtest(strategy, sample_data):
    """Test single backtest."""
    result = run_backtest(sample_data, strategy, 100000, 0.001, 0.0005)
    assert "portfolio" in result
    assert "metrics" in result
    assert isinstance(result["portfolio"], pd.DataFrame)
    assert "total" in result["portfolio"].columns
    assert result["portfolio"]["total"].iloc[-1] > 0


def test_run_backtest_invalid_data(strategy):
    """Test invalid data raises error."""
    invalid_data = pd.DataFrame()
    with pytest.raises(EngineError, match="invalid data"):
        run_backtest(invalid_data, strategy, 100000, 0.001, 0.0005)


def test_run_parallel_backtests(mocker):
    """Test parallel backtests with mocking, using sequential mode."""
    mock_run = mocker.patch("backtester.engine.backtest.run_backtest")
    mock_run.return_value = {"portfolio": pd.DataFrame(), "metrics": {}}

    strategy_config = {
        "type": "moving_average",
        "params": {"short_window": 2, "long_window": 3},
    }
    backtest_config = {
        "initial_capital": 100000,
        "commission": 0.001,
        "slippage": 0.0005,
        "parallel": False,
    }  # Fix: Sequential for mock compatibility

    datas = [pd.DataFrame({"Adj Close": [100, 101, 102]})] * 2
    results = run_parallel_backtests(datas, strategy_config, backtest_config)

    assert len(results) == 2
    assert mock_run.call_count == 2


def test_run_parallel_backtests_invalid_strategy():
    """Test invalid strategy raises error."""
    strategy_config = {"type": "invalid"}
    backtest_config = {
        "initial_capital": 100000,
        "commission": 0.001,
        "slippage": 0.0005,
        "parallel": False,
    }
    datas = [pd.DataFrame({"Adj Close": [100, 101]})]

    with pytest.raises(EngineError, match="Invalid strategy type"):
        run_parallel_backtests(datas, strategy_config, backtest_config)
