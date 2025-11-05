import pytest
import logging
import numpy as np
import pandas as pd
from backtester.metrics.performance import calculate_metrics
from backtester.utils.helpers import MetricsError

def test_calculate_metrics(sample_portfolio):
    """Test metrics calculation."""
    metrics = calculate_metrics(sample_portfolio)
    assert isinstance(metrics, dict)
    assert 'sharpe_ratio' in metrics
    assert metrics['num_trades'] > 0
    assert np.isfinite(metrics['profit_factor']) or np.isinf(metrics['profit_factor'])

def test_calculate_metrics_invalid():
    """Test invalid portfolio raises error."""
    invalid_portfolio = pd.DataFrame({'invalid': [1]})
    with pytest.raises(MetricsError, match="Missing required columns"):
        calculate_metrics(invalid_portfolio)

def test_calculate_metrics_empty(caplog):
    """Test empty data returns defaults and warns."""
    empty_portfolio = pd.DataFrame({'returns': [], 'total': []})
    with caplog.at_level(logging.WARNING):
        metrics = calculate_metrics(empty_portfolio)
    assert "Insufficient data" in caplog.text
    assert metrics['sharpe_ratio'] == 0.0
    assert np.isnan(metrics['profit_factor'])

def test_calculate_metrics_no_trades(sample_portfolio, caplog):
    """Test no trades case warns."""
    no_trades_portfolio = sample_portfolio.copy()
    no_trades_portfolio['returns'] = 0.0
    with caplog.at_level(logging.WARNING):
        metrics = calculate_metrics(no_trades_portfolio)
    assert "No trades" in caplog.text
    assert metrics['num_trades'] == 0
    assert metrics['win_rate'] == 0.0
    assert np.isnan(metrics['profit_factor'])