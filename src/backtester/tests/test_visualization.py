import pytest
import matplotlib
from backtester.visualization.plots import plot_equity_curve, plot_drawdown
from backtester.utils.helpers import VisualizationError
matplotlib.use('Agg')  # Set non-interactive backend for tests

def test_plot_equity_curve(mocker, sample_portfolio):
    """Test equity curve plot."""
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.savefig')
    plot_equity_curve(sample_portfolio, display=False)
    # No raise means success

def test_plot_drawdown(mocker, sample_portfolio):
    """Test drawdown plot."""
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.savefig')
    plot_drawdown(sample_portfolio, display=False)
    # No raise means success