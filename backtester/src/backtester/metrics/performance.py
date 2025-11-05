import pandas as pd
import numpy as np
from typing import Dict
from ..utils.logger import setup_logger
from ..utils.helpers import MetricsError

logger = setup_logger(__name__)

def calculate_metrics(portfolio: pd.DataFrame) -> Dict[str, float]:
    """Calculates key performance metrics for a trading strategy.

    This function takes a portfolio DataFrame, which includes the returns and
    total value over time, and computes a variety of performance metrics to
    evaluate the trading strategy's effectiveness.

    Args:
        portfolio (pd.DataFrame): A pandas DataFrame with a DatetimeIndex,
                                  containing 'returns' and 'total' columns.
                                  'returns' are the periodic returns of the
                                  strategy, and 'total' is the cumulative
                                  portfolio value.

    Returns:
        Dict[str, float]: A dictionary containing the calculated performance
                          metrics, such as Sharpe ratio, CAGR, max drawdown,
                          and more.

    Raises:
        MetricsError: If the portfolio DataFrame is missing required columns
                      or if an unexpected error occurs during calculation.
    """
    try:
        if 'returns' not in portfolio.columns or 'total' not in portfolio.columns:
            raise MetricsError("Missing required columns in portfolio")

        if len(portfolio) < 2:
            logger.warning("Insufficient data for metrics; returning defaults")
            return _default_metrics()

        returns = portfolio['returns'].dropna()
        total = portfolio['total']

        logger.debug(f"Calculating metrics for {len(returns)} returns")

        total_return = (total.iloc[-1] / total.iloc[0]) - 1 if total.iloc[0] != 0 else 0.0

        num_days = len(returns)
        cagr = (total.iloc[-1] / total.iloc[0]) ** (252 / num_days) - 1 if total.iloc[0] != 0 and num_days > 0 else 0.0

        mean_return = returns.mean()
        std_dev = returns.std()
        sharpe_ratio = (mean_return * 252) / (std_dev * np.sqrt(252)) if std_dev != 0 else 0.0

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if not downside_returns.empty else 0.0
        sortino_ratio = (mean_return * 252) / (downside_std * np.sqrt(252)) if downside_std != 0 else 0.0

        peak = total.cummax()
        drawdown = (total - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        trades = returns[returns != 0]
        num_trades = len(trades)
        if num_trades == 0:
            logger.warning("No trades; trade metrics set to defaults")
            win_rate = 0.0
            avg_return_per_trade = 0.0
            profit_factor = np.nan
        else:
            wins = trades > 0
            win_rate = wins.sum() / num_trades
            avg_return_per_trade = trades.mean()
            gross_profit = trades[trades > 0].sum()
            gross_loss = abs(trades[trades < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'profit_factor': profit_factor,
        }

        logger.debug(f"Metrics calculated: {metrics}")
        return metrics
    except MetricsError as e:
        logger.error(f"Metrics error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in metrics: {e}")
        raise MetricsError(f"Metrics calculation failed: {e}")

def _default_metrics() -> Dict[str, float]:
    """Returns a dictionary with default values for all performance metrics.

    This function is used when there is insufficient data to calculate
    meaningful metrics, providing a consistent structure for the results.

    Returns:
        Dict[str, float]: A dictionary of performance metrics initialized to
                          zero or NaN.
    """
    return {
        'total_return': 0.0,
        'cagr': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'max_drawdown': 0.0,
        'calmar_ratio': 0.0,
        'num_trades': 0,
        'win_rate': 0.0,
        'avg_return_per_trade': 0.0,
        'profit_factor': np.nan,
    }
