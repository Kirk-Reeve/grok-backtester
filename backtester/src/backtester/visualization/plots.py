import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from pathlib import Path
from ..utils.logger import setup_logger
from ..strategies.base import BaseStrategy
from ..utils.helpers import VisualizationError

logger = setup_logger(__name__)

def plot_equity_curve(portfolio: pd.DataFrame, title: str = "Equity Curve", save_path: Optional[str] = None, display: bool = False) -> None:
    """Plot the portfolio's total value over time.

    Args:
        portfolio: DataFrame with 'total' column for portfolio value.
        title: Plot title (default: "Equity Curve").
        save_path: Optional file path to save the plot (e.g., 'equity_curve.png').
        display: If True, show the plot interactively (default: False).

    Raises:
        ValueError: If required columns are missing.
    """
    if 'total' not in portfolio.columns:
        logger.error("Portfolio DataFrame missing 'total' column")
        raise ValueError("Portfolio DataFrame must contain 'total' column")

    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(portfolio.index, portfolio['total'], label='Portfolio Value', color='blue')
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Equity curve saved to {save_path}")

        if display:
            plt.show()

        plt.close(fig)
        logger.debug("Equity curve plotted successfully")
    except Exception as e:
        logger.error(f"Error plotting equity curve: {e}")
        raise

def plot_drawdown(portfolio: pd.DataFrame, title: str = "Drawdown", save_path: Optional[str] = None, display: bool = False) -> None:
    """Plot the portfolio drawdown over time.

    Args:
        portfolio: DataFrame with 'total' column.
        title: Plot title (default: "Drawdown").
        save_path: Optional file path to save the plot.
        display: If True, show the plot interactively (default: False).

    Raises:
        ValueError: If required columns are missing.
    """
    if 'total' not in portfolio.columns:
        logger.error("Portfolio DataFrame missing 'total' column")
        raise ValueError("Portfolio DataFrame must contain 'total' column")

    try:
        # Calculate drawdown vectorized
        peak = portfolio['total'].cummax()
        drawdown = (portfolio['total'] - peak) / peak

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(portfolio.index, drawdown, 0, color='red', alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Drawdown plot saved to {save_path}")

        if display:
            plt.show()

        plt.close(fig)
        logger.debug("Drawdown plotted successfully")
    except Exception as e:
        logger.error(f"Error plotting drawdown: {e}")
        raise

def plot_price_with_signals(data: pd.DataFrame, signals: pd.Series, symbol: str, title: Optional[str] = None, save_path: Optional[str] = None, display: bool = False) -> None:
    """Plot stock price with buy/sell signals.

    Args:
        data: Historical data DataFrame with 'Adj Close' column.
        signals: Series of positions (1: long, -1: short).
        symbol: Stock symbol for labeling.
        title: Plot title (default: f"{symbol} Price with Signals").
        save_path: Optional file path to save the plot.
        display: If True, show the plot interactively (default: False).

    Raises:
        ValueError: If required columns are missing.
    """
    if 'Adj Close' not in data.columns:
        logger.error("Data DataFrame missing 'Adj Close' column")
        raise ValueError("Data DataFrame must contain 'Adj Close' column")

    if title is None:
        title = f"{symbol} Price with Signals"

    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Adj Close'], label='Adj Close Price', color='black')

        # Buy signals (where signals == 1 and previous !=1)
        buy_signals = signals[(signals == 1) & (signals.shift(1) != 1)]
        ax.scatter(buy_signals.index, data.loc[buy_signals.index, 'Adj Close'], marker='^', color='green', label='Buy', s=100)

        # Sell signals (where signals == -1 and previous !=-1)
        sell_signals = signals[(signals == -1) & (signals.shift(1) != -1)]
        ax.scatter(sell_signals.index, data.loc[sell_signals.index, 'Adj Close'], marker='v', color='red', label='Sell', s=100)

        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Price with signals plot saved to {save_path}")

        if display:
            plt.show()

        plt.close(fig)
        logger.debug("Price with signals plotted successfully")
    except Exception as e:
        logger.error(f"Error plotting price with signals: {e}")
        raise

def generate_backtest_report(results: List[Dict[str, any]], symbols: List[str], data_dict: Dict[str, pd.DataFrame], strategies: List[BaseStrategy], output_dir: str = 'reports', save_plots: bool = True, display_plots: bool = False) -> None:
    """Generate a full report with plots."""
    try:
        if len(results) != len(symbols) or len(results) != len(data_dict) or len(results) != len(strategies):
            raise VisualizationError("Input lengths mismatch")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating report for {len(symbols)} symbols in {output_dir}")

        for i, (result, symbol, data, strategy) in enumerate(zip(results, symbols, data_dict.values(), strategies)):
            logger.debug(f"Processing visuals for {symbol}")
            portfolio = result['portfolio']
            signals = strategy.generate_signals(data)

            equity_path = f"{output_dir}/{symbol}_equity_curve.png" if save_plots else None
            drawdown_path = f"{output_dir}/{symbol}_drawdown.png" if save_plots else None
            signals_path = f"{output_dir}/{symbol}_signals.png" if save_plots else None

            plot_equity_curve(portfolio, title=f"{symbol} Equity Curve", save_path=equity_path, display=display_plots)
            plot_drawdown(portfolio, title=f"{symbol} Drawdown", save_path=drawdown_path, display=display_plots)
            plot_price_with_signals(data, signals, symbol, save_path=signals_path, display=display_plots)

            logger.info(f"Metrics for {symbol}: {result['metrics']}")

        logger.info(f"Report generated (save: {save_plots}, display: {display_plots})")
    except VisualizationError as e:
        logger.error(f"Visualization error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in report generation: {e}")
        raise VisualizationError(f"Report generation failed: {e}")