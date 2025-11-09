"""Visualization functions for backtesting results."""

from os.path import abspath
from pathlib import Path
from typing import Any, Dict, List, Optional

from matplotlib.pyplot import close, savefig, show, subplots
from pandas import DataFrame, Series

from ..strategies.base import BaseStrategy
from ..utils.helpers import VisualizationError, get_project_root
from ..utils.logger import setup_logger

logger = setup_logger(__name__, file_path="visualization.log")


def plot_equity_curve(
    portfolio: DataFrame,
    title: str = "Equity Curve",
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """Plots the portfolio's total value over time.

    Args:
        portfolio (DataFrame): A pandas DataFrame with a 'total' column
                                  representing the portfolio value.
        title (str): The title for the plot. Defaults to "Equity Curve".
        save_path (Optional[str]): The file path to save the plot to. If None,
                                   the plot is not saved. Defaults to None.
        display (bool): If True, the plot is displayed interactively. Defaults
                        to False.

    Raises:
        ValueError: If the portfolio DataFrame is missing the 'total' column.
    """
    if "total" not in portfolio.columns:
        logger.error("Portfolio DataFrame missing 'total' column")
        raise ValueError("Portfolio DataFrame must contain 'total' column")

    try:
        figure, axes = subplots(figsize=(12, 6))
        axes.plot(
            portfolio.index, portfolio["total"], label="Portfolio Value", color="blue"
        )
        axes.set_title(title)
        axes.set_xlabel("Date")
        axes.set_ylabel("Portfolio Value")
        axes.legend()
        axes.grid(True)

        if save_path:
            abs_path = abspath(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            savefig(save_path)
            logger.debug("Equity curve saved to: %s", abs_path)

        if display:
            show()

        close(figure)
        logger.debug("Equity curve plotted successfully")
    except Exception as error:
        logger.error("Error plotting equity curve: %s", error)
        raise


def plot_drawdown(
    portfolio: DataFrame,
    title: str = "Drawdown",
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """Plots the portfolio drawdown over time.

    Args:
        portfolio (DataFrame): A pandas DataFrame with a 'total' column.
        title (str): The title for the plot. Defaults to "Drawdown".
        save_path (Optional[str]): The file path to save the plot to. If None,
                                   the plot is not saved. Defaults to None.
        display (bool): If True, the plot is displayed interactively. Defaults
                        to False.

    Raises:
        ValueError: If the portfolio DataFrame is missing the 'total' column.
    """
    if "total" not in portfolio.columns:
        logger.error("Portfolio DataFrame missing 'total' column")
        raise ValueError("Portfolio DataFrame must contain 'total' column")

    try:
        # Calculate drawdown vectorized
        peak = portfolio["total"].cummax()
        drawdown = (portfolio["total"] - peak) / peak

        figure, axes = subplots(figsize=(12, 6))
        axes.fill_between(portfolio.index, drawdown, 0, color="red", alpha=0.5)
        axes.set_title(title)
        axes.set_xlabel("Date")
        axes.set_ylabel("Drawdown")
        axes.grid(True)

        if save_path:
            abs_path = abspath(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            savefig(save_path)
            logger.info("Drawdown plot saved to %s", abs_path)

        if display:
            show()

        close(figure)
        logger.debug("Drawdown plotted successfully")
    except Exception as error:
        logger.error("Error plotting drawdown: %s", error)
        raise


def plot_price_with_signals(
    data: DataFrame,
    signals: Series,
    symbol: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    display: bool = False,
) -> None:
    """Plots the stock price with buy and sell signals.

    Args:
        data (DataFrame): A pandas DataFrame with historical market data,
                             including an 'Adj Close' column.
        signals (Series): A pandas Series containing the trading signals
                             (1 for long, -1 for short).
        symbol (str): The stock symbol for labeling.
        title (Optional[str]): The title for the plot. Defaults to
                               f"{symbol} Price with Signals".
        save_path (Optional[str]): The file path to save the plot to. If None,
                                   the plot is not saved. Defaults to None.
        display (bool): If True, the plot is displayed interactively. Defaults
                        to False.

    Raises:
        ValueError: If the data DataFrame is missing the 'Adj Close' column.
    """
    if "Adj Close" not in data.columns:
        logger.error("Data DataFrame missing 'Adj Close' column")
        raise ValueError("Data DataFrame must contain 'Adj Close' column")

    if title is None:
        title = f"{symbol} Price with Signals"

    try:
        figure, axes = subplots(figsize=(12, 6))
        axes.plot(data.index, data["Adj Close"], label="Adj Close Price", color="black")

        # Buy signals (where signals == 1 and previous !=1)
        buy_signals = signals[(signals == 1) & (signals.shift(1) != 1)]
        axes.scatter(
            buy_signals.index,
            data.loc[buy_signals.index, "Adj Close"],
            marker="^",
            color="green",
            label="Buy",
            s=100,
        )

        # Sell signals (where signals == -1 and previous !=-1)
        sell_signals = signals[(signals == -1) & (signals.shift(1) != -1)]
        axes.scatter(
            sell_signals.index,
            data.loc[sell_signals.index, "Adj Close"],
            marker="v",
            color="red",
            label="Sell",
            s=100,
        )

        axes.set_title(title)
        axes.set_xlabel("Date")
        axes.set_ylabel("Price")
        axes.legend()
        axes.grid(True)

        if save_path:
            abs_path = abspath(save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            savefig(save_path)
            logger.info("Price with signals plot saved to %s", abs_path)

        if display:
            show()

        close(figure)
        logger.debug("Price with signals plotted successfully")
    except Exception as error:
        logger.error("Error plotting price with signals: %s", error)
        raise


def generate_backtest_report(
    results: List[Dict[str, Any]],
    symbols: List[str],
    data_dict: Dict[str, DataFrame],
    strategies: List[BaseStrategy],
    save_plots: bool = True,
    display_plots: bool = False,
) -> None:
    """Generates a full report with plots for a backtest.

    Args:
        results (List[Dict[str, any]]): A list of backtest result dictionaries.
        symbols (List[str]): A list of stock symbols.
        data_dict (Dict[str, DataFrame]): A dictionary of historical data
                                             DataFrames.
        strategies (List[BaseStrategy]): A list of strategy instances.
        save_plots (bool): If True, the plots are saved to the output
                           directory. Defaults to True.
        display_plots (bool): If True, the plots are displayed interactively.
                              Defaults to False.

    Raises:
        VisualizationError: If the lengths of the input lists do not match or
                            if an unexpected error occurs.
    """
    try:
        project_root = get_project_root()
        abs_output_dir = project_root / "reports" / strategies[0].__class__.__name__

        if (
            len(results) != len(symbols)
            or len(results) != len(data_dict)
            or len(results) != len(strategies)
        ):
            raise VisualizationError("Input lengths mismatch")

        abs_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating report for %s symbols in reports", len(symbols))

        for _, (result, symbol, data, strategy) in enumerate(
            zip(results, symbols, data_dict.values(), strategies)
        ):
            logger.debug("Processing visuals for %s", symbol)
            portfolio = result["portfolio"]
            signals = strategy.generate_signals(data)

            equity_path = (
                str(abs_output_dir / symbol / "equity_curve.png")
                if save_plots
                else None
            )
            drawdown_path = (
                str(abs_output_dir / symbol / "drawdown.png") if save_plots else None
            )
            signals_path = (
                str(abs_output_dir / symbol / "signals.png") if save_plots else None
            )

            plot_equity_curve(
                portfolio,
                title=f"{symbol} Equity Curve",
                save_path=equity_path,
                display=display_plots,
            )
            plot_drawdown(
                portfolio,
                title=f"{symbol} Drawdown",
                save_path=drawdown_path,
                display=display_plots,
            )
            plot_price_with_signals(
                data, signals, symbol, save_path=signals_path, display=display_plots
            )

            logger.info("Metrics for %s: %s", symbol, result["metrics"])

        logger.info(
            "Report generated (save: %s, display: %s)", save_plots, display_plots
        )
    except VisualizationError as error:
        logger.error("Visualization error: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error in report generation: %s", error)
        raise VisualizationError(f"Report generation failed: {error}") from error
