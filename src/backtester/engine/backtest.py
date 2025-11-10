"""Backtesting engine for simulating trading strategies on historical data."""

from typing import Any, Dict, List

from joblib import Parallel, delayed  # type: ignore[import-untyped]
from numpy import abs as _abs
from pandas import DataFrame

from ..metrics.performance import PerformanceCalculator
from ..strategies import STRATEGY_REGISTRY
from ..strategies.base import BaseStrategy
from ..utils.helpers import EngineError
from ..utils.logger import setup_logger

logger = setup_logger(__name__, file_path="backtest_engine.log")


def run_backtest(
    data: DataFrame,
    strategy: BaseStrategy,
    initial_capital: float,
    commission: float,
    slippage: float,
) -> Dict[str, Any]:
    """Runs a backtest for a single symbol using vectorized operations.

    This function simulates a trading strategy on historical data for a single
    financial instrument. It calculates the strategy's performance, including
    portfolio value over time and various performance metrics.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the historical
                             market data, including 'Adj Close' prices.
        strategy (BaseStrategy): An instance of a trading strategy that
                                 generates trading signals.
        initial_capital (float): The starting capital for the backtest.
        commission (float): The commission fee per trade as a fraction of the
                            trade value.
        slippage (float): The slippage per trade as a fraction of the trade
                          value.

    Returns:
        Dict[str, Any]: A dictionary containing the backtest results, including
                        the 'portfolio' DataFrame and a 'metrics' dictionary.

    Raises:
        EngineError: If the input data is insufficient or invalid, or if an
                     unexpected error occurs during the backtest.
    """
    try:
        if "Adj Close" not in data.columns or len(data) < 2:
            raise EngineError("Insufficient or invalid data for backtest")

        logger.info("Starting backtest for %s days", len(data))

        signals = strategy.generate_signals(data)

        positions = signals.shift(1).fillna(0)
        asset_returns = data["Adj Close"].pct_change().fillna(0)
        strategy_returns = positions * asset_returns

        position_diff = signals.diff().fillna(0)
        traded_fraction = _abs(position_diff)

        transaction_costs = (commission + slippage) * traded_fraction
        net_returns = strategy_returns - transaction_costs

        portfolio = DataFrame(index=data.index)
        portfolio["returns"] = net_returns
        portfolio["total"] = initial_capital * (1 + net_returns).cumprod()
        portfolio["holdings"] = positions * portfolio["total"]
        portfolio["cash"] = portfolio["total"] - portfolio["holdings"]

        calculator = PerformanceCalculator()
        metrics = calculator.calculate_metrics(portfolio)
        logger.info(
            "Backtest completed: final value %.2f, Sharpe %.2f",
            portfolio["total"].iloc[-1],
            metrics["sharpe_ratio"],
        )
        return {"portfolio": portfolio, "metrics": metrics}
    except EngineError as error:
        logger.error("Engine error: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error in backtest: %s", error)
        raise EngineError(f"Backtest failed: {error}") from error


def run_parallel_backtests(
    datas: List[DataFrame], strategy_config: Dict, backtest_config: Dict
) -> List[Dict]:
    """Runs backtests in parallel for multiple symbols.

    This function orchestrates the execution of backtests for multiple symbols,
    potentially in parallel, to improve performance.

    Args:
        datas (List[DataFrame]): A list of pandas DataFrames, where each
                                     DataFrame contains the historical market
                                     data for a single symbol.
        strategy_config (Dict): A dictionary containing the configuration for
                                the trading strategy, including its 'type' and
                                'params'.
        backtest_config (Dict): A dictionary containing the configuration for
                                the backtest, including 'initial_capital',
                                'commission', 'slippage', and 'parallel' flag.

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary contains the
                    backtest results for a single symbol.

    Raises:
        EngineError: If the strategy type is invalid or if an unexpected error
                     occurs during the parallel backtest execution.
    """
    try:
        strategy_type: str = strategy_config.get("type", "")
        strategy_class = STRATEGY_REGISTRY.get(strategy_type)
        if not strategy_class:
            raise EngineError(f"Invalid strategy type: {strategy_type}")

        strategy_params = strategy_config.get("params", {})
        strategies = [strategy_class(strategy_params) for _ in datas]

        logger.debug(
            "Running backtests for %s symbols (parallel: %s)",
            len(datas),
            backtest_config["parallel"],
        )

        if backtest_config["parallel"]:
            results = Parallel(n_jobs=-1)(
                delayed(run_backtest)(
                    data,
                    strategy,
                    backtest_config["initial_capital"],
                    backtest_config["commission"],
                    backtest_config["slippage"],
                )
                for data, strategy in zip(datas, strategies)
            )
        else:
            results = []
            for data, strategy in zip(datas, strategies):
                results.append(
                    run_backtest(
                        data,
                        strategy,
                        backtest_config["initial_capital"],
                        backtest_config["commission"],
                        backtest_config["slippage"],
                    )
                )

        logger.debug("All backtests completed")
        return results
    except EngineError as error:
        logger.error("Engine error in parallel backtests: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error in parallel backtests: %s", error)
        raise EngineError(f"Parallel backtests failed: {error}") from error
