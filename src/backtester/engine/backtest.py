import pandas as pd
import numpy as np
from typing import Dict, List
from joblib import Parallel, delayed
from ..strategies.base import BaseStrategy
from ..strategies import STRATEGY_REGISTRY
from ..utils.logger import setup_logger
from ..metrics.performance import calculate_metrics
from ..utils.helpers import EngineError

logger = setup_logger(__name__)

def run_backtest(data: pd.DataFrame, strategy: BaseStrategy, initial_capital: float, commission: float, slippage: float) -> Dict[str, any]:
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
        Dict[str, any]: A dictionary containing the backtest results, including
                        the 'portfolio' DataFrame and a 'metrics' dictionary.

    Raises:
        EngineError: If the input data is insufficient or invalid, or if an
                     unexpected error occurs during the backtest.
    """
    try:
        if 'Adj Close' not in data.columns or len(data) < 2:
            raise EngineError("Insufficient or invalid data for backtest")

        logger.info(f"Starting backtest for {len(data)} days")

        signals = strategy.generate_signals(data)

        positions = signals.shift(1).fillna(0)
        asset_returns = data['Adj Close'].pct_change().fillna(0)
        strategy_returns = positions * asset_returns

        position_diff = signals.diff().fillna(0)
        traded_fraction = np.abs(position_diff)

        transaction_costs = (commission + slippage) * traded_fraction
        net_returns = strategy_returns - transaction_costs

        portfolio = pd.DataFrame(index=data.index)
        portfolio['returns'] = net_returns
        portfolio['total'] = initial_capital * (1 + net_returns).cumprod()
        portfolio['holdings'] = positions * portfolio['total']
        portfolio['cash'] = portfolio['total'] - portfolio['holdings']

        metrics = calculate_metrics(portfolio)
        logger.info(f"Backtest completed: final value {portfolio['total'].iloc[-1]:.2f}, Sharpe {metrics['sharpe_ratio']:.2f}")
        return {'portfolio': portfolio, 'metrics': metrics}
    except EngineError as e:
        logger.error(f"Engine error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in backtest: {e}")
        raise EngineError(f"Backtest failed: {e}")

def run_parallel_backtests(datas: List[pd.DataFrame], strategy_config: Dict, backtest_config: Dict) -> List[Dict]:
    """Runs backtests in parallel for multiple symbols.

    This function orchestrates the execution of backtests for multiple symbols,
    potentially in parallel, to improve performance.

    Args:
        datas (List[pd.DataFrame]): A list of pandas DataFrames, where each
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
        strategy_type = strategy_config.get('type')
        strategy_class = STRATEGY_REGISTRY.get(strategy_type)
        if not strategy_class:
            raise EngineError(f"Invalid strategy type: {strategy_type}")

        strategy_params = strategy_config.get('params', {})
        strategies = [strategy_class(strategy_params) for _ in datas]

        logger.info(f"Running backtests for {len(datas)} symbols (parallel: {backtest_config['parallel']})")

        if backtest_config['parallel']:
            results = Parallel(n_jobs=-1)(
                delayed(run_backtest)(
                    data, strat, backtest_config['initial_capital'], backtest_config['commission'], backtest_config['slippage']
                ) for data, strat in zip(datas, strategies)
            )
        else:
            results = []
            for data, strat in zip(datas, strategies):
                results.append(run_backtest(data, strat, backtest_config['initial_capital'], backtest_config['commission'], backtest_config['slippage']))

        logger.info("All backtests completed")
        return results
    except EngineError as e:
        logger.error(f"Engine error in parallel backtests: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in parallel backtests: {e}")
        raise EngineError(f"Parallel backtests failed: {e}")
