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
    """Run backtest for a single symbol using vectorized operations."""
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
    """Run backtests in parallel for multiple symbols."""
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