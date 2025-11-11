"""Parameter optimization module for the trading backtester.

This module provides a grid search optimizer to find the best hyperparameters
for a given strategy by evaluating performance metrics (e.g., Sharpe ratio)
across a grid of parameter combinations. It supports parallel execution for
efficiency on multi-core systems.

Key Features:
- Modular integration with STRATEGY_REGISTRY and run_parallel_backtests.
- Parallel processing using concurrent.futures for speed.
- Configurable objective metric (e.g., 'sharpe_ratio' to maximize).
- Logging of progress and results.
- Handles multiple symbols and data splits.

Usage:
    optimizer = GridSearchOptimizer(
        strategy_type='macd',  # Key in STRATEGY_REGISTRY
        param_grid={
            'fast_period': [8, 12, 16],
            'slow_period': [20, 26, 32],
            # etc.
        },
        objective_metric='sharpe_ratio',
        data=train_data,  # Dict[symbol: DataFrame]
        backtest_config=backtest_config,  # Dict with initial_capital, etc.
        max_workers=4,  # Parallel threads
    )
    best_params, best_score = optimizer.optimize()
"""

import concurrent.futures
import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..engine.backtest import run_parallel_backtests
from ..metrics.performance import PerformanceCalculator
from ..strategies import STRATEGY_REGISTRY
from ..utils.logger import setup_logger


class GridSearchOptimizer:
    """Grid search optimizer for trading strategy parameters.

    Performs exhaustive search over a parameter grid, evaluates each combination
    using run_parallel_backtests, and selects the best based on a specified metric.

    Args:
        strategy_type (str): Registered strategy type (key in STRATEGY_REGISTRY).
        param_grid (Dict[str, List[Any]]): Grid of parameters to search
            (key: param name, value: list of values).
        objective_metric (str): Metric to optimize (e.g., 'sharpe_ratio'; from
            PerformanceCalculator). Maximized by default; prefix with '-' to
            minimize (e.g., '-max_drawdown').
        data (Dict[str, pd.DataFrame]): Historical data for symbols (key: symbol, value: DataFrame).
        backtest_config (Dict[str, Any]): Backtest config (initial_capital, commission,
            slippage, parallel).
        risk_free_rate (float): Risk-free rate for metrics (default: 0.0).
        max_workers (Optional[int]): Max parallel workers (default: None; uses all cores).
        verbose (bool): Log progress (default: True).

    Attributes:
        strategy_type (str): Stored strategy type.
        param_grid (Dict[str, List[Any]]): Stored parameter grid.
        objective_metric (str): Stored objective metric.
        data (Dict[str, pd.DataFrame]): Stored data.
        backtest_config (Dict[str, Any]): Stored backtest config.
        risk_free_rate (float): Stored risk-free rate.
        max_workers (Optional[int]): Stored max workers.
        verbose (bool): Verbosity flag.
        logger (logging.Logger): Optimizer-specific logger.
    """

    def __init__(
        self,
        strategy_type: str,
        param_grid: Dict[str, List[Any]],
        objective_metric: str = "sharpe_ratio",
        data: Optional[Dict[str, pd.DataFrame]] = None,
        backtest_config: Optional[Dict[str, Any]] = None,
        risk_free_rate: float = 0.0,
        max_workers: Optional[int] = None,
        verbose: bool = True,
    ):
        if not param_grid:
            raise ValueError("param_grid must be non-empty.")
        if strategy_type not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

        self.strategy_type = strategy_type
        self.param_grid = param_grid
        self.objective_metric = objective_metric
        self.data = data or {}
        self.backtest_config = backtest_config or {}
        self.risk_free_rate = risk_free_rate
        self.max_workers = max_workers
        self.verbose = verbose
        self.logger = setup_logger(__name__, file_path="grid_search_optimizer.log")
        self.logger.debug(
            "Initialized GridSearchOptimizer for %s with grid size %d, objective '%s'.",
            strategy_type,
            self._grid_size(),
            objective_metric,
        )

    def _grid_size(self) -> int:
        """Calculate total number of parameter combinations.

        Returns:
            int: Total grid points.
        """
        size = 1
        for values in self.param_grid.values():
            size *= len(values)
        return size

    def _generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations from the grid.

        Uses itertools.product for efficient iteration.

        Returns:
            List[Dict[str, Any]]: List of param dicts.
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        return combos

    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """Evaluate a single parameter set by running backtest and extracting metric.

        Args:
            params (Dict[str, Any]): Parameter dictionary for strategy.

        Returns:
            float: Objective metric value (higher better; negated if minimizing).
        """
        try:
            # Prepare datas as list for run_parallel_backtests
            datas = list(self.data.values())

            # Strategy config for registry-based instantiation
            strategy_config = {"type": self.strategy_type, "params": params}

            # Run backtests
            results = run_parallel_backtests(
                datas, strategy_config, self.backtest_config
            )

            # Aggregate metrics across symbols (e.g., average objective metric)
            metric_calculator = PerformanceCalculator(
                risk_free_rate=self.risk_free_rate
            )
            metrics_list = [
                metric_calculator.calculate_metrics(res["portfolio"]) for res in results
            ]
            avg_metric: float = float(
                np.mean([m[self.objective_metric] for m in metrics_list])
            )

            # Ensure a native Python float for consistent typing and mypy compatibility

            # If metric to minimize (e.g., '-max_drawdown'), negate
            if self.objective_metric.startswith("-"):
                return float(-avg_metric)  # To maximize the negated value
            return float(avg_metric)

        except (KeyError, TypeError, ValueError, RuntimeError) as e:
            self.logger.exception("Evaluation failed for params %s: %s", params, e)
            return -np.inf  # Penalize failed combos

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """Perform grid search optimization in parallel.

        Returns:
            Tuple[Dict[str, Any], float]: Best parameters and best score.
        """
        combos = self._generate_combinations()
        if self.verbose:
            self.logger.info("Optimizing %d parameter combinations.", len(combos))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_params = {
                executor.submit(self._evaluate_params, params): params
                for params in combos
            }
            results = {}
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    score = future.result()
                    results[tuple(params.items())] = score  # Use tuple key for dict
                    if self.verbose:
                        self.logger.debug("Params %s scored %.4f.", params, score)
                except concurrent.futures.TimeoutError as e:
                    self.logger.error("Timeout for params %s: %s", params, e)
                except concurrent.futures.CancelledError as e:
                    self.logger.error("Task cancelled for params %s: %s", params, e)
                except (ValueError, RuntimeError, TypeError) as e:
                    self.logger.error("Error evaluating params %s: %s", params, e)

        if not results:
            raise ValueError("No valid parameter combinations evaluated.")

        # Find best (max score)
        best_params_tuple = max(results, key=lambda x: results[x])
        best_params = dict(best_params_tuple)
        best_score = results[best_params_tuple]

        self.logger.info("Best params: %s with score %.4f.", best_params, best_score)
        return best_params, best_score
