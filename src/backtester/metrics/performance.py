"""Performance metrics calculation for trading strategies."""

from typing import Dict, Optional

from numpy import cov, inf, nan, sqrt
from pandas import DataFrame, DatetimeIndex, Series

from ..utils.logger import setup_logger


class PerformanceCalculator:
    """Calculator for key performance metrics of trading strategies.

    This class computes various metrics from a portfolio DataFrame to evaluate
    strategy effectiveness. It supports configurable parameters like risk-free
    rate and benchmark for advanced analysis.

    Args:
        risk_free_rate (float): Annual risk-free rate (default: 0.0).
        trading_days_per_year (int): Number of trading days per year (default: 252 for stocks).
        benchmark_returns (Optional[Series]): Benchmark returns for beta/alpha
            (aligned index; default: None).

    Attributes:
        risk_free_rate (float): Stored risk-free rate.
        trading_days_per_year (int): Stored trading days per year.
        benchmark_returns (Optional[Series]): Stored benchmark returns.
        logger (logging.Logger): Class-specific logger.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        trading_days_per_year: int = 252,
        benchmark_returns: Optional[Series] = None,
    ):
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.benchmark_returns = benchmark_returns
        self.logger = setup_logger(__name__, file_path="performance_metrics.log")
        self.logger.debug(
            "Initialized PerformanceCalculator with risk_free_rate=%.4f, "
            "trading_days_per_year=%d, has_benchmark=%s",
            self.risk_free_rate,
            self.trading_days_per_year,
            bool(self.benchmark_returns),
        )

    def calculate_metrics(self, portfolio: DataFrame) -> Dict[str, float]:
        """Calculate key performance metrics for a trading strategy.

        Args:
            portfolio (DataFrame): DataFrame with DatetimeIndex, containing 'returns'
                (periodic returns) and 'total' (cumulative portfolio value) columns.

        Returns:
            Dict[str, float]: Dictionary of calculated metrics.

        Raises:
            ValueError: If portfolio is missing required columns or insufficient data.
        """
        if "returns" not in portfolio.columns or "total" not in portfolio.columns:
            raise ValueError("Portfolio missing required 'returns' or 'total' columns.")

        if len(portfolio) < 2:
            self.logger.warning("Insufficient data for metrics; returning defaults.")
            return self._default_metrics()

        returns = portfolio["returns"].dropna()
        total = portfolio["total"]

        self.logger.debug("Calculating metrics for %d returns.", len(returns))

        # Total return: (final / initial) - 1
        total_return = (
            (total.iloc[-1] / total.iloc[0]) - 1 if total.iloc[0] != 0 else 0.0
        )

        # CAGR: (final / initial) ^ (trading_days_per_year / num_days) - 1
        num_days = (
            (portfolio.index[-1] - portfolio.index[0]).days
            if isinstance(portfolio.index, DatetimeIndex)
            else len(returns)
        )
        cagr = (
            (total.iloc[-1] / total.iloc[0]) ** (self.trading_days_per_year / num_days)
            - 1
            if total.iloc[0] != 0 and num_days > 0
            else 0.0
        )

        # Annualized mean return and volatility
        ann_mean_return = returns.mean() * self.trading_days_per_year
        ann_std_dev = returns.std() * sqrt(self.trading_days_per_year)

        # Sharpe: (ann_mean_return - risk_free) / ann_std_dev
        sharpe_ratio = (
            (ann_mean_return - self.risk_free_rate) / ann_std_dev
            if ann_std_dev != 0
            else 0.0
        )

        # Sortino: Uses downside deviation
        downside_returns = returns[returns < 0]
        ann_downside_std = (
            downside_returns.std() * sqrt(self.trading_days_per_year)
            if not downside_returns.empty
            else 0.0
        )
        sortino_ratio = (
            (ann_mean_return - self.risk_free_rate) / ann_downside_std
            if ann_downside_std != 0
            else 0.0
        )

        # Max drawdown: Min of (total - cummax(total)) / cummax(total)
        peak = total.cummax()
        drawdown = (total - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        # Calmar: CAGR / |max_drawdown|
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Trade metrics
        trades = returns[returns != 0]
        num_trades = len(trades)
        if num_trades == 0:
            self.logger.warning("No trades; trade metrics set to defaults.")
            win_rate = 0.0
            avg_return_per_trade = 0.0
            profit_factor = nan
        else:
            wins = trades > 0
            win_rate = wins.sum() / num_trades
            avg_return_per_trade = trades.mean()
            gross_profit = trades[trades > 0].sum()
            gross_loss = abs(trades[trades < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else inf

        # Additional metrics
        annualized_volatility = ann_std_dev

        # Beta/Alpha if benchmark provided
        beta = alpha = nan
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            # Align benchmark with returns
            aligned_bench = self.benchmark_returns.reindex(returns.index).fillna(0)
            cov_ = cov(returns, aligned_bench)[0, 1]
            bench_var = aligned_bench.var()
            beta = cov_ / bench_var if bench_var != 0 else nan
            ann_bench_mean = aligned_bench.mean() * self.trading_days_per_year
            alpha = ann_mean_return - (
                self.risk_free_rate + beta * (ann_bench_mean - self.risk_free_rate)
            )

        metrics = {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "annualized_volatility": annualized_volatility,
            "beta": beta,
            "alpha": alpha,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_return_per_trade": avg_return_per_trade,
            "profit_factor": profit_factor,
        }

        self.logger.info("Metrics calculated: %s", metrics)
        return metrics

    def _default_metrics(self) -> Dict[str, float]:
        """Return default values for all performance metrics.

        Used when insufficient data is provided.

        Returns:
            Dict[str, float]: Dictionary of metrics initialized to zero or NaN.
        """
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "annualized_volatility": 0.0,
            "beta": nan,
            "alpha": nan,
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_return_per_trade": 0.0,
            "profit_factor": nan,
        }
