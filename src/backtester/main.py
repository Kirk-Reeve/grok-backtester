"""Main module for the Python share trading strategy backtester."""

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from pydantic import ValidationError

from .data.fetcher import fetch_historical_data
from .engine.backtest import run_parallel_backtests
from .strategies import STRATEGY_REGISTRY
from .utils.helpers import AppConfig, BacktestError, open_no_symlink
from .utils.logger import setup_logger
from .visualization.plots import generate_backtest_report

logger = setup_logger(__name__, file_path="main.log")


def load_config(config_path: str) -> AppConfig:
    """Loads and validates the application configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        AppConfig: A validated application configuration object.

    Raises:
        ValueError: If the configuration file is not found, is invalid YAML,
                    or fails validation.
    """
    try:
        with open_no_symlink(Path(config_path)) as file:
            raw_config = yaml.safe_load(file)

        config = AppConfig(**raw_config)
        logger.info("Configuration loaded and validated successfully")
        return config
    except FileNotFoundError as error:
        logger.error("Config file not found: %s", config_path)
        raise ValueError(f"Config file not found: {error}") from error
    except yaml.YAMLError as error:
        logger.error("Invalid YAML in config: %s", error)
        raise ValueError(f"Invalid YAML: {error}") from error
    except ValidationError as error:
        logger.error("Config validation failed: %s", error)
        raise ValueError(f"Config validation failed: {error}") from error


def main() -> None:
    """The main entry point for the backtester application.

    This function handles command-line argument parsing, loads the
    configuration, fetches historical data, runs the backtest, and
    generates a report.
    """
    parser = argparse.ArgumentParser(
        description="Advanced Python Share Trading Strategy Backtester"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        default=True,
        help="Save plots (default: True)",
    )
    parser.add_argument(
        "--no-save-plots",
        dest="save_plots",
        action="store_false",
        help="Do not save plots",
    )
    parser.add_argument(
        "--display-plots",
        action="store_true",
        default=False,
        help="Display plots (default: False)",
    )
    args = parser.parse_args()

    logger.debug(
        "CLI args: config=%s, save_plots=%s, display_plots=%s",
        args.config,
        args.save_plots,
        args.display_plots,
    )

    try:
        config = load_config(args.config)
        data_config = config.data
        strategy_config = config.strategy
        backtest_config = config.backtest

        historical_data = fetch_historical_data(
            data_config.symbols, data_config.start_date, data_config.end_date
        )

        available_symbols = [
            symbol for symbol in data_config.symbols if symbol in historical_data
        ]
        if not available_symbols:
            raise BacktestError("No available symbols after fetching")

        datas: List[pd.DataFrame] = [
            historical_data[symbol] for symbol in available_symbols
        ]

        strategy_class = STRATEGY_REGISTRY.get(strategy_config.type)
        if not strategy_class:
            raise BacktestError(f"Strategy type '{strategy_config.type}' not found")
        strategies = [strategy_class(strategy_config.params) for _ in datas]

        results = run_parallel_backtests(
            datas, strategy_config.model_dump(), backtest_config.model_dump()
        )

        generate_backtest_report(
            results,
            available_symbols,
            {symbol: historical_data[symbol] for symbol in available_symbols},
            strategies,
            save_plots=args.save_plots,
            display_plots=args.display_plots,
        )

        logger.info("Backtest completed successfully")
    except BacktestError as error:
        logger.error("Backtest failed: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected failure: %s", error)
        raise BacktestError(f"Unexpected error: {error}") from error


if __name__ == "__main__":
    main()
