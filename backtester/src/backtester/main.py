import yaml
import argparse  # Standard library for CLI
from typing import List
import pandas as pd
from pydantic import ValidationError
from .data.fetcher import fetch_historical_data
from .engine.backtest import run_parallel_backtests
from .visualization.plots import generate_backtest_report
from .utils.logger import setup_logger
from .utils.helpers import AppConfig, BacktestError
from .strategies import STRATEGY_REGISTRY  # For dynamic strategy loading

logger = setup_logger(__name__, file_path='backtest.log')

def load_config(config_path: str) -> AppConfig:
    """Load and validate YAML config.

    Args:
        config_path: Path to config file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ValueError: If loading or validation fails.
    """
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Validate with Pydantic
        config = AppConfig(**raw_config)
        logger.info("Configuration loaded and validated successfully")
        return config
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {config_path}")
        raise ValueError(f"Config file not found: {e}")
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config: {e}")
        raise ValueError(f"Invalid YAML: {e}")
    except ValidationError as e:
        logger.error(f"Config validation failed: {e}")
        raise ValueError(f"Config validation failed: {e}")

def main() -> None:
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(description="Advanced Python Share Trading Strategy Backtester")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config YAML')
    parser.add_argument('--save-plots', action='store_true', default=True, help='Save plots (default: True)')
    parser.add_argument('--no-save-plots', dest='save_plots', action='store_false', help='Do not save plots')
    parser.add_argument('--display-plots', action='store_true', default=False, help='Display plots (default: False)')
    args = parser.parse_args()

    logger.info(f"CLI args: config={args.config}, save_plots={args.save_plots}, display_plots={args.display_plots}")

    try:
        config = load_config(args.config)
        data_config = config.data
        strategy_config = config.strategy
        backtest_config = config.backtest

        data_dict = fetch_historical_data(data_config.symbols, data_config.start_date, data_config.end_date)

        available_symbols = [sym for sym in data_config.symbols if sym in data_dict]
        if not available_symbols:
            raise BacktestError("No available symbols after fetching")

        datas: List[pd.DataFrame] = [data_dict[sym] for sym in available_symbols]

        strategy_class = STRATEGY_REGISTRY.get(strategy_config.type)
        if not strategy_class:
            raise BacktestError(f"Strategy type '{strategy_config.type}' not found")
        strategies = [strategy_class(strategy_config.params) for _ in datas]

        results = run_parallel_backtests(datas, strategy_config.dict(), backtest_config.dict())

        generate_backtest_report(
            results, available_symbols, {sym: data_dict[sym] for sym in available_symbols}, strategies,
            output_dir='../reports', save_plots=args.save_plots, display_plots=args.display_plots
        )

        logger.info("Backtest completed successfully")
    except BacktestError as e:
        logger.error(f"Backtest failed: {e}")
        raise  # Re-raise for script exit
    except Exception as e:
        logger.error(f"Unexpected failure: {e}")
        raise BacktestError(f"Unexpected error: {e}")

if __name__ == '__main__':
    main()