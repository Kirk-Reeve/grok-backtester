# Advanced Python Share Trading Strategy Backtester

This project is a powerful and flexible backtesting engine for share trading strategies, written in Python. It allows you to test your trading ideas on historical data and analyze their performance with a variety of metrics.

## Features

- **Vectorized Backtesting:** The backtesting engine is fully vectorized, making it fast and efficient.
- **Multiple Strategies:** The backtester supports multiple trading strategies, and it's easy to add your own.
- **Parallel Processing:** Backtests for multiple symbols can be run in parallel to save time.
- **Caching:** Historical data is cached locally to speed up subsequent backtests.
- **Comprehensive Metrics:** The backtester calculates a wide range of performance metrics, including Sharpe ratio, Sortino ratio, max drawdown, and more.
- **Detailed Reports:** The backtester can generate detailed reports with equity curves, drawdown plots, and trade signals.
- **Configuration Driven:** The backtester is configured using a simple YAML file, making it easy to change settings and run different backtests.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Kirk-Reeve/grok-backtester.git
    cd grok-backtester/backtester
    ```

2.  **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Configure your backtest:**

    Open the `config/config.yaml` file and customize the settings for your backtest. You can specify the data source, symbols, date range, trading strategy, and backtest parameters.

2.  **Run the backtester:**

    ```bash
    python -m src.backtester.main --config config/config.yaml
    ```

3.  **View the results:**

    The backtester will generate a report in the `reports` directory, containing plots of the equity curve, drawdown, and trade signals. The performance metrics will be printed to the console.

## Adding a New Strategy

To add a new trading strategy, you need to create a new Python file in the `src/backtester/strategies` directory and define a new class that inherits from `BaseStrategy`. You will also need to implement the `generate_signals` method, which should return a pandas Series of trading signals.

Finally, you need to register your new strategy in the `src/backtester/strategies/__init__.py` file.

## Disclaimer

This backtester is for educational and research purposes only. It is not intended to be used for live trading.
