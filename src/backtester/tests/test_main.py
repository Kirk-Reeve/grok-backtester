import pandas as pd
import pytest

from backtester.main import main


def test_main(mocker):
    """Test main with mocks."""
    mocker.patch("backtester.main.load_config")
    mocker.patch(
        "backtester.main.fetch_historical_data",
        return_value={"AAPL": pd.DataFrame({"Adj Close": [100, 101]})},
    )
    mocker.patch(
        "backtester.main.run_parallel_backtests",
        return_value=[{"portfolio": pd.DataFrame(), "metrics": {}}],
    )
    mocker.patch("backtester.main.generate_backtest_report")
    with pytest.raises(SystemExit):  # If main raises, but for success
        main()
