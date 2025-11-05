import pytest
import pandas as pd
import numpy as np
from backtester.data.fetcher import fetch_historical_data, clear_data_cache
from backtester.utils.helpers import DataError

def test_fetch_historical_data(mocker):
    """Test fetching historical data with mocking."""
    mock_download = mocker.patch('backtester.data.fetcher.yf.download')
    mock_data = pd.DataFrame({
        'Open': [99, 100],
        'High': [101, 102],
        'Low': [98, 99],
        'Close': [100, 101],
        'Adj Close': [100, 101],
        'Volume': [1000, 1100],
    }, index=pd.date_range('2020-01-02', periods=2))
    mock_download.return_value = mock_data

    data_dict = fetch_historical_data(['AAPL'], '2020-01-01', '2020-01-03')

    assert 'AAPL' in data_dict
    assert isinstance(data_dict['AAPL'], pd.DataFrame)
    assert set(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']) <= set(data_dict['AAPL'].columns)
    mock_download.assert_called_once()

def test_fetch_historical_data_multi(mocker):
    """Test multi-symbol fetching."""
    mock_download = mocker.patch('backtester.data.fetcher.yf.download')
    columns = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits'], ['AAPL', 'MSFT']])
    mock_data = pd.DataFrame(np.random.rand(2, 16), index=pd.date_range('2020-01-02', periods=2), columns=columns)
    mock_download.return_value = mock_data

    data_dict = fetch_historical_data(['AAPL', 'MSFT'], '2020-01-01', '2020-01-03')

    assert set(data_dict.keys()) == {'AAPL', 'MSFT'}
    for df in data_dict.values():
        assert set(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']) <= set(df.columns)

def test_fetch_historical_data_empty(mocker):
    """Test empty data raises error."""
    mock_download = mocker.patch('backtester.data.fetcher.yf.download')
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(DataError, match="No data returned"):
        fetch_historical_data(['INVALID'], '2020-01-01', '2020-01-03')

def test_fetch_historical_data_force_refresh(mocker):
    """Test force refresh clears cache."""
    mock_clear = mocker.patch('backtester.data.fetcher.clear_data_cache')
    mock_download = mocker.patch('backtester.data.fetcher.yf.download')
    mock_data = pd.DataFrame({
        'Open': [99],
        'High': [101],
        'Low': [98],
        'Close': [100],
        'Adj Close': [100],
        'Volume': [1000],
        'Dividends': [0],
        'Stock Splits': [0]
    }, index=pd.date_range('2020-01-02', periods=1))
    mock_download.return_value = mock_data

    data_dict = fetch_historical_data(['AAPL'], '2020-01-01', '2020-01-03', force_refresh=True)
    assert 'AAPL' in data_dict
    mock_clear.assert_called_once()

def test_clear_data_cache(mocker):
    """Test cache clearing."""
    mock_memory = mocker.patch('backtester.data.fetcher.memory.clear')
    clear_data_cache()
    mock_memory.assert_called_once()