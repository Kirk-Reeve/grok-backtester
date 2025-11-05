# first line: 21
@memory.cache
def _fetch_historical_data_internal(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Internal cached function to fetch historical stock data."""
    try:
        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False, actions=True)
        if data.empty:
            raise DataError("No data returned from yfinance")
        logger.info(f"Fetched data for {symbols} from {start} to {end}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise DataError(f"Data fetch failed: {e}")
