import yfinance as yf
import pandas as pd
from typing import List, Dict, Optional
from joblib import Memory
from ..utils.logger import setup_logger
from ..utils.helpers import DataError

logger = setup_logger(__name__)

# Caching setup
memory = Memory(location='data/cache', verbose=0)

def clear_data_cache() -> None:
    """Clears the entire data cache.

    This function removes all cached data, forcing subsequent data requests
    to be fetched from the source again.
    """
    try:
        memory.clear(warn=False)
        logger.info("Data cache cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

@memory.cache
def _fetch_historical_data_internal(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    """Internal cached function to fetch historical stock data.

    Args:
        symbols (List[str]): A list of stock symbols to fetch.
        start (str): The start date for the data in 'YYYY-MM-DD' format.
        end (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the historical data.

    Raises:
        DataError: If no data is returned from yfinance or if there's an
                   error during fetching.
    """
    try:
        data = yf.download(symbols, start=start, end=end, progress=False, auto_adjust=False, actions=True)
        if data.empty:
            raise DataError("No data returned from yfinance")
        logger.info(f"Fetched data for {symbols} from {start} to {end}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        raise DataError(f"Data fetch failed: {e}")

def fetch_historical_data(symbols: List[str], start: str, end: str, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetches and splits historical stock data, with caching.

    This function retrieves historical stock data for a list of symbols between
    the specified start and end dates. It uses caching to speed up subsequent
    requests for the same data. The function is designed to be robust against
    different column layouts returned by the yfinance library.

    Args:
        symbols (List[str]): A list of stock symbols to fetch.
        start (str): The start date for the data in 'YYYY-MM-DD' format.
        end (str): The end date for the data in 'YYYY-MM-DD' format.
        force_refresh (bool): If True, the cache will be cleared before fetching data.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are stock symbols and
                                 values are pandas DataFrames containing the
                                 historical data for each symbol.

    Raises:
        DataError: If no valid data is fetched for any of the symbols or
                   if there is an unexpected error.
    """
    if force_refresh:
        clear_data_cache()
        logger.info("Force refresh: Cache cleared for this fetch")

    required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    def extract_symbol_df(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Returns a DataFrame for a single symbol, robust to different MultiIndex layouts.

        Args:
            raw (pd.DataFrame): The raw pandas DataFrame, which may have a MultiIndex.
            symbol (str): The stock symbol to extract.

        Returns:
            pd.DataFrame: A pandas DataFrame for the specified symbol.

        Raises:
            KeyError: If the symbol cannot be found in the raw data.
        """
        # If raw is not multi-indexed, return a copy
        if not isinstance(raw.columns, pd.MultiIndex):
            return raw.copy()

        # Strategy 1: try xs at level=1 then level=0
        for level in (1, 0):
            try:
                df = raw.xs(symbol, axis=1, level=level)
                # If the returned object is a Series (single column), convert to DF
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                return df.copy()
            except Exception:
                pass

        # Strategy 2: fallback - find all columns where any level equals the symbol
        cols = [col for col in raw.columns if symbol in col]
        if not cols:
            raise KeyError(f"Symbol {symbol} not found in fetched data (tried multiple strategies)")

        # For each selected column tuple, figure out the "field name" (Open/Close/etc.)
        series_list = []
        col_names = []
        for col in cols:
            series = raw[col]
            # If column label is tuple-like, find a part that matches a required column
            field = None
            if isinstance(col, tuple):
                for part in col:
                    if part in required_columns:
                        field = part
                        break
                if field is None:
                    # choose the first non-symbol (string) part as fallback
                    field = next((str(part) for part in col if str(part) != str(symbol)), str(col))
            else:
                field = col
            series_list.append(series)
            col_names.append(field)

        df = pd.concat(series_list, axis=1)
        df.columns = col_names
        return df

    try:
        raw_data = _fetch_historical_data_internal(symbols, start, end)

        # Build per-symbol dictionary
        data_dict: Dict[str, pd.DataFrame] = {}

        if len(symbols) == 1:
            sym = symbols[0]
            # If raw_data is a DataFrame for the single symbol, normalise columns defensively
            try:
                df = extract_symbol_df(raw_data, sym)
            except KeyError:
                # If extraction fails, but raw_data has flat columns that look like fields, use it
                df = raw_data.copy()
            data_dict[sym] = df
        else:
            for symbol in symbols:
                try:
                    df = extract_symbol_df(raw_data, symbol)
                    data_dict[symbol] = df
                except KeyError:
                    logger.warning(f"Symbol {symbol} not found in fetched data")
                    continue

        # Validate and clean each DataFrame
        for symbol, df in list(data_dict.items()):
            df = df.copy()  # avoid SettingWithCopy warnings

            # If there is some MultiIndex leftover, try to collapse to a single meaningful level
            if isinstance(df.columns, pd.MultiIndex):
                # choose level with the most overlap with required_columns
                best_lvl = None
                best_overlap = -1
                for lvl in range(df.columns.nlevels):
                    vals = set(df.columns.get_level_values(lvl))
                    overlap = len(vals & set(required_columns))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_lvl = lvl
                if best_overlap > 0:
                    df.columns = df.columns.get_level_values(best_lvl)
                else:
                    # last resort: join multi-level names into single strings
                    df.columns = ['_'.join([str(x) for x in col if x is not None and str(x) != '']) for col in df.columns]

            # Provide fallback for missing 'Adj Close'
            if 'Adj Close' not in df.columns and 'Close' in df.columns:
                df['Adj Close'] = df['Close'].copy()
                logger.debug(f"'Adj Close' column missing for {symbol}; copied from 'Close'")

            # Check required columns
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns for {symbol}: {df.columns.tolist()}")
                raise DataError(f"Incomplete data for {symbol}")

            # Forward fill any NaNs and ensure dtypes inferred
            df = df.ffill().infer_objects(copy=False)
            data_dict[symbol] = df
            logger.debug(f"Cleaned data for {symbol}: {len(df)} rows")

        if not data_dict:
            raise DataError("No valid data fetched for any symbols")

        logger.debug(f"Data fetched and split for {list(data_dict.keys())}")
        return data_dict

    except DataError:
        logger.error("Data error encountered in fetch_historical_data")
        raise
    except Exception as e:
        logger.exception("Unexpected error in data fetching")
        raise DataError(f"Unexpected data error: {e}")
