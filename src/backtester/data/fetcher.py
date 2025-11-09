from typing import Dict, List

from joblib import Memory  # type: ignore[import-untyped]
from pandas import DataFrame, Index, MultiIndex, Series, concat
from yfinance import download  # type: ignore[import-untyped]

from ..utils.helpers import DataError
from ..utils.logger import setup_logger

logger = setup_logger(__name__, file_path="data_fetcher.log")

# Caching setup
memory = Memory(location="data/cache", verbose=0)


def clear_data_cache() -> None:
    """Clears the entire data cache.

    This function removes all cached data, forcing subsequent data requests
    to be fetched from the source again.
    """
    try:
        memory.clear(warn=False)
        logger.debug("Data cache cleared successfully")
    except (OSError, ValueError, ConnectionError) as error:
        logger.error("Failed to clear cache: %s", error)


@memory.cache
def _fetch_historical_data_internal(
    symbols: List[str], start: str, end: str
) -> DataFrame:
    """Internal cached function to fetch historical stock data.

    Args:
        symbols (List[str]): A list of stock symbols to fetch.
        start (str): The start date for the data in 'YYYY-MM-DD' format.
        end (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        DataFrame: A pandas DataFrame containing the historical data.

    Raises:
        DataError: If no data is returned from yfinance or if there's an
                   error during fetching.
    """
    try:
        data = download(
            symbols,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            actions=True,
        )
        if data.empty:
            raise DataError("No data returned from yfinance")
        logger.debug("Fetched data for %s from %s to %s", symbols, start, end)
        return data
    except (ValueError, IOError, ConnectionError) as error:
        logger.error("Error fetching data: %s", error)
        raise DataError(f"Data fetch failed: {error}") from error


def fetch_historical_data(
    symbols: List[str], start: str, end: str, force_refresh: bool = False
) -> Dict[str, DataFrame]:
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
        Dict[str, DataFrame]: A dictionary where keys are stock symbols and
                                 values are pandas DataFrames containing the
                                 historical data for each symbol.

    Raises:
        DataError: If no valid data is fetched for any of the symbols or
                   if there is an unexpected error.
    """
    if force_refresh:
        clear_data_cache()
        logger.info("Force refresh: Cache cleared for this fetch")

    required_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    def extract_symbol_dataframe(raw: DataFrame, symbol: str) -> DataFrame:
        """Returns a DataFrame for a single symbol, robust to different MultiIndex layouts.

        Args:
            raw (DataFrame): The raw pandas DataFrame, which may have a MultiIndex.
            symbol (str): The stock symbol to extract.

        Returns:
            DataFrame: A pandas DataFrame for the specified symbol.

        Raises:
            KeyError: If the symbol cannot be found in the raw data.
        """
        # If raw is not multi-indexed, return a copy
        if not isinstance(raw.columns, MultiIndex):
            return raw.copy()

        # Strategy 1: try xs at level=1 then level=0
        for level in (1, 0):
            try:
                dataframe = raw.xs(symbol, axis=1, level=level)
                # If the returned object is a Series (single column), convert to DF
                if isinstance(dataframe, Series):
                    dataframe = dataframe.to_frame()
                if isinstance(dataframe, Series):
                    return dataframe.to_frame().copy()
                return dataframe.copy()
            except (KeyError, IndexError, ValueError):
                pass

        # Strategy 2: fallback - find all columns where any level equals the symbol
        columns = [column for column in raw.columns if symbol in column]
        if not columns:
            raise KeyError(
                f"Symbol {symbol} not found in fetched data (tried multiple strategies)"
            )

        # For each selected column tuple, figure out the "field name" (Open/Close/etc.)
        series_list = []
        column_names = []
        for column in columns:
            series = raw[column]
            # If column label is tuple-like, find a part that matches a required column
            field = None
            if isinstance(column, tuple):
                for part in column:
                    if part in required_columns:
                        field = part
                        break
                if field is None:
                    # choose the first non-symbol (string) part as fallback
                    field = next(
                        (str(part) for part in column if str(part) != str(symbol)),
                        str(column),
                    )
            else:
                field = column
            series_list.append(series)
            column_names.append(field)

        dataframe = concat(series_list, axis=1)
        dataframe.columns = column_names
        if isinstance(dataframe, Series):
            dataframe = dataframe.to_frame()
        return dataframe

    try:
        raw_data = _fetch_historical_data_internal(symbols, start, end)

        # Build per-symbol dictionary
        data_dict: Dict[str, DataFrame] = {}

        if len(symbols) == 1:
            single_symbol = symbols[0]
            # If raw_data is a DataFrame for the single symbol, normalize columns defensively
            try:
                dataframe = extract_symbol_dataframe(raw_data, single_symbol)
            except KeyError:
                # If extraction fails, but raw_data has flat columns that look like fields, use it
                dataframe = raw_data.copy()
            data_dict[single_symbol] = dataframe
        else:
            for symbol in symbols:
                try:
                    dataframe = extract_symbol_dataframe(raw_data, symbol)
                    data_dict[symbol] = dataframe
                except KeyError:
                    logger.warning("Symbol %s not found in fetched data", symbol)
                    continue

        # Validate and clean each DataFrame
        for symbol, dataframe in list(data_dict.items()):
            dataframe = dataframe.copy()  # avoid SettingWithCopy warnings

            # If there is some MultiIndex leftover, try to collapse to a single meaningful level
            if isinstance(dataframe.columns, MultiIndex):
                # choose level with the most overlap with required_columns
                best_lvl = None
                best_overlap = -1
                for lvl in range(dataframe.columns.nlevels):
                    vals = set(dataframe.columns.get_level_values(lvl))
                    overlap = len(vals & set(required_columns))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_lvl = lvl
                if best_overlap > 0 and best_lvl is not None:
                    dataframe.columns = dataframe.columns.get_level_values(
                        int(best_lvl)
                    )
                else:
                    # last resort: join multi-level names into single strings
                    dataframe.columns = Index(
                        "_".join(
                            [str(x) for x in column if x is not None and str(x) != ""]
                        )
                        for column in dataframe.columns
                    )

            # Provide fallback for missing 'Adj Close'
            if "Adj Close" not in dataframe.columns and "Close" in dataframe.columns:
                dataframe["Adj Close"] = dataframe["Close"].copy()
                logger.debug(
                    "'Adj Close' column missing for %s; copied from 'Close'", symbol
                )

            # Check required columns
            if not all(column in dataframe.columns for column in required_columns):
                logger.error(
                    "Missing required columns for %s: %s",
                    symbol,
                    dataframe.columns.tolist(),
                )
                raise DataError(f"Incomplete data for {symbol}")

            # Forward fill any NaNs and ensure dtypes inferred
            dataframe = dataframe.ffill().infer_objects(copy=False)
            data_dict[symbol] = dataframe
            logger.debug("Cleaned data for %s: %s rows", symbol, len(dataframe))

        if not data_dict:
            raise DataError("No valid data fetched for any symbols")

        logger.debug("Data fetched and split for %s", list(data_dict.keys()))
        return data_dict

    except DataError:
        logger.error("Data error encountered in fetch_historical_data")
        raise
    except Exception as error:
        logger.exception("Unexpected error in data fetching")
        raise DataError(f"Unexpected data error: {error}") from error
