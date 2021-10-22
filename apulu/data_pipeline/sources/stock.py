"""
fetch historical stocks prices
"""
from tqdm.auto import tqdm
import pandas as pd
import pandas_datareader as pdr


def get_stock_price(symbol, start, end):
    """get stock price of a company over a time range
    Args:
        symbol (str): ticker symbol of a stock
        start (datetime.datetime): start time
        end (datetime.datetime): end time
    Returns:
        pd.DataFrame: stock price of a company over a time range
    """
    return (
        pdr.yahoo.daily.YahooDailyReader(symbol, start=start, end=end)
        .read()
        .reset_index()[["Date", "High", "Low", "Open", "Close", "Volume", "Adj Close"]]
    )


def get_stock_prices(symbols, start, end):
    """get stock prices of companies over a time range
    Args:
        symbol (list): ticker symbols of stocks
        start (datetime.datetime): start time
        end (datetime.datetime): end time
    Returns:
        pd.DataFrame: stock prices of companies over a time range
    """
    dfs = pd.DataFrame()
    for symbol in tqdm(symbols):
        df = get_stock_price(symbol, start, end)
        df["ticker_symbol"] = symbol
        dfs = dfs.append(df)
    return dfs.reset_index(drop=True)
