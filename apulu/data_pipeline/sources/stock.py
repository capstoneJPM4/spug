"""
fetch historical stocks prices
"""
from tqdm import tqdm
import pandas as pd
import pandas_datareader as pdr
from .base import DataFetcher


def get_stock_price(symbol, start, end):
    """get stock price of a company over a time range
    Args:
        symbol (str): ticker symbol of a stock
        start (datetime.datetime): start time
        end (datetime.datetime): end time
    Returns:
        pd.DataFrame: stock price of a company over a time range
    """
    df = (
        pdr.yahoo.daily.YahooDailyReader(symbol, start=start, end=end)
        .read()
        .reset_index()[["Date", "High", "Low", "Open", "Close", "Volume", "Adj Close"]]
    )
    df["datetime"] = pd.to_datetime(df.Date)
    df = df.assign(
        date=df.datetime.dt.date,
        month=df.datetime.dt.month,
        year=df.datetime.dt.year,
        quarter=df.datetime.apply(lambda x: f"{x.year}_q{x.quarter}"),
    )
    return df.drop("datetime", axis=1)


class StockFetcher(DataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self):
        """get stock prices of companies over a time range
        Args:
            symbol (list): ticker symbols of stocks
            start (datetime.datetime): start time
            end (datetime.datetime): end time
        Returns:
            pd.DataFrame: stock prices of companies over a time range
        """
        dfs = pd.DataFrame()
        symbols = self.companies
        symbols = list(map(lambda x: list(x.keys())[0], symbols))
        for symbol in tqdm(symbols):
            df = get_stock_price(symbol, self.start_date, self.end_date)
            df["ticker_symbol"] = symbol
            dfs = dfs.append(df)
        return dfs.reset_index(drop=True)
