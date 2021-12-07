"""Matrix Constructor for Stock
"""
from tqdm import tqdm
import pandas as pd
import numpy as np
from .base import MatrixConstructor


class StockMatrixConstructor(MatrixConstructor):
    """Construct Correlation Matrix for Stock Data"""

    implemented_options = ["price", "volume"]

    def __init__(self, option, **configs):
        super().__init__(**configs)
        self.option = option

    def get_matrix(self, df, interval):
        if interval == "month":
            df = df.assign(
                interval=df.date.apply(lambda x: f"{x.year}_{str(x.month).zfill(2)}")
            )
        elif interval == "quarter":
            df = df.assign(interval=df.date.apply(lambda x: f"{x.year}_q{x.quarter}"))
        else:
            raise NotImplementedError

        time_intervals = sorted(df.interval.unique())
        companies = [list(com.keys())[0] for com in self.companies]
        res = {
            interval: pd.DataFrame(0, index=companies, columns=companies)
            for interval in time_intervals
        }
        if self.option == "price":
            agg_val = "Close"
        elif self.option == "volume":
            agg_val = "Volume"
        else:
            raise NotImplementedError(
                f"please specify option in {self.implemented_options}"
            )
        for T in tqdm(time_intervals):
            quarter_df = df[df.interval == T]
            res[T] = quarter_df.pivot_table(
                index="date", columns="ticker_symbol", values=agg_val
            ).corr()
        to_return = {}
        for T, mat in res.items():
            to_return[T] = np.square(mat.values)
        return to_return
