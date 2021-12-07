"""Matrix Constructor for Twitter
"""
from tqdm import tqdm
import pandas as pd
from .base import MatrixConstructor


class TwitterMatrixConstructor(MatrixConstructor):
    """Twitter Matrix Constructor"""

    def __init__(self, **configs):
        super().__init__(**configs)

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
        alias = list(map(lambda x: list(x.items())[0][1]["alias"], self.companies))
        res = {
            interval: pd.DataFrame(0, index=companies, columns=companies)
            for interval in time_intervals
        }
        for T in tqdm(time_intervals):
            sub_df = df[df.interval == T]
            for company1, search_items1 in zip(companies, alias):
                for company2, search_items2 in zip(companies, alias):
                    if company1 != company2:
                        search_items = search_items1 + search_items2
                    else:
                        search_items = search_items1
                    pat = "|".join(search_items)
                    res[T][company1][company2] += sub_df.text.str.contains(pat).sum()
        to_return = {}
        for T, mat in res.items():
            to_return[T] = mat.values
        return to_return
