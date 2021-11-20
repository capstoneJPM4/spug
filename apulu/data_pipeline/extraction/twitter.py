"""Matrix Constructor for Twitter
"""
from tqdm import tqdm
import pandas as pd
from .base import MatrixConstructor


class TwitterMatrixConstructor(MatrixConstructor):
    """Twitter Matrix Constructor"""

    def __init__(self, **configs):
        super().__init__(**configs)

    def get_matrix(self, df):
        months = df.month.unique()
        companies = [list(com.keys())[0] for com in self.companies]
        alias = list(map(lambda x: list(x.items())[0][1]["alias"], self.companies))
        res = {
            month: pd.DataFrame(0, index=companies, columns=companies)
            for month in months
        }
        for month in tqdm(months):
            month_df = df[df.month == month]
            for company1, search_items1 in zip(companies, alias):
                for company2, search_items2 in zip(companies, alias):
                    if company1 != company2:
                        search_items = search_items1 + search_items2
                    else:
                        search_items = search_items1
                    pat = "|".join(search_items)
                    res[month][company1][company2] += month_df.text.str.contains(
                        pat
                    ).sum()
        for month, mat in res.items():
            res[month] = mat.values
        return res
