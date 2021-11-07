"""Matrix Constructor for News
"""
from tqdm import tqdm
import pandas as pd
from .base import MatrixConstructor


class NewsMatrixConstructor(MatrixConstructor):
    """News Matrix Constructor"""

    def __init__(self, **configs):
        super().__init__(**configs)

    def get_matrix(self, df):
        quarters = df.quarter.unique()
        companies = [list(com.keys())[0] for com in self.companies]
        alias = list(map(lambda x: list(x.items())[0][1]["alias"], self.companies))
        res = {
            quarter: pd.DataFrame(0, index=companies, columns=companies)
            for quarter in quarters
        }
        for quarter in tqdm(quarters):
            quarter_df = df[df.quarter == quarter]
            for company1, search_items1 in zip(companies, alias):
                for company2, search_items2 in zip(companies, alias):
                    if company1 != company2:
                        search_items = search_items1 + search_items2
                    else:
                        search_items = search_items1
                    pat = "|".join(search_items)
                    res[quarter][company1][company2] += quarter_df.texts.str.contains(
                        pat
                    ).sum()
        for quarter, mat in res.items():
            res[quarter] = mat.values
        return res
