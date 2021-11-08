from typing import Optional, Callable, List

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import dense_to_sparse


class StockGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        path: str,
        stock_path: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.name = name.lower()
        if not os.path.exists(os.path.join(root)):
            raise FileNotFoundError
        self.data = os.path.join(root, "raw", self.name, path)
        self.stock_path = os.path.join(root, "raw", stock_path)
        self.path = path.split(".")[0]
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        if not os.path.exists(os.path.join(self.root, "processed")):
            os.mkdir(os.path.join(self.root, "processed"))
        if not os.path.exists(os.path.join(self.root, "processed", self.name)):
            os.mkdir(os.path.join(self.root, "processed", self.name))
        return [f"{os.path.join(self.name,self.path)}.pt"]

    # @property
    # def download(self):
    #     return

    def process(self):
        X, y = self._process_stock()
        edges = np.load(os.path.join(self.data))
        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(edges))
        data = Data(x=X, y=y, edge_index=edge_index, edge_attr=edge_attr)
        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    def _process_stock(self):
        year, quarter = self.path.split("_")[-2:]
        year, quarter = int(year), int(quarter[1])
        df = pd.read_csv(
            self.stock_path,
            usecols=["ticker_symbol", "Date", "Close"],
            parse_dates=["Date"],
        )
        curr = df[(df.Date.dt.year == year) & (df.Date.dt.quarter == quarter)]
        X = curr.pivot_table(
            index="Date", columns="ticker_symbol", values="Close"
        ).values.T
        if quarter < 4:
            next_year, next_quarter = year, quarter + 1
        else:
            next_year, next_quarter = year + 1, 1
        nxt = df[(df.Date.dt.year == next_year) & (df.Date.dt.quarter == next_quarter)]
        y = nxt.pivot_table(
            index="Date", columns="ticker_symbol", values="Close"
        ).values.T
        y = (y.mean(1) - X.mean(1)) / X.mean(1)
        return X, y
