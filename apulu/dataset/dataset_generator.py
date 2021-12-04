import os
from glob import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from sklearn.preprocessing import normalize


class DatasetGenerator:
    _implimented_freqs = ["month", "quarter"]

    def __init__(self, data_list: str, stock_path: str, sec_path: str, freq: str):
        self.data_list = data_list
        self.stock_df = pd.read_csv(
            stock_path,
            usecols=["ticker_symbol", "date", "Close"],
            parse_dates=["date"],
        )
        self.sec_path = sec_path
        if freq not in self._implimented_freqs:
            raise NotImplementedError
        self.freq = freq

    def process(self):
        X_y = []
        edge_idx = []
        edge_att = []

        for data in self.data_list[:-1]:
            year, time = data.split("/")[-1].split(".")[0].split("_")
            X_tensor, y_tensor = self._process_stock(year, time)
            X_y.append((X_tensor, y_tensor))

        assert len(X_y) == len(self.data_list) - 1

        for data in self.data_list[:-1]:
            edge_index, edge_attr = dense_to_sparse(torch.from_numpy(np.load(data)))
            edge_idx.append(edge_index.numpy())
            edge_att.append(edge_attr.numpy())

        features = []
        targets = []
        for X, y in X_y:
            features.append(normalize(X, axis=1, norm="max"))
            targets.append(y)

        n_features = max(features, key=lambda x: x.shape[1]).shape[1]
        padded_features = []
        for i in features:
            padded_features.append(
                np.pad(i, [(0, 0), (0, n_features - i.shape[1])], "mean")
            )
        comp_emb = []
        for fp in sorted(glob(os.path.join(self.sec_path, "*.npy"))):
            comp_emb.append(np.load(fp))
        comp_emb = np.stack(comp_emb)
        comp_emb = np.asarray([comp_emb for _ in range(len(padded_features))])
        padded_features = np.concatenate((padded_features, comp_emb), axis=2)

        return DynamicGraphTemporalSignal(
            edge_indices=edge_idx,
            edge_weights=edge_att,
            features=padded_features,
            targets=targets,
        )

    def _process_stock(self, year, time):
        if self.freq == "month":
            year, time = int(year), int(time)
            if time < 12:
                next_year, next_time = year, time + 1
            else:
                next_year, next_time = year + 1, 1
            curr = self.stock_df[
                (self.stock_df.date.dt.year == year)
                & (self.stock_df.date.dt.month == time)
            ]
            nxt = self.stock_df[
                (self.stock_df.date.dt.year == next_year)
                & (self.stock_df.date.dt.month == next_time)
            ]
        elif self.freq == "quarter":
            year, time = int(year), int(time[1])
            if time < 4:
                next_year, next_time = year, time + 1
            else:
                next_year, next_time = year + 1, 1
            curr = self.stock_df[
                (self.stock_df.date.dt.year == year)
                & (self.stock_df.date.dt.quarter == time)
            ]
            nxt = self.stock_df[
                (self.stock_df.date.dt.year == next_year)
                & (self.stock_df.date.dt.quarter == next_time)
            ]
        X = curr.pivot_table(
            index="date", columns="ticker_symbol", values="Close"
        ).values.T
        x_logret = np.diff(np.log(X))
        col_zeros = np.zeros((X.shape[0], 1))
        x_normalized = np.append(col_zeros, x_logret, 1)

        if len(nxt) == 0:
            return None
        y = nxt.pivot_table(
            index="date", columns="ticker_symbol", values="Close"
        ).values.T

        y = (y.mean(1) - X.mean(1)) / X.mean(1)

        return x_normalized, y
