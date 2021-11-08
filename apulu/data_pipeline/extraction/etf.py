import os
from glob import glob
import pandas as pd
import numpy as np
from .base import MatrixConstructor


class EtfMatrixConstructor(MatrixConstructor):
    """etf matrix preprocessor"""

    implemented_options = ["bipartite", "occurence"]

    def __init__(self, option, **configs):
        """ETF Matrix Constructor

        Args:
            option (dict): config parameter dictionary in yaml
        """
        super().__init__(**configs)
        self.option = option

    def get_matrix(self, df=None):
        """main method to construct matrix
        options:
            - bipartite:
                etf x companies matrix
            - occurence:
                companies x companies matrix
        support files in {sss}-{yyyy}-q{x}.csv format sss means ticker symbol, yyyy means for years in four digits and x means for quarters from 1-4.
        MUST specify etf_config['external_path'] in your configuration yaml file.
        Args:
            df (None, optional): None value. Defaults to None.

        Returns:
            np.array: adjacency matrix in numpy format.
        """
        files = glob(os.path.join(self.etf_config["external_path"], "*.csv"))
        num_quarters = len(set(map(lambda x: x.split("/")[-1][-11:-4], files)))
        companies = [list(com.keys())[0] for com in self.companies]
        if self.option == "bipartite":
            mat = self._generate_bipartite(files, companies, num_quarters)
        elif self.option == "occurence":
            mat = self._generate_bipartite(files, companies, num_quarters)
            mat = mat @ mat.T
        else:
            raise NotImplementedError(
                f"please specify option in {self.implemented_options}"
            )
        return {"****_**": mat}

    def _generate_bipartite(self, files, companies, num_quarters):
        """generate stock ticker - etf bipartite graph
        Args:
            files (list): list of file directory of raw datum
            symbols (list): list of stock ticker symbols
            num_quarters (int): number of quarters convered by the files
        Returns:
            np.array: 2d np array m by n, where m represents number of stocks
                n represents number of etfs
        """
        etfs = self._get_etf_set(files)
        etfs = {etf: i for i, etf in enumerate(etfs)}
        symbols = {symbol: i for i, symbol in enumerate(companies)}
        matrix = np.zeros((len(symbols), len(etfs)))
        for file in files:
            symbol, _, _ = file.split("/")[-1].split(".")[0].split("-")
            data = pd.read_csv(file)
            etf_lst = data.Filer.str.lower().tolist()
            etf_share_holding = data["Shares Held"]
            i = symbols.get(symbol, None)
            if i is not None:
                for etf, share_holding in zip(etf_lst, etf_share_holding):
                    j = etfs[etf]
                    matrix[i, j] += share_holding
        return matrix / num_quarters

    def _get_etf_set(self, files):
        """helper function to extract etf set from raw file
        Args:
            files (list): list of file directory of raw datum
        Returns:
            list: list of unique etfs addressed in the files
        """
        etfs = []
        for file in files:
            etf_lst = pd.read_csv(file).Filer.str.lower().tolist()
            etfs.extend(etf_lst)
        etfs = set(etfs)
        return sorted(list(etfs))
