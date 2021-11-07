"""Pipeline for every etl process in this project
"""
import os
import yaml
import json
import pandas as pd
import numpy as np

from .sources import TwitterFetcher, NewsFetcher, SecFetcher, StockFetcher
from .extraction import (
    TwitterMatrixConstructor,
    NewsMatrixConstructor,
    SecMatrixConstructor,
    StockMatrixConstructor,
    EtfMatrixConstructor,
)

IMPLEMENTED_FETCHER = {
    "twitter": TwitterFetcher,
    "news": NewsFetcher,
    "sec": SecFetcher,
    "stock": StockFetcher,
    "etf": None,
}
IMPLEMENTED_PREPROCESSOR = {
    "twitter": TwitterMatrixConstructor,
    "news": NewsMatrixConstructor,
    "sec": SecMatrixConstructor,
    "stock": StockMatrixConstructor,
    "etf": EtfMatrixConstructor,
}
FILE_TREE = [
    ["raw", list(IMPLEMENTED_FETCHER.keys())],
    ["processed", list(IMPLEMENTED_FETCHER.keys())],
]


class DataPipeline:
    """Data Pipeline Object class"""

    def __init__(self, config_path, components=list(IMPLEMENTED_FETCHER.keys())):
        """[summary]

        Args:
            config_path (str): file path for config files
            components ([list], optional): components for pipeline to run with. Defaults to list(IMPLEMENTED_FETCHER.keys()).

        Raises:
            NotImplementedError: only support implemented classes
        """
        for component in components:
            if component not in IMPLEMENTED_FETCHER.keys():
                raise NotImplementedError(
                    f"component not implemented, please refers to implemented modules: {IMPLEMENTED_FETCHER.keys()}"
                )
        self.components = components
        self.configs = yaml.safe_load(open(config_path, encoding="utf-8"))

    def run_pipeline(self, redownload=False):
        """main function to execute to run data pipeline

        Args:
            redownload (bool, optional): True if redownload the raw files. Defaults to False.
        """
        print("starting pipeline")
        self._create_file_tree()
        print("starting requesting data downloading")
        self._download(redownload)
        print("=======================================")
        self._preprocess()
        print("=======================================")
        print("data pipeline executed successfully")

    def _download(self, redownload):
        """helper function for downloading raw files

        Args:
            redownload (bool): True if redownload the raw files.
        """
        for component in self.components:
            print(f"working on downloading data for {component}")
            if component == "etf":
                continue
            elif component == "sec":
                if redownload or not os.path.exists(
                    os.path.join(
                        self.configs["data_root"], "raw", component, "raw.json"
                    )
                ):
                    fetcher = IMPLEMENTED_FETCHER[component](**self.configs)
                    res = fetcher.get_data()
                    with open(
                        os.path.join(
                            self.configs["data_root"], "raw", component, "raw.json"
                        ),
                        "w+",
                    ) as fp:
                        json.dump(res, fp)
            else:
                if redownload or not os.path.exists(
                    os.path.join(self.configs["data_root"], "raw", component, "raw.csv")
                ):
                    fetcher = IMPLEMENTED_FETCHER[component](**self.configs)
                    df = fetcher.get_data()
                    df.to_csv(
                        os.path.join(
                            self.configs["data_root"], "raw", component, "raw.csv"
                        ),
                        index=False,
                    )
        return

    def _preprocess(self):
        """helper function for converting raw files into numpy matrix"""
        for component in self.components:
            print(f"working on constructing matrices for {component}")
            if component == "stock":
                df = pd.read_csv(
                    os.path.join(self.configs["data_root"], "raw", component, "raw.csv")
                )
                for option in ["price", "volume"]:
                    matrix_constructer = IMPLEMENTED_PREPROCESSOR[component](
                        option, **self.configs
                    )
                    matrices = matrix_constructer.get_matrix(df)
                    for quarter, matrix in matrices.items():
                        np.save(
                            os.path.join(
                                self.configs["data_root"],
                                "raw",
                                component,
                                f"{option}_{quarter}.npy",
                            ),
                            matrix,
                        )
            elif component == "etf":
                df = None
                for option in ["bipartite", "occurence"]:
                    matrix_constructer = IMPLEMENTED_PREPROCESSOR[component](
                        option, **self.configs
                    )
                    matrices = matrix_constructer.get_matrix(df)
                    for quarter, matrix in matrices.items():
                        np.save(
                            os.path.join(
                                self.configs["data_root"],
                                "raw",
                                component,
                                f"{option}_{quarter}.npy",
                            ),
                            matrix,
                        )
            elif component == "sec":
                df = json.load(
                    open(
                        os.path.join(
                            self.configs["data_root"], "raw", component, "raw.json"
                        )
                    )
                )
                matrix_constructer = IMPLEMENTED_PREPROCESSOR[component](**self.configs)
                matrices = matrix_constructer.get_matrix(df)
                for company, matrix in matrices.items():
                    np.save(
                        os.path.join(
                            self.configs["data_root"],
                            "raw",
                            component,
                            f"{company}.npy",
                        ),
                        matrix,
                    )
            else:
                df = pd.read_csv(
                    os.path.join(self.configs["data_root"], "raw", component, "raw.csv")
                )
                matrix_constructer = IMPLEMENTED_PREPROCESSOR[component](**self.configs)
                matrices = matrix_constructer.get_matrix(df)
                for quarter, matrix in matrices.items():
                    np.save(
                        os.path.join(
                            self.configs["data_root"],
                            "raw",
                            component,
                            f"{quarter}.npy",
                        ),
                        matrix,
                    )
        return

    def _create_file_tree(self):
        """create file tree of the project"""
        if os.path.exists(self.configs["data_root"]):
            os.makedirs(self.configs["data_root"], exist_ok=True)
        for files in FILE_TREE:
            parent, children = files
            for child in children:
                os.makedirs(
                    os.path.join(self.configs["data_root"], parent, child),
                    exist_ok=True,
                )
