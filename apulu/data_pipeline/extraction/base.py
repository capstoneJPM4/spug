"""base file for Matrix Constructor
"""
import numpy as np


class MatrixConstructor:
    """Matrix Constructor Base"""

    def __init__(self, **configs):
        """base matrix constructor"""
        self.companies = []
        self.etf_config = {}
        self.sec_config = {}
        self.__dict__.update(configs)
        self.companies.sort(key=lambda x: list(x.keys())[0])

    def get_matrix(self, df):
        """main function to execute output with numpy 2d array.

        Args:
            df (pd.DataFrame, optional): dataframe in memory to preprocess. Defaults to None.

        Returns:
            [np.array: 2d np array
        """
        return np.zeros((0, 0))
