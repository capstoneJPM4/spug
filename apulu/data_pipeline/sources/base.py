"""base for data fetcher
"""


class DataFetcher:
    """base for data fetcher"""

    def __init__(self, **configs):
        self.start_date = None
        self.end_date = None
        self.companies = []
        self.twitter_conifg = {}
        self.sec_config = {}
        self.__dict__.update(configs)
        self.companies.sort(key=lambda x: list(x.keys())[0])

    def get_data(self):
        """main file for get data"""
        pass
