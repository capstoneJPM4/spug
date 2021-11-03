from sec_edgar_downloader import Downloader
from .base import DataFetcher

def 


class SecFetcher(DataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self, start, end, sleep_time):
        return super().get_data(start, end, sleep_time)
