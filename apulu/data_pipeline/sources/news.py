from .base import DataFetcher


class NewsFetcher(DataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self, start, end, sleep_time):
        """get raw data from news API
        Args:
            symbol (list): ticker symbols of stocks
            start (datetime.datetime): start time
            end (datetime.datetime): end time
        """
        return super().get_data(start, end, sleep_time)
