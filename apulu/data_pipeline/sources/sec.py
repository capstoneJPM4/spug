from .base import dataFetcher


class secFetcher(dataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self, start, end, sleep_time):
        return super().get_data(start, end, sleep_time)
