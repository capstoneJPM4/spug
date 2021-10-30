from sources import TwitterFetcher, NewsFetcher, StockFetcher, SecFetcher

IMPLEMENTED_FETCHER = {
    "twitter": TwitterFetcher,
    "news": NewsFetcher,
    "sec": SecFetcher,
    "stock": StockFetcher,
    "etf": None,
}


class DataPipeline:
    def __init__(self, components):
        pass
