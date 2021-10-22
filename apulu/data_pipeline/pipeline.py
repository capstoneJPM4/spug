from sources import twitterFetcher, newsFetcher, stockFetcher, secFetcher

IMPLEMENTED_FETCHER = {
    "twitter": twitterFetcher,
    "news": newsFetcher,
    "sec": secFetcher,
    "stock": stockFetcher,
    "etf": None,
}


class dataPipeline:
    def __init__(self, components):
        pass
