"""
fetch twitter data
"""
import itertools
import time
import datetime as dt
from tqdm import tqdm
import pandas as pd
import snscrape.modules.twitter as sntwitter
from .base import DataFetcher

LANGUAGE = " lang:en"


def _preprocess_query(query, start, end):
    """helper function to preprocess requesting query.
    Args:
        query (list): list of ticker symbol or equavalent alias
        start (datetime.datetime): start time
        end (datetime.datetime): end time
    Returns:
        str: query string for snscrape twitter API
    """
    since = " since:" + str(start)
    until = " until:" + str(end)
    return " OR ".join(query) + LANGUAGE + since + until


def _call_twitter_api(query):
    """helper function to call twitter api
    Args:
        query (str): query string made by _preprocess_query function
    Returns:
        generator: response object in generator
    """
    return sntwitter.TwitterSearchScraper(query=query).get_items()


def _sample_generator(items, start, stop, step):
    """helper function to sample response from twitter api.
    Args:
        items (generator): generator object
        start (int): start index for sample
        stop (int): end index for sample
        step (int): step range
    Returns:
        [type]: [description]
    """
    return list(itertools.islice(items, start, stop, step))


def _process_tweets_df(tweet_list):
    """helper function to format tweets list.
    Args:
        tweet_list (list): list of tweets with elements as follow:
            [
                "datetime",
                "tweet_id",
                "text",
                "username",
                "ticker_symbol"
            ]
    Returns:
        pandas.DataFrame: pandas dataframe of fetched twitter response
    """
    df = pd.DataFrame(
        tweet_list,
        columns=["datetime", "tweet_id", "text", "username", "ticker_symbol"],
    ).drop_duplicates()
    df.datetime = pd.to_datetime(df.datetime)
    df = df.assign(
        date=df.datetime.dt.date,
        month=df.datetime.apply(lambda x: f"{x.year}_{x.month}"),
        year=df.datetime.dt.year,
        quarter=df.datetime.apply(lambda x: f"{x.year}_q{x.quarter}"),
    )
    return df.dropna(subset=["quarter"])


class TwitterFetcher(DataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self):
        """get raw data from twitter API
        Args:
            symbol (list): ticker symbols of stocks
            start (datetime.datetime): start time
            end (datetime.datetime): end time
        """
        tweets = []
        start_year = self.start_date.year
        end_year = self.end_date.year
        start_month = self.start_date.month
        end_month = self.end_date.month
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                if year == start_year and month < start_month:
                    continue
                if year == end_year and month > end_month:
                    continue
                print(f"downloading tweets for {year}, {month}")
                start_date = dt.datetime(year, month, 1)
                if month != 12:
                    end_date = dt.datetime(year, month + 1, 1)
                else:
                    end_date = dt.datetime(year + 1, 1, 1)
                for company in tqdm(self.companies):
                    # scraping tweets
                    symbol, alias = list(company.items())[0]
                    alias = alias["alias"]
                    time.sleep(self.twitter_conifg["sleep_time"])
                    processed_query = _preprocess_query(alias, start_date, end_date)
                    scraped_tweets = _call_twitter_api(processed_query)

                    sample = _sample_generator(scraped_tweets, 0, 2000, 1)
                    sample = list(
                        map(
                            lambda tweet: [
                                tweet.date,
                                tweet.id,
                                tweet.content,
                                tweet.username,
                                symbol,
                            ],
                            sample,
                        )
                    )
                    tweets.extend(sample)

        return _process_tweets_df(tweets)
