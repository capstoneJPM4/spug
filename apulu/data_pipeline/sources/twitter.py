"""
fetch twitter data
"""
import itertools
import time
from tqdm import tqdm
import pandas as pd
import snscrape.modules.twitter as sntwitter
from .base import dataFetcher

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
    df = df.assign(date=df.datetime.dt.date, month=df.datetime.dt.month)
    return df


class twitterFetcher(dataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self, start, end):
        """get raw data from twitter API
        Args:
            symbol (list): ticker symbols of stocks
            start (datetime.datetime): start time
            end (datetime.datetime): end time
        """
        tweets = []

        for company in tqdm(self.companies):
            # scraping tweets
            symbol, alias = list(company.items())[0]
            alias = alias["alias"]
            time.sleep(self.twitter_conifg["sleep_time"])
            processed_query = _preprocess_query(alias, start, end)
            scraped_tweets = _call_twitter_api(processed_query)

            sample = (
                _sample_generator(scraped_tweets, 0, 100, 1)
                + _sample_generator(scraped_tweets, 100, 500, 10)
                + _sample_generator(scraped_tweets, 500, 1000, 100)
            )
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
