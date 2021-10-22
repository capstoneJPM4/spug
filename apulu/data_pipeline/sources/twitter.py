"""
fetch twitter data
"""
import itertools
import time
from tqdm.auto import tqdm
import pandas as pd
import snscrape.modules.twitter as sntwitter

LANGUAGE = " lang:en"
SLEEP_TIME = 5


def _preprocess_query(query, start, end):
    since = " since:" + str(start)
    until = " until:" + str(end)
    return " OR ".join(query) + LANGUAGE + since + until


def _call_twitter_api(query):
    return sntwitter.TwitterSearchScraper(query=query).get_items()


def _sample_generator(items, start, stop, step):
    return list(itertools.islice(items, start, stop, step))


def _process_tweets_df(tweet_list):
    df = pd.DataFrame(
        tweet_list,
        columns=["datetime", "tweet_id", "text", "username", "ticker_symbol"],
    ).drop_duplicates()
    df.datetime = pd.to_datetime(df.datetime)
    df = df.assign(date=df.datetime.dt.date, month=df.datetime.dt.month)
    return df


def get_tweets_raw(symbols, start, end):
    """get raw data from twitter API
    Args:
        symbol (list): ticker symbols of stocks
        start (datetime.datetime): start time
        end (datetime.datetime): end time
    """
    tweets = []

    for symbol, query in tqdm(symbols.items()):
        # scraping tweets
        time.sleep(SLEEP_TIME)
        processed_query = _preprocess_query(query, start, end)
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
