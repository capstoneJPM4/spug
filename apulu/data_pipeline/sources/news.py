"""
fetch news data
"""
import itertools
import time
import pandas as pd
import datetime
import dateutil
from tqdm import tqdm
from urllib.request import urlopen
from bs4 import BeautifulSoup
from .base import DataFetcher

NYT_KEY = "MN3m4QX4XDbGsaBFDBpJG4waHymmqZ3O"
BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json?"
START_DATE = datetime.date(2016, 9, 1)
END_DATE = datetime.date(2021, 9, 1)


def _article_search(company, start_day, end_day):
    """helper function to search for articles mentioning the company
    Args:
      company (string): the company name that should be mentioned
      start_day (datetime.date): the earliest publishing day of the articles
      end_day (datetime.date): the latest publishing day of the articles
    Returns:
      response: the json object storing the info about the articles and urls
    """
    url = (
        BASE_URL
        + "q="
        + company
        + "&begin_date="
        + start_day
        + "&end_date="
        + end_day
        + "&fq=Financial"
        + "&fl=web_url"
        + "&api-key="
        + NYT_KEY
    )
    response = requests.get(url).json()
    time.sleep(6)
    return response


def scraping(url):
    """helper function to scrape an article from a website
    Args:
      url (string): the url of the website
    Returns:
      article: a string containing the full article
    """
    try:
        page = urlopen(url)
    except:
        print("Error opening the URL")
    soup = BeautifulSoup(page, "html.parser")
    content = soup.findAll("div", {"class": "css-53u6y8"})
    article = ""
    for each_content in content:
        for i in each_content.findAll("p"):
            article = article + " " + i.text
    return article


class NewsFetcher(DataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self):
        """get raw data from news API
        Args:
            symbol (list): ticker symbols of stocks
            start (datetime.datetime): start time
            end (datetime.datetime): end time
        """
        df_article = pd.DataFrame({"Date": [], "url": []})
        num = 0
        for c in tqdm(self.companies):
            day = self.start_date
            while day < self.end_date:
                res = article_search(
                    c,
                    day.strftime("%Y%m%d"),
                    (day + datetime.timedelta(30)).strftime("%Y%m%d"),
                )
                day += datetime.timedelta(31)
                num += 1
                if "response" not in res:
                    continue
                for i in res["response"]["docs"]:
                    tmp = i["web_url"]
                    df_article = df_article.append(
                        {"Date": day.strftime("%Y-%m-%d"), "url": tmp},
                        ignore_index=True,
                    )
                if num > 3000:
                    break
            if num > 3000:
                break

        txts = []
        for i in tqdm(df_article.url):
            time.sleep(2)
            txts.append(scraping(i))
        df_article["texts"] = txts
        return df_article
