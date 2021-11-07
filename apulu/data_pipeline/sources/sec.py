"""
fetch SEC reports
"""
import os
import shutil
import re
import bs4 as bs
from tqdm.auto import tqdm
import pandas as pd
import json
from sec_edgar_downloader import Downloader
from .base import DataFetcher


def _parse_10K(directory, ticker, items=[]):
    """parse 10-K files for one company.

    Parameters
    ----------
    directory : str
        current working directory where file is downloaded to.

    ticker : str
        company ticker.

    items : list[str]
        sections in 10K that need to be parsed. e.g. ["1A","1B","7A","7","8","9"]
        blank will appear if regex does not capture anything.
    """
    full_path = os.path.join(directory, "sec-edgar-filings", ticker, "10-K")
    result = {}
    for folder in os.listdir(full_path):
        year = folder.split("-")[1]
        file_path = os.path.join(full_path, folder, "full-submission.txt")
        with open(file_path, "r") as f:
            webpage = f.read()
        # Regex to find <DOCUMENT> tags
        doc_start_pattern = re.compile(r"<DOCUMENT>")
        doc_end_pattern = re.compile(r"</DOCUMENT>")
        # Regex to find <TYPE> tag prceeding any characters, terminating at new line
        type_pattern = re.compile(r"<TYPE>[^\n]+")

        # Create 3 lists with the span idices for each regex
        ### There are many <Document> Tags in this text file, each as specific exhibit like 10-K, EX-10.17 etc
        ### First filter will give us document tag start <end> and document tag end's <start>
        ### We will use this to later grab content in between these tags
        doc_start_is = [x.end() for x in doc_start_pattern.finditer(webpage)]
        doc_end_is = [x.start() for x in doc_end_pattern.finditer(webpage)]

        ### Type filter is interesting, it looks for <TYPE> with Not flag as new line, ie terminare there, with + sign
        ### to look for any char afterwards until new line \n. This will give us <TYPE> followed Section Name like '10-K'
        ### Once we have have this, it returns String Array, below line will with find content after <TYPE> ie, '10-K'
        ### as section names
        doc_types = [x[len("<TYPE>") :] for x in type_pattern.findall(webpage)]

        document = {}

        # Create a loop to go through each section type and save only the 10-K section in the dictionary
        for doc_type, doc_start, doc_end in zip(doc_types, doc_start_is, doc_end_is):
            if doc_type == "10-K":
                document[doc_type] = webpage[doc_start:doc_end]

        # Write the regex
        items_regex = "|".join(items)

        regex = re.compile(
            f"(>Item(\s|&#160;|&nbsp;)({items_regex})\.{{0,1}})|(ITEM\s({items_regex}))"
        )

        # Use finditer to math the regex
        matches = regex.finditer(document["10-K"])
        # Create the dataframe
        test_df = pd.DataFrame([(x.group(), x.start(), x.end()) for x in matches])

        test_df.columns = ["item", "start", "end"]
        test_df["item"] = test_df.item.str.lower()
        # Get rid of unnesesary charcters from the dataframe
        test_df.replace("&#160;", " ", regex=True, inplace=True)
        test_df.replace("&nbsp;", " ", regex=True, inplace=True)
        test_df.replace(" ", "", regex=True, inplace=True)
        test_df.replace("\.", "", regex=True, inplace=True)
        test_df.replace(">", "", regex=True, inplace=True)
        # Drop duplicates
        pos_dat = test_df.sort_values("start", ascending=True).drop_duplicates(
            subset=["item"], keep="last"
        )
        # Set item as the dataframe index
        pos_dat.set_index("item", inplace=True)

        curr_result = {}
        for i in range(len(pos_dat) - 1):

            content = document["10-K"][
                pos_dat["start"].iloc[i] : pos_dat["start"].iloc[i + 1]
            ]
            content = bs.BeautifulSoup(content, "lxml")
            curr_result[pos_dat.index[i]] = content.get_text("\n\n")
        result[year] = curr_result
    return result


def _download_sec(tickers, directory, amount, items, external_fp):
    """batch download companies sec reports.

    Parameters
    ----------
    tickers : list[str]
        company tickers.

    directory : str
        current working directory where file is downloaded to.

    amount : int
        the most recent amount of years of sec report.

    items : list[str]
        sections in 10K that need to be parsed. e.g. ["1A","1B","7A","7","8","9"]
        blank will appear if regex does not capture anything.
    """
    dl = Downloader(directory)

    for ticker in tqdm(tickers):
        if not os.path.exists(directory):
            os.mkdir(os.path.join(directory))
        dl.get("10-K", ticker, amount=amount)
        if not os.path.exists(os.path.join(directory, "parsed")):
            os.mkdir(os.path.join(directory, "parsed"))
        errors = []
        try:
            res = _parse_10K(directory, ticker, items)
            with open(os.path.join(directory, "parsed", f"{ticker}.json"), "w+") as f:
                json.dump(res, f)
        except:
            if ticker + ".json" in os.listdir(external_fp):
                shutil.copy(
                    os.path.join(external_fp, ticker + ".json"),
                    os.path.join(directory, "parsed"),
                )
            else:
                errors.append(ticker)
            continue
    print(
        f"{len(errors)} companies 10-Ks failed to be downloaded to the directory specified at etl.yaml sec_config section"
    )
    if len(errors) > 0:
        print(errors)
    return


class SecFetcher(DataFetcher):
    def __init__(self, **configs):
        super().__init__(**configs)

    def get_data(self):
        """return raw data after fetcher created as a dictionary.

        dict format:
        {company_name:
            {year(e.g. "17","18"):
                {"10K section":raw text}
            }
        }

        """
        _download_sec(**self.sec_config["parser_config"])
        result = {}
        fp = self.sec_config["parser_config"]["directory"]
        for comp_fp in tqdm(os.listdir(os.path.join(fp, "parsed"))):
            comp = {}
            try:
                comp = json.load(open(os.path.join(fp, "parsed", comp_fp)))
            except:
                comp = eval(open(os.path.join(fp, "parsed", comp_fp)).read())
            result[comp_fp.split(".")[0]] = comp
        return result
