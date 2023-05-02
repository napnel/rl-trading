import glob
import os
from typing import Optional

import pandas as pd
import yfinance as yf

# from rl_trading.preprocessor import Preprocessor


class DataLoader:
    @classmethod
    def load_data(self, filename: str) -> pd.DataFrame:
        data = pd.read_csv(filename, parse_dates=[0]).set_index("Datetime")
        return data

    @classmethod
    def split_data(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df.loc[start:end, :]

    @classmethod
    def fetch_data(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
    ):
        """
        Feath historical data (OHLCV) using Yahoo Finance API.
        """
        data = yf.download(
            ticker, start=start, end=end, interval=interval, auto_adjust=True
        )
        if interval != "1d":
            data = data.reset_index().rename(columns={"index": "Date"})
            data["Date"] = data["Date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M"))
            data = data.set_index("Date")

        assert data.index.name == "Date"
        assert list(data.columns.values) == ["Open", "High", "Low", "Close", "Volume"]
        return data

    # @classmethod
    # def prepare_data(self, ticker: str, data_path: str):

    #     data_path = os.path.join(data_path, ticker)
    #     if len(glob.glob(f"{data_path}/*.csv")) != 0:
    #         data = DataLoader.load_data(f"{data_path}/ohlcv.csv")
    #         features = DataLoader.load_data(f"{data_path}/features.csv")
    #     else:
    #         os.makedirs(f"{data_path}", exist_ok=True)
    #         data = DataLoader.fetch_data(f"{ticker}", interval="1d", start="2009-01-01")
    #         data, features = Preprocessor.extract_features_v2(data)
    #         # features = Preprocessor.select_feature(data, features)
    #         data.to_csv(f"{data_path}/ohlcv.csv")
    #         features.to_csv(f"{data_path}/features.csv")

    #     assert len(data) == len(features)
    #     return data, features
