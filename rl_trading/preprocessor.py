import warnings
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
# from minepy import MINE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import BollingerBands, average_true_range
from ta.volume import chaikin_money_flow, money_flow_index


class Preprocessor:
    @classmethod
    def preprocessing(
        self,
        data: pd.DataFrame,
        train_start: str = "2009-01-01",
        train_end: str = "2018-01-01",
        eval_start: str = "2018-01-01",
        eval_end: str = "2021-01-01",
    ):
        self.clean_data(data)
        features = self.extract_features(data)

        features_train = features.loc[train_start:train_end]
        features_eval = features.loc[eval_start:eval_end]

        data_train, features_train = self.align_date(data, features_train)
        data_eval, features_eval = self.align_date(data, features_eval)

        scaler = StandardScaler()
        features_train = pd.DataFrame(
            scaler.fit_transform(features_train),
            index=features_train.index,
            columns=features_train.columns,
        )
        features_eval = pd.DataFrame(
            scaler.fit_transform(features_eval),
            index=features_eval.index,
            columns=features_eval.columns,
        )

        return data_train, features_train, data_eval, features_eval

    @classmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        assert not data.isnull().sum().any()
        return data

    @classmethod
    def extract_features(self, data: pd.DataFrame):
        features = pd.DataFrame(index=data.index)

        open, high, low, close, volume = (
            data["Open"],
            data["High"],
            data["Low"],
            data["Close"],
            data["Volume"],
        )
        prev_close = close.shift(1)

        # features["log_return"] = data["Close"].apply(np.log1p).diff()
        features["rate_of_return"] = close.pct_change()
        features["volume"] = volume.apply(np.log1p)
        features["candle_value"] = ((close - open) / (high - low)).fillna(0)
        features["true_range"] = (
            pd.concat(
                [np.abs(high - prev_close), np.abs(low - prev_close)], axis=1
            ).max(axis=1)
            / prev_close
        )
        features["gap_range"] = np.abs(open - prev_close) / prev_close
        features["day_range"] = (high - low) / prev_close
        features["shadow_range"] = ((high - low) - np.abs(open - close)) / prev_close
        features = features.fillna(0)
        aggregated_periods = [5, 10, 20, 50, 100, 200]

        for period in aggregated_periods:
            features[f"rate_of_return_{period}"] = close.pct_change(period)
            features[f"volume_{period}"] = features["volume"].rolling(period).mean()
            features[f"candle_value_{period}"] = (
                features["candle_value"].rolling(period).mean()
            )
            features[f"volatility_{period}"] = (
                features["rate_of_return"].rolling(period).std()
            )
            features[f"gap_ma_{period}"] = (
                close - close.rolling(period).mean()
            ) / close.rolling(period).mean()
            features[f"true_range_{period}"] = (
                features["true_range"].rolling(period).mean()
            )
            features[f"gap_range_{period}"] = (
                features["gap_range"].rolling(period).mean()
            )
            features[f"day_range_{period}"] = (
                features["day_range"].rolling(period).mean()
            )
            features[f"shadow_range_{period}"] = (
                features["shadow_range"].rolling(period).mean()
            )
            features[f"hl_range_{period}"] = (
                high.rolling(period).max() - low.rolling(period).min()
            )

        # Trend Indicators
        macd = MACD(data["Close"], window_slow=26, window_fast=12, window_sign=9)
        adx = ADXIndicator(data["High"], low, data["Close"], window=14)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            features["ADX"] = adx.adx()

        features["MACD Hist"] = macd.macd_diff()

        # Oscillator Indicator
        rsi = RSIIndicator(data["Close"], window=14).rsi()
        stoch_rsi = StochRSIIndicator(data["Close"])
        features["RSI"] = rsi
        features["StochRSI"] = stoch_rsi.stochrsi_k() - stoch_rsi.stochrsi_d()

        # Volume Indicator
        features["MFI"] = money_flow_index(
            data["High"], low, data["Close"], volume, window=14
        )
        features["CMF"] = chaikin_money_flow(
            data["High"], low, data["Close"], data["Volume"]
        )

        # Volatility Indicator
        features["ATR"] = average_true_range(high, low, close, window=14)
        bb = BollingerBands(data["Close"], window=20, window_dev=2)
        features["BB %B"] = bb.bollinger_pband()
        data, features = self.align_date(data, features)
        return data, features.sort_index(axis=1)

    @classmethod
    def extract_features_v2(self, data: pd.DataFrame):
        features = pd.DataFrame(index=data.index)

        open, high, low, close, volume = (
            data["Open"],
            data["High"],
            data["Low"],
            data["Close"],
            data["Volume"],
        )
        prev_close = close.shift(1)

        features["log_return"] = close.apply(np.log1p).diff()
        # features["rate_of_return"] = close.pct_change()
        features["log_volume"] = volume.apply(np.log1p)
        features["candle_value"] = ((close - open) / (high - low)).fillna(0)
        features["true_range"] = (
            pd.concat(
                [np.abs(high - prev_close), np.abs(low - prev_close), high - low],
                axis=1,
            ).max(axis=1)
            / prev_close
        )
        features["gap_range"] = np.abs(open - prev_close) / prev_close
        features["shadow_range"] = ((high - low) - np.abs(open - close)) / prev_close
        features["real_body"] = abs(open - close) / prev_close
        features = features.fillna(0)
        aggregated_periods = [5, 20, 50, 100, 200]

        for period in aggregated_periods:
            features[f"candle_value_{period}"] = (
                features["candle_value"].rolling(period).mean()
            )
            features[f"gap_ma_{period}"] = (
                close - close.rolling(period).mean()
            ) / close.rolling(period).mean()
            features[f"true_range_{period}"] = (
                features["true_range"].rolling(period).mean()
            )
            features[f"high_low_range_{period}"] = (
                pd.concat(
                    [
                        high.rolling(period).max() - close,
                        close - low.rolling(period).min(),
                    ],
                    axis=1,
                ).min(axis=1)
                / close
            )
        data, features = self.align_date(data, features)
        return data, features.sort_index(axis=1)

    @classmethod
    def extract_features_from_ohlc(self, data: pd.DataFrame):
        features = pd.DataFrame(index=data.index)
        open, high, low, close = (
            data["Open"],
            data["High"],
            data["Low"],
            data["Close"],
        )
        prev_close = close.shift(1)

        features["log_return"] = close.apply(np.log1p).diff()
        features["candle_value"] = ((close - open) / (high - low)).fillna(0)
        features["true_range"] = pd.concat(
            [np.abs(high - prev_close), np.abs(low - prev_close), high - low],
            axis=1,
        ).max(axis=1)
        features["gap_range"] = np.abs(open - prev_close)
        features["shadow_range"] = (high - low) - np.abs(open - close)
        features["real_body"] = abs(open - close)
        features = features.fillna(0)
        aggregated_periods = [5, 20, 50]

        for period in aggregated_periods:
            features[f"candle_value_{period}"] = (
                features["candle_value"].rolling(period).mean()
            )
            features[f"gap_ma_{period}"] = (
                close - close.rolling(period).mean()
            ) / close.rolling(period).mean()
            features[f"true_range_{period}"] = (
                features["true_range"].rolling(period).mean()
            )
            features[f"high_low_range_{period}"] = pd.concat(
                [
                    high.rolling(period).max() - close,
                    close - low.rolling(period).min(),
                ],
                axis=1,
            ).min(axis=1)
        data, features = self.align_date(data, features)
        return data, features.sort_index(axis=1)

    @classmethod
    def time_train_test_split(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        train_start: str = "2009-01-01",
        train_end: str = "2019-01-01",
        eval_start: str = "2019-01-01",
        eval_end: str = "2021-01-01",
        scaling: bool = True,
    ):
        # assert (data.index == features.index).all(), f"{data.index}, {features.index}"
        data_train = data.loc[train_start:train_end]
        data_eval = data.loc[eval_start:eval_end]

        features_train = features.loc[train_start:train_end]
        features_eval = features.loc[eval_start:eval_end]

        if scaling:
            scaler = StandardScaler()
            features_train = pd.DataFrame(
                scaler.fit_transform(features_train),
                index=features_train.index,
                columns=features_train.columns,
            )
            features_eval = pd.DataFrame(
                scaler.transform(features_eval),
                index=features_eval.index,
                columns=features_eval.columns,
            )

        return data_train, features_train, data_eval, features_eval

    @classmethod
    def blocked_cross_validation(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        n_splits: int = 5,
        train_years: int = 5,
        eval_years: int = 1,
        train_start: str = "2010-01-01",
        scaling: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_start = datetime.strptime(train_start, "%Y-%m-%d")
        for fold_idx in range(n_splits):
            train_end = train_start + relativedelta(years=train_years)
            eval_start = train_start + relativedelta(years=train_years)
            eval_end = eval_start + relativedelta(years=eval_years)

            data_train = data.loc[train_start:train_end]
            data_eval = data.loc[eval_start:eval_end]
            features_train = features.loc[train_start:train_end]
            features_eval = features.loc[eval_start:eval_end]

            if scaling:
                scaler = StandardScaler()
                features_train = pd.DataFrame(
                    scaler.fit_transform(features_train),
                    index=features_train.index,
                    columns=features_train.columns,
                )
                features_eval = pd.DataFrame(
                    scaler.transform(features_eval),
                    index=features_eval.index,
                    columns=features_eval.columns,
                )

            yield data_train, features_train, data_eval, features_eval

            train_start = train_start + relativedelta(years=1)

    @classmethod
    def create_cv_from_index(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        index,
        train_years,
        eval_years,
        train_start="2010-01-01",
    ):
        train_start = datetime.strptime(train_start, "%Y-%m-%d") + relativedelta(
            years=index
        )
        train_end = train_start + relativedelta(years=train_years)
        eval_start = train_end
        eval_end = eval_start + relativedelta(years=eval_years)
        test_start = eval_end
        test_end = test_start + relativedelta(years=1)

        data_train = data.loc[train_start:train_end].copy()
        data_eval = data.loc[eval_start:eval_end].copy()
        data_test = data.loc[test_start:test_end].copy()

        features_train = features.loc[train_start:train_end].copy()
        features_eval = features.loc[eval_start:eval_end].copy()
        features_test = features.loc[test_start:test_end].copy()

        scaler = StandardScaler()
        features_train = pd.DataFrame(
            scaler.fit_transform(features_train),
            index=features_train.index,
            columns=features_train.columns,
        )
        features_eval = pd.DataFrame(
            scaler.transform(features_eval),
            index=features_eval.index,
            columns=features_eval.columns,
        )
        features_test = pd.DataFrame(
            scaler.transform(features_test),
            index=features_test.index,
            columns=features_test.columns,
        )

        assert len(data_train) == len(features_train)
        assert len(data_eval) == len(features_eval)
        assert len(data_test) == len(features_test)

        return (
            data_train,
            features_train,
            data_eval,
            features_eval,
            data_test,
            features_test,
        )

    @classmethod
    def train_test_split(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        test_size: float = 0.2,
        scaling: bool = True,
    ):
        # assert (data.index == features.index).all(), f"{data.index}, {features.index}"
        data_train, data_eval = train_test_split(
            data, test_size=test_size, shuffle=False
        )
        features_train, features_eval = train_test_split(
            features, test_size=test_size, shuffle=False
        )

        if scaling:
            scaler = MinMaxScaler()
            features_train = pd.DataFrame(
                scaler.fit_transform(features_train),
                index=features_train.index,
                columns=features_train.columns,
            )
            features_eval = pd.DataFrame(
                scaler.transform(features_eval),
                index=features_eval.index,
                columns=features_eval.columns,
            )

        return data_train, features_train, data_eval, features_eval

    @classmethod
    def align_date(self, data: pd.DataFrame, features: pd.DataFrame):
        features = features.dropna(axis=0, how="any")
        data = data.loc[features.index[0] : features.index[-1]].dropna(axis=0)
        assert len(data) == len(
            features
        ), f"data length: {len(data)}, features length: {len(features)}"
        return data, features


def main():
    import matplotlib.pyplot as plt

    df = (
        pd.read_csv("./data/3600/ethusd/2021-01-01.csv", parse_dates=[0]).set_index(
            "Date"
        )
        / 10
    )

    df = Preprocessor.clean_data(df)
    features = Preprocessor.extract_features(df)
    df, features = Preprocessor.align_date(df, features)

    features.hist(figsize=(12, 10), bins=30)
    plt.show()


if __name__ == "__main__":
    main()
