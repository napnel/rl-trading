import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DATA_PATH = Path("./data").resolve()


def attach_features(df: pd.DataFrame):
    features = pd.DataFrame()
    open, high, low, close, volume = (
        df["Open"],
        df["High"],
        df["Low"],
        df["Close"],
        df["Volume"],
    )
    prev_close = close.shift()

    base_features = [
        "candle_value",
        "log_return",
        "real_body",
        "shadow_range",
        "upper_shadow",
        "lower_shadow",
        "true_range",
        "return_volume_pct",
        "true_log_return",
    ]

    log_return = close.apply(np.log).diff()
    true_price = (high + low + close) / 3
    true_log_return = true_price.apply(np.log).diff()
    features["candle_value"] = (close - open) / (high - low)
    features["log_return"] = log_return
    features["true_log_return"] = true_log_return
    features["range"] = (high - low) / prev_close
    features["real_body"] = np.abs(close - open) / prev_close
    features["upper_shadow"] = (high - close) / prev_close
    features["lower_shadow"] = (low - close) / prev_close
    features["shadow_range"] = ((high - low) - np.abs(open - close)) / prev_close
    features["true_range"] = (
        pd.concat(
            [np.abs(high - prev_close), np.abs(low - prev_close), high - low],
            axis=1,
        ).max(axis=1)
        / prev_close
    )
    flow = np.log1p(close * volume)
    log_volume = df["Volume"].apply(np.log)
    features["log_volume_diff"] = log_volume.diff()
    features["return_volume_pct"] = features["log_return"] * log_volume.diff()
    features["rank_volume"] = log_volume.rolling(20).rank(pct=True)

    # log_volume = np.log1p(close * volume).diff()

    for period in [3, 5, 10, 20]:
        features[f"price_momentum_{period}"] = close / close.shift(period)
        features[f"volume_momentum_{period}"] = log_volume / log_volume.shift(period)
        # features[f"high_low_range_{period}"] = (
        #     high.rolling(period).max() - low.rolling(period).min()
        # ) / close
        features[f"gap_ma_{period}"] = close / close.rolling(period).mean()
        features[f"exceed_high_{period}"] = close / high.rolling(period).max().shift()
        features[f"exceed_low_{period}"] = close / low.rolling(period).max().shift()
        features[f"volatility_{period}"] = features["log_return"].rolling(period).std()

    # diff_features = pd.concat(
    #     [
    #         features[tech_features].pct_change(period).add_suffix(f"_diff_{period}")
    #         for period in [1, 5, 10]
    #     ],
    #     axis=1,
    # )
    # base_features.remove("log_volume")
    lag_features = pd.concat(
        [
            features[base_features].shift(lag).add_suffix(f"_lag_{lag}")
            for lag in range(1, 3)
        ],
        axis=1,
    )
    # features = features.drop(["obv", "adi"], axis=1)
    agg_features = pd.concat(
        [
            features[base_features]
            .rolling(window)
            .agg([np.mean])
            .rename(columns=lambda col: col + f"_{window}", level=1)
            for window in [5, 10, 20]
        ],
        axis=1,
    )
    agg_features.columns = ["_".join(col) for col in agg_features.columns.values]
    # features = pd.concat([features, lag_features, diff_features, agg_features], axis=1)
    features = pd.concat([features, agg_features], axis=1)
    # features = features.drop(["log_volume"], axis=1)
    features = features.add_prefix("feature_")

    return pd.concat([df, features], axis=1).sort_index(axis=1).dropna()


def simple_attach_features(df: pd.DataFrame):
    df = np.log1p(df)
    open, high, low, close, volume = (
        df["Open"],
        df["High"],
        df["Low"],
        df["Close"],
        df["Volume"],
    )
    # open = np.log1p(open)
    # high = np.log1p(high)
    # low = np.log1p(low)
    # close = np.log1p(close)
    # volume = np.log1p(volume)
    df["candle_value"] = (close - open) / (high - low)
    df["log_return"] = close.diff()
    df["log_volume"] = volume
    df["real_body"] = np.abs(close - open)
    df["upper_shadow"] = np.abs(high - close)
    df["lower_shadow"] = np.abs(low - close)
    return df.dropna()


def setup_data(pair: str, filename: str):
    path = DATA_PATH / pair
    df: pd.DataFrame = pd.read_pickle(path / "candlesticks" / filename)
    assert set(df.columns) >= set(["Open", "High", "Low", "Close", "Volume"])
    df = df[-3000:]
    df = attach_features(df)
    features_list = [name for name in list(df.columns) if "feature_" in name]
    print(features_list)
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False)
    scaler = RobustScaler()
    df_train[features_list] = pd.DataFrame(
        scaler.fit_transform(df_train[features_list]),
        columns=features_list,
        index=df_train.index,
    )
    df_test[features_list] = pd.DataFrame(
        scaler.transform(df_test[features_list]),
        columns=features_list,
        index=df_test.index,
    )
    df.to_pickle(path / "features" / "df.pkl")
    df_train.to_pickle(path / "features" / "df_train.pkl")
    df_test.to_pickle(path / "features" / "df_test.pkl")
    with open(path / "features" / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(df_train.describe())
    print(df_test.describe())


if __name__ == "__main__":
    for pair in ["BTCUSDT", "ETHUSDT"]:
        print(pair)
        [setup_data(pair, filename) for filename in ["5T.pkl", "15T.pkl"]]
