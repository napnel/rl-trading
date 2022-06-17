import numpy as np
import pandas as pd


def extract_features(data: pd.DataFrame):
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
    features["log_volume"] = volume.apply(np.log1p)
    features["candle_value"] = ((close - open) / (high - low)).fillna(0)
    features["true_range"] = (
        pd.concat(
            [np.abs(high - prev_close), np.abs(low - prev_close), high - low],
            axis=1,
        ).max(axis=1)
        / prev_close
    )
    features["shadow_range"] = ((high - low) - np.abs(open - close)) / prev_close
    features["real_body"] = abs(open - close) / prev_close

    features = features.fillna(0)
    # aggregated_periods = [5, 20, 50, 100, 200]
    aggregated_periods = [5, 20]

    for period in aggregated_periods:
        features[f"candle_value_{period}"] = (
            features["candle_value"].rolling(period).mean()
        )
        features[f"gap_ma_{period}"] = (
            close - close.rolling(period).mean()
        ) / close.rolling(period).mean()
        features[f"true_range_{period}"] = features["true_range"].rolling(period).mean()
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
    return features.dropna(how="any", axis=0)
