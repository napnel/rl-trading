import argparse
import os
from pathlib import Path
from typing import List

import pandas as pd
import ray

DATA_PATH = Path().home() / "data"

# data
# - BTCUSDT
# -- transaction.pkl
# -- ohlcv.pkl
# -- features.pkl
# - ETHUSDT
# -- transaction
# ...


@ray.remote(num_cpus=os.cpu_count())
def combine_csv_files(csv_files: List[Path]):
    df = pd.concat(
        [
            pd.read_csv(file, index_col="Datetime", parse_dates=True)
            for file in csv_files
        ]
    )
    return df


def create_ohlcv(df: pd.DataFrame, rule: str):
    ohlcv: pd.DataFrame = df["price"].resample(rule).ohlc()
    ohlcv["Volume"] = df["size"].resample(rule).sum()
    separete_volume = df.groupby("side").resample(rule).sum()
    ohlcv["Buy Volume"] = separete_volume.loc["Buy", "size"]
    ohlcv["Sell Volume"] = separete_volume.loc["Sell", "size"]
    ohlcv = ohlcv.rename(columns=str.capitalize)
    ohlcv = ohlcv.fillna(method="ffill").fillna(method="bfill")
    return ohlcv


def transac_to_ohlcv(pairs: List[str], rules: List[str]):
    """Transaction to OHLCV
    params:
        pairs: List[str] - BTCUSDT, ETHUSDT, ...
        rules: List[str] - 1T, 3T, 5T, 15H, 1H, 1D, ...
    """
    for pair in pairs:
        save_path = DATA_PATH / pair
        save_path.mkdir(exist_ok=True, parents=True)
        if not (save_path / "transaction.pkl").exists():
            transaction_path = DATA_PATH / pair / "transactions"
            save_path = DATA_PATH / pair
            csv_files = sorted(list(transaction_path.glob("*.csv")))
            print(f"{pair}: {len(csv_files)}")
            assert len(csv_files) != 0, f"{transaction_path} has no csv file"

            # df = ray.get(combine_csv_files.remote(csv_files))
            # df = ray.get(obj)
            df = pd.concat(
                [
                    pd.read_csv(file, index_col="Datetime", parse_dates=True)
                    for file in csv_files
                ]
            )
            df.to_pickle(save_path / "transaction.pkl")
        else:
            df = pd.read_pickle(save_path / "transaction.pkl")

        for rule in rules:
            ohlcv = create_ohlcv(df, rule=rule)
            ohlcv.to_pickle(save_path / f"ohlcv_{rule}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        nargs="+",
        type=str,
        default=["BTCUSDT", "ETHUSDT"],
    )
    parser.add_argument(
        "--rules",
        nargs="+",
        type=str,
        default=["5T", "15H", "1H", "4H", "1D"],
    )
    args = parser.parse_args()
    transac_to_ohlcv(args.pairs, args.rules)
