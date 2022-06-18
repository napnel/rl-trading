import argparse
import subprocess
from pathlib import Path
from typing import List

import pandas as pd
import ray
import requests
from bs4 import BeautifulSoup

DATA_PATH = Path.home() / "data"


@ray.remote
def download_transactions(url: Path, filename: str, save_path: Path):
    if not (save_path / filename.strip(".gz")).exists():
        subprocess.run(["wget", "-q", "-nd", "-P", str(save_path), url + filename])

    if (save_path / filename).exists():
        subprocess.run(["gzip", "-d", str(save_path / filename)])

        transaction = pd.read_csv(
            str(save_path / filename).strip(".gz"),
            usecols=["timestamp", "side", "size", "price"],
        )
        transaction = transaction.rename(columns={"timestamp": "Datetime"})
        transaction = transaction.set_index("Datetime")
        transaction.index = (
            pd.to_datetime(transaction.index.astype(int), unit="s")
            .tz_localize("UTC")
            .tz_convert("Asia/Tokyo")
        )
        transaction = transaction.groupby(["Datetime", "side"]).agg(
            {"size": "sum", "price": "mean"}
        )
        transaction.to_csv(str(save_path / filename).strip(".gz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--years", type=str, nargs="+", default=["2021", "2022"])
    args = parser.parse_args()

    ray.init()
    # ray.put(compressed_transaction)

    for pair in args.pairs:
        save_path = DATA_PATH / pair / "transactions"
        # save_path = Path.home() / f"data/{pair}"
        save_path.mkdir(exist_ok=True, parents=True)
        url = f"https://public.bybit.com/trading/{pair}/"

        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        a_tag = soup.find_all("a")
        target_filenames: List[str] = [a.get("href") for a in a_tag]
        target_filenames = [
            filename
            for filename in target_filenames
            for year in args.years
            if filename.startswith(pair + year)
        ]

        print("Start downloading transactions...")
        print(f"Save path: {save_path}")

        done, yet = ray.wait(
            [
                download_transactions.remote(url, filename, save_path)
                for filename in target_filenames
            ]
        )
        while len(yet):
            done, yet = ray.wait(yet)
            print(f"{len(yet)}/{len(target_filenames)}...")

        assert len(list(save_path.glob("*.csv"))) == len(
            target_filenames
        ), "Download failed."
        print("Downloading transactions finished.")

    ray.shutdown()

# Example
# https://public.bybit.com/trading/BTCUSDT/BTCUSDT2020-03-25.csv.gz
