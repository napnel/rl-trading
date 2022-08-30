from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from gym import spaces

if TYPE_CHECKING:
    from rl_bot.envs.environment import TradingEnv


class ObserverScheme:
    def __init__(
        self,
        df_path: str,
        window_size: int,
        **kwargs,
    ):
        df: pd.DataFrame = pd.read_pickle(df_path)
        ohlcv = df[["Open", "High", "Low", "Close", "Volume"]]
        # features = df.drop(["Open", "High", "Low", "Close", "Volume"], axis=1)
        fe_cols = [col for col in df.columns if col.startswith("feature_")]
        features = df[fe_cols]
        assert features.shape[1] > 0, f"No features found, {features.columns}"
        assert len(ohlcv) == len(features), "lenth of ohlcv and features is diffrent"

        self.env: "TradingEnv" = None
        self._ohlcv = ohlcv
        self._features = features
        self.ohlcv = ohlcv.values
        self.features = features.values
        self.window_size = window_size
        self.prev_observation = None

    @property
    def observation_space(self) -> spaces.Box:
        raise NotImplementedError

    @property
    def observation(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def single_observation(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def candlestick(self) -> np.ndarray:
        """return current candlestick data"""
        return self.ohlcv[self.env.current_step, :]

    @property
    def prev_candlestick(self) -> np.ndarray:
        return self.ohlcv[self.env.current_step - 1, :]

    @property
    def price(self) -> float:
        return self.candlestick[3]

    @property
    def datetime(self) -> pd.DatetimeIndex:
        return self._features.index

    @property
    def max_steps(self) -> int:
        return len(self.ohlcv)

    def reset(self, env: "TradingEnv") -> np.ndarray:
        self.env = env
        return self.observation

    def step(self) -> np.ndarray:
        raise NotImplementedError


class PublicObserver(ObserverScheme):
    def __init__(
        self,
        df_path: str,
        window_size: int,
        **kwargs,
    ):
        super().__init__(df_path, window_size)
        self._observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(window_size, self.features.shape[1] + 2),
            dtype=np.float32,
        )
        self.positions = np.zeros((len(self.ohlcv), 2))

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def observation(self) -> np.ndarray:
        features_obs = self.features[
            self.env.current_step - self.env.window_size : self.env.current_step, :
        ]
        position_obs = self.positions[
            self.env.current_step - self.env.window_size : self.env.current_step, :
        ]
        obs = np.hstack((features_obs, position_obs))
        return obs

    def step(self) -> np.ndarray:
        long_position = self.env.position.pnl_pct if self.env.position.is_long else 0
        short_position = self.env.position.pnl_pct if self.env.position.is_short else 0
        self.positions[self.env.current_step, :] = [long_position, short_position]
        return self.observation


class MultiTimeframeObserver:
    def __init__(
        self,
        df_paths: Dict[str, str],
        step_multi: Dict[str, str],
        window_size: int,
        **kwargs,
    ):
        assert len(df_paths) > 0, "No dataframe paths found"
        self.window_size = window_size
        self.prior_tf = next(iter(df_paths.keys()))  # tf := timeframe

        df = {
            key: pd.read_pickle(path).astype(np.float32)
            for key, path in df_paths.items()
        }
        self.datetimes = {key: df[key].index for key in df.keys()}
        self.step_multi = step_multi

        self.ohlcv = {
            k: d[["Open", "High", "Low", "Close", "Volume"]] for k, d in df.items()
        }

        fe_cols = [
            col for col in df[self.prior_tf].columns if col.startswith("feature_")
        ]
        self.features = {key: d[fe_cols] for key, d in df.items()}

        space_dict = {
            key: spaces.Box(
                -np.inf,
                np.inf,
                shape=(window_size, fe_df.shape[1]),
                dtype=np.float32,
            )
            for key, fe_df in self.features.items()
        }
        space_dict["position"] = spaces.Box(
            -np.inf,
            np.inf,
            shape=(window_size, 2),
            dtype=np.float32,
        )
        self._observation_space = spaces.Dict(space_dict)
        self.positions = np.zeros((len(self.ohlcv[self.prior_tf]), 2), dtype=np.float32)
        self.steps = {key: 0 for key in self.ohlcv.keys()}

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    @property
    def observation(self) -> OrderedDict:
        obs = OrderedDict(
            [
                (
                    tf,
                    fe_df.iloc[
                        self.steps[tf] - self.env.window_size : self.steps[tf], :
                    ].values,
                )
                for tf, fe_df in self.features.items()
            ]
        )
        obs["position"] = self.positions[
            self.env.current_step - self.env.window_size : self.env.current_step, :
        ]
        # obs = self.observation_space.sample()
        return obs

    def reset(self, env: "TradingEnv") -> np.ndarray:
        self.env = env
        self.steps = {key: self.env.window_size for key in self.ohlcv.keys()}
        return self.observation

    def step(self) -> OrderedDict:
        for ts, multi in self.step_multi.items():
            if (self.steps[ts] - self.steps[self.prior_tf]) % multi == 0:
                self.steps[ts] += 1
        long_position = self.env.position.pnl_pct if self.env.position.is_long else 0
        short_position = self.env.position.pnl_pct if self.env.position.is_short else 0
        self.positions[self.env.current_step, :] = [long_position, short_position]
        return self.observation

    @property
    def candlestick(self) -> np.ndarray:
        return self.ohlcv[self.prior_tf].iloc[self.env.current_step, :]

    @property
    def prev_candlestick(self) -> np.ndarray:
        return self.ohlcv[self.prior_tf].iloc[self.env.current_step - 1, :]

    @property
    def datetime(self) -> pd.DatetimeIndex:
        return self.datetimes[self.prior_tf][self.env.current_step]

    @property
    def price(self) -> float:
        return self.candlestick[3]

    @property
    def max_steps(self) -> int:
        return len(self.ohlcv[self.prior_tf])


registry = {
    "PublicObserver": PublicObserver,
    "MultiTimeframeObserver": MultiTimeframeObserver,
}


def get(class_name: str):
    if class_name not in list(registry.keys()):
        raise KeyError(f"{class_name} does not include the registry")
    return registry[class_name]
