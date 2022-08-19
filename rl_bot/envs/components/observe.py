from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from gym import spaces

if TYPE_CHECKING:
    from rl_bot.envs.environment import TradingEnv


class ObserverScheme:
    def __init__(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int,
        **kwargs,
    ):
        assert len(ohlcv) == len(features), "lenth of ohlcv and features is diffrent"
        assert list(ohlcv.columns) == [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ], f"{list(ohlcv.columns)}"
        self.env: "TradingEnv" = None
        self._ohlcv = ohlcv
        self._features = features
        self.ohlcv = ohlcv.values
        self.features = features.values
        self.window_size = window_size
        self.datetime = features.index
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

    def reset(self, env: "TradingEnv") -> np.ndarray:
        self.env = env
        return self.observation

    def step(self) -> np.ndarray:
        raise NotImplementedError


class PublicObserver(ObserverScheme):
    def __init__(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int,
        **kwargs,
    ):
        super().__init__(ohlcv, features, window_size)
        self._observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(window_size, self.features.shape[1] + 2),
            dtype=np.float32,
        )
        self.positions = np.zeros((len(ohlcv), 2))
        # self.positions = np.tile([0, 0], (self.window_size, len(ohlcv)))

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

    # @property
    # def single_observation(self) -> np.ndarray:
    #     return self.features[self.env.current_step, :]

    def step(self) -> np.ndarray:
        long_position = self.env.position.pnl_pct if self.env.position.is_long else 0
        short_position = self.env.position.pnl_pct if self.env.position.is_short else 0
        self.positions[self.env.current_step, :] = [long_position, short_position]
        return self.observation


class MultiTimeframeObserver(ObserverScheme):
    def __init__(
        self,
        ohlcv: List[pd.DataFrame],
        features: List[pd.DataFrame],
        window_size: int,
        **kwargs,
    ):
        super().__init__(ohlcv, features, window_size)
        self._observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(window_size, self.features.shape[1] + 2),
            dtype=np.float32,
        )
        self.positions = np.zeros((len(ohlcv), 2))
        # self.positions = np.tile([0, 0], (self.window_size, len(ohlcv)))

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

    # @property
    # def single_observation(self) -> np.ndarray:
    #     return self.features[self.env.current_step, :]

    def step(self) -> np.ndarray:
        long_position = self.env.position.pnl_pct if self.env.position.is_long else 0
        short_position = self.env.position.pnl_pct if self.env.position.is_short else 0
        self.positions[self.env.current_step, :] = [long_position, short_position]
        return self.observation


class NormObserver(ObserverScheme):
    def __init__(
        self,
        ohlcv: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int,
        **kwargs,
    ):
        super().__init__(ohlcv, features, window_size)
        self._observation_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(window_size, self.features.shape[1]),
            dtype=np.float32,
        )

    @property
    def observation_space(self) -> spaces.Box:
        return self._observation_space

    @property
    def observation(self) -> np.ndarray:
        features_obs = self.features[
            self.env.current_step - self.env.window_size : self.env.current_step, :
        ]
        assert not np.isnan(features_obs).sum().all(), "Include NaN"
        assert features_obs.shape == (
            self.env.window_size,
            self.features.shape[1],
        ), f"{features_obs.shape}"
        return features_obs

    @property
    def single_observation(self) -> np.ndarray:
        return self.features[self.current_step, :]

    def step(self) -> np.ndarray:
        return self.observation


registry = {
    "PublicObserver": PublicObserver,
}


def get(class_name: str):
    if class_name not in list(registry.keys()):
        raise KeyError(f"{class_name} does not include the registry")
    return registry[class_name]
