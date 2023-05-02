import sys
import warnings
from enum import IntEnum
from math import copysign
from typing import Any, Callable, Dict, List, Optional

import gym
import numpy as np
import pandas as pd
from gym import spaces

from envs.actions import LongNeutralShort
from envs.core import Order, Position, Trade
from envs.reward_func import equity_log_return_reward


class TradingEnv(gym.Env):
    def __init__(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        window_size: int = 25,
        fee: float = 0.001,
        reward_func: Callable = equity_log_return_reward,
        actions: IntEnum = LongNeutralShort,
        debug: bool = False,
    ):
        """ """
        assert len(data) == len(
            features
        ), f"The data and features sizes are different: Data: {len(data)}, Features: {len(features)}"
        self.data = data.copy()
        self.features = features.copy()
        self.fee = fee
        self.window_size = window_size
        self.reward_func = reward_func
        self.actions = actions
        self.debug = debug

        self.current_step = 0
        self.initial_assets = 100000
        self.assets = self.initial_assets
        self.trade_size = self.assets * 0.75 // self.data["High"].max()
        self.position: Optional[Position] = Position(self)
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.observation_size = len(self.features.columns) + 3

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.observation_size),
            dtype=np.float32,
        )

        self._leverage = 1
        self._hedging = False

    def reset(self):
        self.done = False
        self.next_done = False
        self.action = None
        self.reward = 0
        self.current_step = self.window_size
        self.assets = self.initial_assets
        self.trade_size = self.assets // self.data["High"].max()
        self.position = Position(self)
        self.orders = []
        self.trades = []
        self.equity_curve = [self.equity]
        self.closed_trades = []

        features_obs = self.features.iloc[: self.window_size, :].values
        account_obs = np.tile([0, 0, 1], (self.window_size, 1))
        self.observation = np.hstack((features_obs, account_obs))
        return self.observation

    def step(self, action):
        self.action = action

        # Trade Start
        if self.next_done:
            self.done = True
            self.position.close()

        else:
            self.actions.perform(self, action)

        if self.debug:
            self.render()
        # Trade End
        self.current_step += 1

        self._process_orders()

        self.equity_curve.append(self.equity)
        if self.equity < self.closing_price * self.trade_size:
            self.next_done = True

        self.next_done = True if self.current_step >= len(self.data) - 3 else False

        self.observation = self.next_observation
        self.reward = self.reward_func(self)
        self.info = {}

        return self.observation, self.reward, self.done, self.info
