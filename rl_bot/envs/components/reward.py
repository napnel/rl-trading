from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rl_bot.envs.environment import TradingEnv


class RewardScheme:
    def __init__(self, **kwargs):
        self.env: "TradingEnv" = None

    def reset(self, env: "TradingEnv"):
        self.env: "TradingEnv" = env

    def step(self) -> float:
        raise NotImplementedError


class LogReturn(RewardScheme):
    def step(self) -> float:
        reward = np.log(self.env.equity_curve[-1]) - np.log(self.env.equity_curve[-2])
        return reward


class CumulativeReturn(RewardScheme):
    def step(self) -> float:
        reward = (self.env.equity - self.env.initial_cash) / self.env.initial_cash
        return reward


class SignalReward(RewardScheme):
    def step(self) -> float:
        reward = 0.0
        if self.env.position.pnl_pct > 0:
            reward += 1.0

        elif self.env.position.pnl_pct < 0:
            reward -= 1.0

        return reward


class PnL(RewardScheme):
    def step(self) -> float:
        # prev_candlestick = self.env.observer.prev_candlestick
        # high, low, close = prev_candlestick[1], prev_candlestick[2], prev_candlestick[3]
        reward = self.env.position.pnl_pct
        # reward_true = 0
        # reward_true = max(high - close, abs(close - low)) / close
        # reward_true = abs(close - low) / close
        return reward


class RelativeLogReturn(RewardScheme):
    def step(self) -> float:
        reward = np.log(self.env.equity_curve[1]) - np.log(self.env.equity[-2])
        long_log_return = self.env.observer.prev_candlestick


class DifferentialSharpeRatio(RewardScheme):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.a_t = None
        self.b_t = None
        self.window_size = kwargs["window_size"]
        self.eta = 2 / (self.window_size + 1)

    def reset(self, env: "TradingEnv"):
        super().reset(env)
        self.a_t = 0
        self.b_t = 0

    def step(self) -> float:
        r_t = self.env.position.pnl_pct
        if r_t == 0:
            return 0

        a_delta = r_t - self.a_t
        b_delta = r_t ** 2 - self.b_t

        nominator = self.b_t * a_delta - (0.5 * self.a_t * b_delta)
        denominator = (self.b_t - self.a_t ** 2) ** 1.5 + np.finfo("float64").eps

        reward = nominator / denominator
        self.a_t = self.a_t + self.eta * a_delta
        self.b_t = self.b_t + self.eta * b_delta

        return reward


registry = {
    "LogReturn": LogReturn,
    "PnL": PnL,
    "CumulativeReturn": CumulativeReturn,
    "DSR": DifferentialSharpeRatio,
}


def get(class_name: str):
    if class_name not in list(registry.keys()):
        raise KeyError(f"{class_name} does not include the registry")
    return registry[class_name]
