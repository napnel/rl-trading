import os
import warnings
from copy import copy
from typing import Optional

import pandas as pd

from rl_bot.envs.environment import TradingEnv

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    from backtesting import Backtest, Strategy

from rl_bot.envs.components import action


class DRLStrategy(Strategy):
    env: TradingEnv = None
    agent = None
    debug = False

    def init(self):
        self.env.reset()
        self.observation = None
        self.done = False
        # self.actions = action.OneWayLimitOrder(self.env.actions.atr)
        # self.actions = action.OneWayLimitOrderV2(self.env.actions.atr)
        # self.actions = self.env.actions
        print(self.env.actions.env)
        self.actions = copy(self.env.actions)
        self.actions.reset(self)
        print(self.actions.env)
        # self.actions = action.MarketOrder()

    def next(self):
        if (
            self.data.index[-1] != self.env.observer.datetime[self.env.current_step]
            or self.done
        ):
            pass

        else:
            assert self.data.Close[-1] == self.env.current_price, self.error()
            assert self._broker._cash == self.env.cash, self.error()
            assert self.equity == self.env.equity, self.error()
            if self.agent == "Random":
                action = self.env.action_space.sample()
            elif self.agent == "Buy&Hold":
                action = 2 if len(self.env.actions) == 3 else 1
            elif self.agent == "Sell&Hold":
                action = 0 if len(self.env.actions) == 2 else 0
            elif self.agent:
                action = self.agent.compute_single_action(
                    self.env.observer.observation,
                    explore=False,
                    clip_action=True,
                )

            if self.debug:
                self.env.render()
                self.render()
            self.actions.step(action)
            self.observation, _, self.done, _ = self.env.step(action)

    def error(self):
        self.env.render()
        self.render()
        print("===" * 10, "DEBUG", "===" * 10)
        print("Env Step: ", self.env.current_step)
        print(
            "Env Position: ", self.env.position, "| Backtest Position: ", self.position
        )
        print(
            "Env Price: ",
            self.env.current_price,
            "| Backtest Price: ",
            self.data.Close[-1],
        )
        print(
            "Env Price with fee: ",
            self.env._adjusted_price(1),
            "| Backtest Price with fee: ",
            self._broker._adjusted_price(1),
        )
        print("Env Equity: ", self.env.equity, "| Backtest Equity: ", self.equity)
        print("Env Assets: ", self.env.cash, "| Backtest Cash: ", self._broker._cash)

        print("Env Trades: ", self.env.trades, "| Backtest Trades", self.trades)
        print("Env Position: ", self.env.position, "| Backtest Postion", self.position)
        print("===" * 10, "=====", "===" * 10)
        return "See Debug Above"

    def render(self):
        print("===" * 5, f"Backtesting ({self.data.index[-1]})", "===" * 5)
        print(f"Price: {self.data.Close[-1]}")
        print(f"Cash: {self._broker._cash}")
        print(f"Equity: {self.equity}")
        print(f"Orders: {self.orders}")
        print(f"Trades: {self.trades}")
        print(f"Position: {self.position}")
        print(f"Closed Trades: {self.closed_trades}")

    @property
    def current_price(self):
        return self.env.current_price

    @property
    def current_step(self):
        return self.env.current_step


def backtest(
    env,
    agent="Random",
    save_dir: str = "./backtest-stats",
    plot: bool = True,
    open_browser: bool = True,
    debug: bool = True,
) -> pd.DataFrame:

    bt = Backtest(
        env.observer._ohlcv,
        DRLStrategy,
        cash=env.initial_cash,
        commission=env.fee,
        trade_on_close=True,
        exclusive_orders=True,
    )
    stats = bt.run(env=env, agent=agent, debug=debug)
    print(stats)
    if plot:
        bt.plot(open_browser=False, filename=save_dir)
