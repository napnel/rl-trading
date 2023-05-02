from typing import TYPE_CHECKING

import numpy as np
from gym import spaces

if TYPE_CHECKING:
    from rl_trading.envs.environment import TradingEnv


class ActionScheme:
    def __init__(self, **kwargs):
        self.env: "TradingEnv" = None

    def reset(self, env: "TradingEnv"):
        self.env = env

    def step(self, action):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError


class OneWayLimitOrder(ActionScheme):
    def __init__(self, atr: np.ndarray, **kwargs) -> None:
        super().__init__()
        self.atr = atr

    @property
    def action_space(self) -> spaces.Box:
        # self._action_space = spaces.Box(
        #     low=np.float32(np.array([0, -1])),
        #     high=np.float32(np.array([1, 1])),
        #     dtype=np.float32,
        # )
        self._action_space = spaces.Box(
            low=np.float32(-1),
            high=np.float32(1),
            dtype=np.float32,
            shape=(1,),
        )
        return self._action_space

    def step(self, action: float):
        [order.cancel() for order in self.env.orders]
        atr_range = action
        # bet_prob = None
        # bet_prob, atr_range = action[0], action[1]
        limit_price = (
            self.env.current_price + self.atr[self.env.current_step] * atr_range
        )

        if self.env.position.size != 0:
            self.env.position.close()

        # if bet_prob == 0:
        #     pass

        elif atr_range > 0:
            self.env.buy(limit=limit_price)

        elif atr_range < 0:
            self.env.sell(limit=limit_price)


class OneWayLimitOrderV2(ActionScheme):
    def __init__(self, atr: np.ndarray, **kwargs) -> None:
        super().__init__()
        self.atr = atr

    @property
    def action_space(self) -> spaces.Box:
        self._action_space = spaces.Box(
            low=np.float32(0.5),
            high=np.float32(5),
            dtype=np.float32,
            shape=(1,),
        )
        return self._action_space

    def step(self, action: float):
        [order.cancel() for order in self.env.orders]
        position = self.env.position

        atr = self.atr[self.env.current_step]
        limit_price = self.env.current_price - atr * action
        # print(self.action_space)
        # print(action, atr)
        # print(limit_price, self.env.current_price)
        if position.size != 0:
            self.env.position.close()

        if limit_price < self.env.current_price:
            if not position.is_long:
                self.env.position.close()
                self.env.buy(size=0.75, limit=limit_price)

        elif limit_price > self.env.current_price:
            if not position.is_short:
                self.env.position.close()
                self.env.sell(size=0.75, limit=limit_price)


class OneWayLimitOrderV3(ActionScheme):
    def __init__(self, atr: np.ndarray, **kwargs) -> None:
        super().__init__()
        self.atr = atr
        self.ratio = np.arange(-2.5, 3.0, 0.5)

    @property
    def action_space(self) -> spaces.Box:
        self._action_space = spaces.Discrete(11)
        return self._action_space

    def step(self, action: float):
        [order.cancel() for order in self.env.orders]
        # if self.env.position.size != 0:
        #     self.env.position.close()
        atr = self.atr[self.env.current_step]
        price = self.env.current_price
        limit_price = price - atr * self.ratio[action]

        if limit_price == price:
            self.env.position.close()

        if limit_price < price:
            self.env.buy(size=0.8, limit=limit_price)

        elif limit_price > price:
            self.env.sell(size=0.8, limit=limit_price)


class MarketOrder(ActionScheme):
    @property
    def action_space(self) -> spaces.Box:
        self._action_space = spaces.Discrete(2)
        return self._action_space

    def step(self, action: np.ndarray):
        position = self.env.position
        # if self.env.position.size != 0:
        #     self.env.position.close()

        if action == 0:
            if not position.is_long:
                self.env.position.close()
                self.env.buy(size=0.75)

        elif action == 1:
            if not position.is_short:
                self.env.position.close()
                self.env.sell(size=0.75)

        else:
            raise ValueError


class LiskedMarketOrder(ActionScheme):
    @property
    def action_space(self) -> spaces.Box:
        self._action_space = spaces.Discrete(5)
        return self._action_space

    def step(self, action: np.ndarray):
        position = self.env.position
        # if self.env.position.size != 0:
        #     self.env.position.close()

        if action == 0:
            if not position.is_long:
                self.env.position.close()
                self.env.buy(size=0.75)

        elif action == 1:
            if not position.is_short:
                self.env.position.close()
                self.env.sell(size=0.75)

        elif action == 2:
            pass

        elif action == 3:
            pass

        elif action == 4:
            pass

        elif action == 5:
            pass

        else:
            raise ValueError


registry = {
    "MarketOrder": MarketOrder,
    # "LimitOrder": OneWayLimitOrderV2,
}


def get(class_name: str):
    if class_name not in list(registry.keys()):
        raise KeyError(f"{class_name} does not include the registry")
    return registry[class_name]
