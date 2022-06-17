from copy import copy
from math import copysign
from typing import Optional, Union

import numpy as np
import pandas as pd


class Position:
    def __init__(self, env):
        self.__env = env

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        return sum(trade.size for trade in self.__env.trades)

    @property
    def pnl(self) -> float:
        return sum(trade.pnl for trade in self.__env.trades)

    @property
    def pnl_pct(self) -> float:
        weights = np.abs([trade.size for trade in self.__env.trades])
        weights = weights / weights.sum()
        pnl_pcts = np.array([trade.pnl_pct for trade in self.__env.trades])
        return (pnl_pcts * weights).sum()

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    def close(self, portion: float = 1.0):
        for trade in self.__env.trades:
            trade.close(portion)

    def __repr__(self):
        return f"<Position: {self.size} ({len(self.__env.trades)} trades)>"


class Order:
    def __init__(
        self,
        broker: "TradingEnv",
        size: float,
        limit_price: float = None,
        stop_price: float = None,
        sl_price: float = None,
        tp_price: float = None,
        parent_trade: "Trade" = None,
    ):
        self.__broker = broker
        assert size != 0
        self.__size = size
        self.__limit_price = limit_price
        self.__stop_price = stop_price
        self.__sl_price = sl_price
        self.__tp_price = tp_price
        self.__parent_trade = parent_trade

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f"_{self.__class__.__qualname__}__{k}", v)
        return self

    def __repr__(self):
        return "<Order {}>".format(
            ", ".join(
                f"{param}={round(value, 5)}"
                for param, value in (
                    ("size", self.__size),
                    ("limit", self.__limit_price),
                    ("stop", self.__stop_price),
                    ("sl", self.__sl_price),
                    ("tp", self.__tp_price),
                    ("contingent", self.is_contingent),
                )
                if value is not None
            )
        )

    def cancel(self):
        """Cancel the order."""
        self.__broker.orders.remove(self)
        trade = self.__parent_trade
        if trade:
            if self is trade._sl_order:
                trade._replace(sl_order=None)
            elif self is trade._tp_order:
                trade._replace(tp_order=None)
            else:
                # XXX: https://github.com/kernc/backtesting.py/issues/251#issuecomment-835634984 ???
                assert False

    # Fields getters

    @property
    def size(self) -> float:
        return self.__size

    @property
    def limit(self) -> Optional[float]:
        return self.__limit_price

    @property
    def stop(self) -> Optional[float]:
        return self.__stop_price

    @property
    def sl(self) -> Optional[float]:
        return self.__sl_price

    @property
    def tp(self) -> Optional[float]:
        return self.__tp_price

    @property
    def parent_trade(self):
        return self.__parent_trade

    # Extra properties

    @property
    def is_long(self):
        return self.__size > 0

    @property
    def is_short(self):
        return self.__size < 0

    @property
    def is_contingent(self):
        return bool(self.__parent_trade)


class Trade:
    def __init__(self, broker: "TradingEnv", size: int, entry_price: float, entry_bar):
        self.__env = broker
        self.__size = size
        self.__entry_price = entry_price
        self.__exit_price: Optional[float] = None
        self.__entry_bar: int = entry_bar
        self.__exit_bar: Optional[int] = None
        self.__sl_order: Optional[Order] = None
        self.__tp_order: Optional[Order] = None

    def __repr__(self):
        return (
            f'<Trade size={self.__size} time={self.__entry_bar}-{self.__exit_bar or ""} '
            f'price={self.__entry_price}-{self.__exit_price or ""} pnl={self.pnl:.0f}>'
        )

    def _replace(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, f"_{self.__class__.__qualname__}__{k}", v)
        return self

    def _copy(self, **kwargs):
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1.0):
        """Place new `Order` to close `portion` of the trade at next market price."""
        assert 0 < portion <= 1, "portion must be a fraction between 0 and 1"
        size = copysign(max(1, round(abs(self.__size) * portion)), -self.__size)
        order = Order(self.__env, size, parent_trade=self)
        self.__env.orders.insert(0, order)

    # Fields getters

    @property
    def size(self):
        return self.__size

    @property
    def entry_price(self) -> float:
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        return self.__exit_price

    @property
    def entry_bar(self) -> int:
        return self.__entry_bar

    @property
    def exit_bar(self) -> Optional[int]:
        return self.__exit_bar

    @property
    def _sl_order(self):
        return self.__sl_order

    @property
    def _tp_order(self):
        return self.__tp_order

    # Extra properties

    @property
    def entry_time(self) -> Union[pd.Timestamp, int]:
        return self.__env.observer.datetime[self.__entry_bar]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        if self.__exit_bar is None:
            return None
        return self.__env.observer.datetime[self.__exit_bar]

    @property
    def is_long(self):
        return self.__size > 0

    @property
    def is_short(self):
        return not self.is_long

    @property
    def pnl(self):
        price = self.__exit_price or self.__env.observer.price
        return self.__size * (price - self.__entry_price)

    @property
    def pnl_pct(self):
        price = self.__exit_price or self.__env.observer.price
        return copysign(1, self.__size) * (price / self.__entry_price - 1)

    @property
    def value(self):
        price = self.__exit_price or self.__env.observer.price
        return abs(self.__size) * price

    # SL/TP management API

    @property
    def sl(self):
        """
        Stop-loss price at which to close the trade.
        This variable is writable. By assigning it a new price value,
        you create or modify the existing SL order.
        By assigning it `None`, you cancel it.
        """
        return self.__sl_order and self.__sl_order.stop

    @sl.setter
    def sl(self, price: float):
        self.__set_contingent("sl", price)

    @property
    def tp(self):
        """
        Take-profit price at which to close the trade.
        This property is writable. By assigning it a new price value,
        you create or modify the existing TP order.
        By assigning it `None`, you cancel it.
        """
        return self.__tp_order and self.__tp_order.limit

    @tp.setter
    def tp(self, price: float):
        self.__set_contingent("tp", price)

    def __set_contingent(self, type, price):
        assert type in ("sl", "tp")
        assert price is None or 0 < price < np.inf
        attr = f"_{self.__class__.__qualname__}__{type}_order"
        order: Order = getattr(self, attr)
        if order:
            order.cancel()
        if price:
            kwargs = dict(stop=price) if type == "sl" else dict(limit=price)
            order = self.__env.new_order(-self.size, trade=self, **kwargs)
            setattr(self, attr, order)
