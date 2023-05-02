import sys
import warnings
from math import copysign
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np
from rl_trading.envs.components.action import ActionScheme
from rl_trading.envs.components.informe import InformerScheme
from rl_trading.envs.components.observe import ObserverScheme
from rl_trading.envs.components.reward import RewardScheme
from rl_trading.envs.components.stop import StopperScheme
from rl_trading.envs.core import Order, Position, Trade


class MarketEnv(gym.Env):
    def __init__(
        self,
        actions: ActionScheme,
        observer: ObserverScheme,
        rewards: RewardScheme,
        stopper: StopperScheme,
        informer: InformerScheme,
        *,
        window_size: int,
        fee: float,
    ):
        self.actions = actions
        self.rewards = rewards
        self.observer = observer
        self.informer = informer
        self.stopper = stopper
        self.window_size = window_size
        self.fee = fee

        self.current_step = 0
        self.initial_cash = 100000
        self.cash = self.initial_cash
        self.position: Optional[Position] = Position(self)
        self.orders: List[Order] = []
        self.trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = []

        self.actions.reset(self)
        self.observer.reset(self)
        self.action_space = self.actions.action_space
        self.observation_space = self.observer.observation_space

    def reset(self) -> np.ndarray:
        self.current_step = self.window_size
        self.action = None
        self.cash = self.initial_cash

        self.position = Position(self)
        self.orders = []
        self.trades = []
        self.equity_curve = [self.equity]
        self.closed_trades = []

        self.actions.reset(self)
        self.obs = self.observer.reset(self)
        self.reward = self.rewards.reset(self)
        self.done = self.stopper.reset(self)
        self.info = self.informer.reset(self)

        return self.obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.action = action
        self.actions.step(self.action)

        self._process_orders()  # step += 1

        self.obs = self.observer.step()
        self.reward = self.rewards.step()
        self.done = self.stopper.step()
        self.info = self.informer.step()
        return self.obs, self.reward, self.done, self.info

    def render(self, mode="human"):
        print("===" * 5, f"Environment ({self.current_step})", "===" * 5)
        print(f"Price: {self.observer.price}")
        print(f"Cash: {self.cash}")
        print(f"Equity: {self.equity}")
        print(f"Orders: {self.orders}")
        print(f"Trades: {self.trades}")
        print(f"Position: {self.position}")
        print(f"Closed Trades: {self.closed_trades}")
        print(f"Action: {self.action}, Reward: {self.reward}, Done: {self.done}\n")

    @property
    def tech_indicators(self):
        return self.features.columns.tolist()

    @property
    def equity(self) -> float:
        return self.cash + sum(trade.pnl for trade in self.trades)

    @property
    def margin_available(self) -> float:
        margin_used = sum(trade.value for trade in self.trades)
        return max(0, self.equity - margin_used)

    class __FULL_EQUITY(float):
        def __repr__(self):
            return ".9999"

    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(
        self,
        size: Optional[float] = _FULL_EQUITY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sl: float = None,
        tp: float = None,
    ):
        assert 0 < size < 1 or round(size) == size, f"size: {size}"
        return self.new_order(size, limit_price, stop_price, sl, tp)

    def sell(
        self,
        size: Optional[float] = _FULL_EQUITY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        sl: float = None,
        tp: float = None,
    ):
        assert 0 < size < 1 or round(size) == size, f"size: {size}"
        return self.new_order(-size, limit_price, stop_price, sl, tp)

    def new_order(
        self,
        size: float,
        limit: float = None,
        stop: float = None,
        sl: float = None,
        tp: float = None,
        *,
        trade: Trade = None,
    ):
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        adjusted_price = self._adjusted_price(size)

        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError(
                    "Long orders require: "
                    f"SL ({sl}) < LIMIT ({limit or stop or adjusted_price}) < TP ({tp})"
                )
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError(
                    "Short orders require: "
                    f"TP ({tp}) < LIMIT ({limit or stop or adjusted_price}) < SL ({sl})"
                )

        order = Order(self, size, limit, stop, sl, tp, trade)
        if trade:
            self.orders.insert(0, order)
        else:
            self.orders.append(order)

        return order

    def _adjusted_price(self, size: int, price: Optional[float] = None) -> float:
        return (price or self.observer.price) * (1 + copysign(self.fee, size))

    def _process_orders(self):
        self.current_step += 1
        self.equity_curve.append(self.equity)
        data = self.observer.candlestick
        open, high, low = (
            data[0],
            data[1],
            data[2],
        )
        prev_close = self.observer.prev_candlestick[3]
        reprocess_orders = False

        # Process orders
        for order in list(self.orders):  # type: Order
            # Related SL/TP order was already removed
            if order not in self.orders:
                continue

            # Check if stop condition was hit
            stop_price = order.stop
            if stop_price:
                is_stop_hit = (
                    (high > stop_price) if order.is_long else (low < stop_price)
                )
                if not is_stop_hit:
                    continue

                # > When the stop price is reached, a stop order becomes a market/limit order.
                # https://www.sec.gov/fast-answers/answersstopordhtm.html
                order._replace(stop_price=None)

            # Determine purchase price.
            # Check if limit order can be filled.
            if order.limit:
                is_limit_hit = (
                    low < order.limit if order.is_long else high > order.limit
                )
                # When stop and limit are hit within the same bar, we pessimistically
                # assume limit was hit before the stop (i.e. "before it counts")
                is_limit_hit_before_stop = is_limit_hit and (
                    order.limit < (stop_price or -np.inf)
                    if order.is_long
                    else order.limit > (stop_price or np.inf)
                )
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                # stop_price, if set, was hit within this bar
                price = (
                    min(stop_price or open, order.limit)
                    if order.is_long
                    else max(stop_price or open, order.limit)
                )
            else:
                # Market-if-touched / market order
                price = prev_close
                price = (
                    max(price, stop_price or -np.inf)
                    if order.is_long
                    else min(price, stop_price or np.inf)
                )

            # Determine entry/exit bar index
            is_market_order = not order.limit and not stop_price
            time_index = (
                (self.current_step - 1) if is_market_order else self.current_step
            )

            # If order is a SL/TP order, it should close an existing trade it was contingent upon
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                # If order.size is "greater" than trade.size, this order is a trade.close()
                # order and part of the trade was already closed beforehand
                size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
                # If this trade isn't already closed (e.g. on multiple `trade.close(.5)` calls)
                if trade in self.trades:
                    self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders  # Removed when trade was closed
                else:
                    # It's a trade.close() order, now done
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            # Else this is a stand-alone trade

            # Adjust price to include commission (or bid-ask spread).
            # In long positions, the adjusted price is a fraction higher, and vice versa.
            adjusted_price = self._adjusted_price(order.size, price)

            # If order size was specified proportionally,
            # precompute true size in units, accounting for margin and spread/commissions
            size = order.size
            if -1 < size < 1:
                size = copysign(
                    int((self.margin_available * abs(size)) // adjusted_price),
                    size,
                )
                # Not enough cash/margin even for a single unit
                if not size:
                    self.orders.remove(order)
                    continue
            assert size == round(size)
            need_size = int(size)

            # Fill position by FIFO closing/reducing existing opposite-facing trades.
            # Existing trades are closed at unadjusted price, because the adjustment
            # was already made when buying.
            for trade in list(self.trades):
                if trade.is_long == order.is_long:
                    continue
                assert trade.size * order.size < 0

                # Order size greater than this opposite-directed existing trade,
                # so it will be closed completely
                if abs(need_size) >= abs(trade.size):
                    self._close_trade(trade, price, time_index)
                    need_size += trade.size
                else:
                    # The existing trade is larger than the new order,
                    # so it will only be closed partially
                    self._reduce_trade(trade, price, need_size, time_index)
                    need_size = 0

                if not need_size:
                    break

            # If we don't have enough liquidity to cover for the order, cancel it
            if abs(need_size) * adjusted_price > self.margin_available:
                self.orders.remove(order)
                continue

            # Open a new trade
            if need_size:
                self._open_trade(
                    adjusted_price, need_size, order.sl, order.tp, time_index
                )

                # We need to reprocess the SL/TP orders newly added to the queue.
                # This allows e.g. SL hitting in the same bar the order was open.
                # See https://github.com/kernc/backtesting.py/issues/119
                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif (
                        low <= (order.sl or -np.inf) <= high
                        or low <= (order.tp or -np.inf) <= high
                    ):
                        print(data.index[-1])
                        warnings.warn(
                            f"({data.index[-1]}) A contingent SL/TP order would execute in the "
                            "same bar its parent stop/limit order was turned into a trade. "
                            "Since we can't assert the precise intra-candle "
                            "price movement, the affected SL/TP order will instead be executed on "
                            "the next (matching) price/bar, making the result (of this trade) "
                            "somewhat dubious. "
                            "See https://github.com/kernc/backtesting.py/issues/119",
                            UserWarning,
                        )

            # Order processed
            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()

    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int):
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            # Reduce existing trade ...
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            # ... by closing a reduced copy of it
            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            self.trades.append(close_trade)

        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade: Trade, price: float, time_index: int):
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        self.closed_trades.append(trade._replace(exit_price=price, exit_bar=time_index))
        self.cash += trade.pnl

    def _open_trade(
        self, price: float, size: int, sl: float, tp: float, time_index: int
    ):
        trade = Trade(self, size, price, time_index)
        self.trades.append(trade)
        if tp:
            trade.tp = tp
        if sl:
            trade.sl = sl
