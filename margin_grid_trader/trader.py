import numpy as np
from typing import Tuple
import pandas as pd

from .simulation import MarginAccountSimulator


class GridManager:
    def __init__(self, initial_price, step):
        assert 0 < step < 0.2
        self.initial_price = initial_price
        self.q = 1 + step

    def get_price_at_idx(self, idx):
        assert type(idx) is int
        if idx >= 0:
            return self.initial_price * (self.q ** idx)
        else:
            return self.initial_price * ((1 / self.q) ** -idx)

    def __getitem__(self, key):
        self.get_price_at_idx(key)


class OrderPair:
    def __init__(self, pos_type, price, value, open_grid_idx):
        assert pos_type is ('l', 's')
        self.pos_type = pos_type
        self.price = price
        self.value = value
        self.amount = value / price
        self.open_grid_idx = open_grid_idx
        if pos_type == 'l':
            self.close_grid_idx = open_grid_idx + 1
        else:
            self.close_grid_idx = open_grid_idx - 1

    def __eq__(self, other):
        assert type(other) is OrderPair
        return self.pos_type == other.pos_type and self.open_grid_idx == other.open_grid_idx


class LongShortTrader:
    """
    Opens a long and a short position at every grid
    """

    def __init__(self, acc: MarginAccountSimulator, initial_price, step, order_value=1):
        self.acc = acc
        self.grid_manager = GridManager(initial_price, step)

        self.order_value = order_value  # value of a single long or short trade in base currency

        # order pair is a buy-sell (long) or a sell-buy (short) order pair
        # if it is in the list the first order is already done and second is not
        self.open_order_pairs = []

        # place initial order pairs
        op_long = OrderPair('l', price=initial_price, value=self.order_value, open_grid_idx=0)
        op_short = OrderPair('s', price=initial_price, value=self.order_value, open_grid_idx=0)
        self.acc.long_borrow_and_buy(value=op_long.value)
        self.acc.short_borrow_and_sell(amount=op_short.amount)
        self.open_order_pairs.append(op_long)
        self.open_order_pairs.append(op_short)

        self.upper_action_grid_idx = 1
        self.lower_action_grid_idx = -1
        self.upper_action_price = self.grid_manager[self.upper_action_grid_idx]
        self.lower_action_price = self.grid_manager[self.lower_action_grid_idx]

        self.grid_values_timeline = []

    @staticmethod
    def get_random_trader_population(n_traders, step_lo, step_hi):
        trader_params = pd.DataFrame(columns=('step', 'profit', 'n_trades'))
        for i in range(n_traders):
            step = np.random.uniform(step_lo, step_hi)
            trader = {
                'step': step,
                'wallet': np.nan,
                'n_trades': np.nan
            }
            trader_params = trader_params.append(trader, ignore_index=True)
        return trader_params

    def check_order_pair_already_open(self, op_long: OrderPair, op_short: OrderPair) -> Tuple[OrderPair, OrderPair]:
        for open_order_pair in self.open_order_pairs:
            if open_order_pair == op_long:
                assert op_long is not None  # check for duplicates
                op_long = None
            if open_order_pair == op_short:
                assert op_short is not None  # check for duplicates
                op_short = None
        return op_long, op_short

    def get_order_pair_to_close(self, grid_idx: int) -> OrderPair:
        """
        There must be one and only one order pair to be closed
        """
        open_order_pair_to_close = None
        len_orig = len(self.open_order_pairs)
        count = 0
        for open_order_pair in self.open_order_pairs:
            if open_order_pair.close_grid_idx == grid_idx:
                open_order_pair_to_close = open_order_pair
                self.open_order_pairs.remove(open_order_pair)
                count += 1
        if count == 0:
            raise ValueError("No order pair to close found")
        elif count > 1:
            raise ValueError("More than one order pair to close found")
        assert len_orig - len(self.open_order_pairs) == 1

        return open_order_pair_to_close

    def execute_combined_order_pairs(self, op_long_to_open: OrderPair, op_short_to_open: OrderPair,
                                     op_to_close: OrderPair):
        assert (op_long_to_open is not None) or (op_short_to_open is not None)
        if op_to_close.pos_type == 'l':
            if op_long_to_open is not None:
                assert op_long_to_open.pos_type == 'l'
                assert op_long_to_open.open_grid_idx == op_to_close.close_grid_idx
                assert op_to_close.amount > op_long_to_open.amount
                self.acc.long_sell_and_repay(amount=op_to_close.amount - op_long_to_open.amount)
            else:
                self.acc.long_sell_and_repay(amount=op_to_close.amount)
            if op_short_to_open is not None:
                self.acc.short_borrow_and_sell(amount=op_short_to_open.amount)
        else:
            if op_short_to_open is not None:
                assert op_short_to_open.pos_type == 's'
                assert op_short_to_open.open_grid_idx == op_to_close.close_grid_idx
                assert op_to_close.amount < op_short_to_open.amount
                self.acc.short_buy_and_repay(amount=op_short_to_open.amount - op_to_close.amount)
            else:
                self.acc.short_buy_and_repay(amount=op_to_close.amount)
            if op_long_to_open is not None:
                self.acc.long_borrow_and_buy(amount=op_long_to_open.amount)

    def do_grid_action(self, price, grid_idx):
        """
        Anatomy of a grid action:
        - create the two order_pairs (long and short)
        - check which order pairs are already opened at that grid and delete that
        - check which open order pairs must be closed
        - combine orders if possible
        """
        op_long = OrderPair(pos_type='l', price=price, value=self.order_value, open_grid_idx=grid_idx)
        op_short = OrderPair(pos_type='s', price=price, value=self.order_value, open_grid_idx=grid_idx)

        op_long, op_short = self.check_order_pair_already_open(op_long, op_short)
        if op_long is not None:  # TODO this may can go into check_order_pair_already_open
            self.open_order_pairs.append(op_long)
        if op_short is not None:
            self.open_order_pairs.append(op_short)

        op_to_close = self.get_order_pair_to_close(grid_idx)

        self.execute_combined_order_pairs(op_long, op_short, op_to_close)

    def update(self, price):
        """
        Anatomy of an update
        - check if price has reached an action price
        - do grid action
        - update action prices
        """
        if price >= self.upper_action_price:
            self.do_grid_action(self.upper_action_grid_idx)
            self.upper_action_grid_idx += 1
            self.lower_action_grid_idx += 1
            self.upper_action_price = self.grid_manager[self.upper_action_grid_idx]
            self.lower_action_price = self.grid_manager[self.lower_action_grid_idx]

        elif price <= self.lower_action_price:
            self.do_grid_action(self.lower_action_grid_idx)
            self.upper_action_grid_idx -= 1
            self.lower_action_grid_idx -= 1
            self.upper_action_price = self.grid_manager[self.upper_action_grid_idx]
            self.lower_action_price = self.grid_manager[self.lower_action_grid_idx]
