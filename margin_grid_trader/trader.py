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
        self.quantity = value / price
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

    def __init__(self, initial_price, step, order_value=1):
        self.grid_manager = GridManager(initial_price, step)

        self.order_value = order_value  # value of a single long or short trade in base currency

        # order pair is a buy-sell (long) or a sell-buy (short) order pair
        # if it is in the list the first order is already done and second is not
        self.open_long_order_pairs = []
        self.open_short_order_pairs = []

        # place initial order pairs
        op_long = OrderPair('l', 0)
        op_short = OrderPair('s', 0)
        pass

        self.upper_action_grid_idx = 1
        self.lower_action_grid_idx = -1
        self.upper_action_price = self.grid_manager[self.upper_action_grid_idx]
        self.lower_action_price = self.grid_manager[self.lower_action_grid_idx]

        self.grid_values_timeline = []

    def get_long_trade_size(self, price):
        return price / self.unit_value

    def get_short_trade_size(self, price):
        return price / self.unit_value

    def get_order_pair_to_close(self, grid_idx: int) -> OrderPair:
        """
        There must be one and only one order pair to be closed
        """
        open_order_pair_to_close = None
        count = 0
        for open_order_pair in self.open_order_pairs:
            if open_order_pair.close_grid_idx:
                open_order_pair_to_close = open_order_pair
                count += 1
        if count == 0:
            raise ValueError("No order pair to close found")
        elif count > 1:
            raise ValueError("More than one order pair to close found")

        return open_order_pair_to_close

    def execute_combined_order_pairs(self, op_long_to_close: OrderPair, op_short_to_close, op_to_open: OrderPair):
        assert (op_long_to_close is not None) or (op_short_to_close is not None)
        if op_to_open.pos_type == 'l':
            if op_long_to_close is not None:
                assert op_long_to_close.pos_type == 'l'
                assert op_long_to_close.close_grid_idx == op_to_open.open_grid_idx
                assert op_long_to_close.quantity > op_to_open.quantity
                # TODO sell op_long_to_close.quantity - op_to_open.quantity
            else:
                pass
                # TODO buy op_to_open.quantity
            if op_short_to_close is not None:
                pass
                # TODO sell op_short_to_close.quantity

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

        # TODO put in function with extra check for duplicates
        for open_long_order_pair in self.open_long_order_pairs:
            if open_long_order_pair == op_long:
                op_long = None
        for open_short_order_pair in self.open_short_order_pairs:
            if open_short_order_pair == op_short:
                op_short = None

        op_to_close = self.get_order_pair_to_close(grid_idx)

        if op_to_close.pos_type == 'l':
            self.execute_combined_long_order_pairs(op_to_close, op_long)
        else:
            self.execute_combined_short_order_pairs(op_to_close, op_short)

        # delete and insert order pairs

    def update(self, price):
        """
        Anatomy of an update
        - check if price has reached an action price
        - do grid action
        - update action prices
        :param price:
        :return:
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

        #####

        self.upper_action_grid_idx = 1
        self.do_grid_action(self.upper_action_grid_idx)

        if price >= self.upper_action_price:
            #
            self.curr_grid_value = self.upper_grid_value
            self.grid_values_timeline.append(self.curr_grid_value)
            self.upper_grid_value = self.curr_grid_value * self.step
            self.lower_grid_value = self.curr_grid_value / self.step
            return "up"

        elif price <= self.lower_action_price:
            self.curr_grid_value = self.lower_grid_value
            self.grid_values_timeline.append(self.curr_grid_value)
            self.upper_grid_value = self.curr_grid_value * self.step
            self.lower_grid_value = self.curr_grid_value / self.step
            return "down"

        ######################

        if self.status == "uninitialized":
            # Open initial trades
            self.curr_grid_value = price
            self.grid_values_timeline.append(self.curr_grid_value)
            self.upper_action_price = self.curr_grid_value * self.step
            self.lower_action_price = self.curr_grid_value / self.step
            return "init_trade"



        else:
            return "do_nothing"
