


class MarginAccountSimulator:
    def __init__(self):
        self.original_wallet = self.wallet = 1
        self.wallet_timeline = []
        self.open_buy_orders = []
        self.open_trades = []
        self.market_actions = []
        self.n_trades = 0

    def update(self, i, price):
        self.wallet_timeline.append(self.calculate_investment_value(price))

    def get_open_buy_orders(self):
        return self.open_buy_orders

    def get_open_trades(self):
        return self.open_trades

    def long(self, price):
        pass

    def short(self, price):
        pass

    def execute_market_buy(self, i, price, qty=1, sell_limit=None, stop_loss=None):
        # limited functionality: one open trade at a time
        # limited functionality: buy with all available money
        assert len(self.open_trades) == 0
        # https://www.binance.com/en/fee/trading
        # self.wallet *= 1 - (0.001 * 2)  # buy and sell commission combined
        self.wallet -= self.wallet * (0.001 * 2)  # buy and sell commission combined
        self.open_trades.append({'buy_price': price,
                                 'qty': qty,
                                 'sell_limit': sell_limit,
                                 'stop_loss': stop_loss})
        assert len(self.open_trades) == 1
        self.market_actions.append(i)
        self.n_trades += 1
        # print(f"buy @ {price:.4}")

    def execute_market_sell(self, i, price):
        # limited functionality: sell all (one) open trades
        assert len(self.open_trades) == 1
        for trade in self.open_trades:
            self.wallet *= price / trade['buy_price']
            self.open_trades.remove(trade)
        assert not self.open_trades  # check if self.open_trades is empty
        self.market_actions.append(i)
        # print(f"sell @ {price:.4}")
        # print(f"wallet {self.wallet:.4}")

    def execute_limit_orders(self, price):
        raise NotImplementedError
        for buy_order in self.open_buy_orders:
            if buy_order['buy_limit']:
                raise NotImplementedError
        for trade in self.open_trades:
            if trade['sell_limit'] is not None:
                if trade['sell_limit'] <= price:
                    self.wallet += (trade['sell_limit'] - trade['buy_price']) * trade['qty']
                    self.open_trades.remove(trade)
                    continue
            if trade['stop_loss'] is not None:
                if price <= trade['stop_loss']:
                    self.wallet += (trade['stop_loss'] - trade['buy_price']) * trade['qty']
                    self.open_trades.remove(trade)
                    continue

    def calculate_investment_value(self, price):
        assert len(self.open_trades) <= 1
        temp_wallet = self.wallet
        for trade in self.open_trades:
            temp_wallet *= price / trade['buy_price']
        return temp_wallet

    def get_wallet(self):
        return self.wallet_timeline[-1]

    def get_n_trades(self):
        return self.n_trades