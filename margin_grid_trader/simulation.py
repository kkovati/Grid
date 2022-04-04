class MarginAccountSimulator:
    def __init__(self):
        self.wallet = 0  # TODO wallet and owned_base_currency can be the same
        self.borrowed_base_currency = 0
        self.borrowed_quote_currency = 0
        self.owned_base_currency = 0
        self.owned_quote_currency = 0

        self.commission = 0.001

        self.price = 0

        self.price_timeline = []
        self.wallet_timeline = []
        self.borrowed_value_timeline = []

        self.max_borrowed_value = 0
        self.trade_count = 0

    def convert_value_amount(self, value=None, amount=None):
        assert (value is None) != (amount is None)  # XOR
        if value is None:
            value = amount * self.price
        else:
            amount = value / self.price
        return value, amount

    def long_borrow_and_buy(self, value=None, amount=None):
        """
        Default input: value
        Price calculated input: amount
        """
        self.trade_count += 1
        value, amount = self.convert_value_amount(value, amount)
        self.borrowed_base_currency += value
        amount = value / self.price
        self.owned_quote_currency += amount
        self.wallet -= value * self.commission

    def long_sell_and_repay(self, amount=None, value=None):
        """
        Default input: amount
        Price calculated input: value
        """
        self.trade_count += 1
        value, amount = self.convert_value_amount(value, amount)
        if amount > self.owned_quote_currency:
            amount = self.owned_quote_currency
            self.owned_quote_currency = 0
        else:
            self.owned_quote_currency -= amount
        value = amount * self.price
        if value > self.borrowed_base_currency:
            self.wallet += value - self.borrowed_base_currency
            self.borrowed_base_currency = 0
        else:
            self.borrowed_base_currency -= value
        self.wallet -= value * self.commission

    def short_borrow_and_sell(self, amount=None, value=None):
        """
        Default input: amount
        Price calculated input: value
        """
        self.trade_count += 1
        value, amount = self.convert_value_amount(value, amount)
        self.borrowed_quote_currency += amount
        value = amount * self.price
        self.owned_base_currency += value
        self.wallet -= value * self.commission

    def short_buy_and_repay(self, value=None, amount=None):
        """
        Default input: value
        Price calculated input: amount
        """
        self.trade_count += 1
        value, amount = self.convert_value_amount(value, amount)
        if value > self.owned_base_currency:
            self.wallet -= value - self.owned_base_currency
            self.owned_base_currency = 0
        else:
            self.owned_base_currency -= value
        amount = value / self.price
        if amount > self.borrowed_quote_currency:
            self.owned_quote_currency += amount - self.borrowed_quote_currency
            self.borrowed_quote_currency = 0
        else:
            self.borrowed_quote_currency -= amount
        self.wallet -= value * self.commission

    def get_investment_value(self, price):
        value = 0
        value += self.wallet
        value += self.owned_base_currency
        value += self.owned_quote_currency * price
        value -= self.borrowed_base_currency
        value -= self.borrowed_quote_currency * price
        return value

    def get_borrowed_value(self, price):
        return self.borrowed_base_currency + self.borrowed_quote_currency * price

    def get_trade_count(self):
        return self.trade_count

    def update(self, i, price):
        self.price = price
        self.price_timeline.append(price)
        self.wallet_timeline.append(self.get_investment_value(price))
        self.borrowed_value_timeline.append(self.get_borrowed_value(price))

        self.max_borrowed_value = max(self.max_borrowed_value, self.get_borrowed_value())
