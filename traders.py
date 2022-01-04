def get_lower_grid_value(price, curr_lower_grid_value, step):
    # multiply and divide multiple times with same number gives same results? according to my tests: yes
    if price < curr_lower_grid_value:
        for _ in range(100):
            curr_lower_grid_value /= step
            if curr_lower_grid_value <= price:
                return curr_lower_grid_value
        raise Exception('Too large price change compared to grid step size')
    elif curr_lower_grid_value <= price < curr_lower_grid_value * step:
        return curr_lower_grid_value
    else:
        for _ in range(100):
            curr_lower_grid_value *= step
            if price < curr_lower_grid_value * step:
                return curr_lower_grid_value
        raise Exception('Too large price change compared to grid step size')


class GridTrader:
    def __init__(self, step, stop_loss_coef):
        self.status = "uninitialized"
        self.allow_trade = True  # controls instant rebuy after a stop-loss sell

        self.step = 1 + step
        self.stop_loss_coef = 1 - stop_loss_coef

        self.curr_lower_grid_value = None
        self.lower_grid_values_timeline = []

        self.stop_loss = None  # stop-loss price of the only one open trade

    def update(self, price):
        if self.status == "uninitialized":
            self.status = "out_of_trade"
            self.curr_lower_grid_value = price * (1 + ((self.step - 1) / 10))
            self.lower_grid_values_timeline.append(self.curr_lower_grid_value)
            return "do_nothing"

        self.curr_lower_grid_value = get_lower_grid_value(price, self.curr_lower_grid_value, self.step)

        if self.lower_grid_values_timeline[-1] != self.curr_lower_grid_value:
            self.lower_grid_values_timeline.append(self.curr_lower_grid_value)
            self.allow_trade = True

        if self.status == "out_of_trade":
            if len(self.lower_grid_values_timeline) < 3:
                return "do_nothing"
            elif self.lower_grid_values_timeline[-3] < self.lower_grid_values_timeline[-2] < \
                    self.lower_grid_values_timeline[-1] and self.allow_trade:
                self.status = "in_trade"
                self.stop_loss = price * self.stop_loss_coef
                return "buy"
            else:
                return "do_nothing"
        elif self.status == "in_trade":
            self.stop_loss = max(self.stop_loss, price * self.stop_loss_coef)
            if price <= self.stop_loss:
                self.status = "out_of_trade"
                self.allow_trade = False
                self.stop_loss = None
                return "sell"
            else:
                return "do_nothing"
        else:
            raise ValueError


class LimitTrader:
    def __init__(self, buy_lim_coef, stop_loss_coef):
        self.status = "uninitialized"

        self.buy_lim_coef = 1 + buy_lim_coef
        self.stop_loss_coef = 1 - stop_loss_coef

        self.buy_lim = None
        self.stop_loss = None

    def update(self, price):
        if self.status == "uninitialized":
            self.status = "out_of_trade"
            self.buy_lim = price * self.buy_lim_coef
            return "do_nothing"
        elif self.status == "out_of_trade":
            self.buy_lim = min(self.buy_lim, price * self.buy_lim_coef)
            if price >= self.buy_lim:
                self.status = "in_trade"
                self.buy_lim = None
                self.stop_loss = price * self.stop_loss_coef
                return "buy"
            else:
                return "do_nothing"
        elif self.status == "in_trade":
            self.stop_loss = max(self.stop_loss, price * self.stop_loss_coef)
            if price <= self.stop_loss:
                self.status = "out_of_trade"
                self.buy_lim = price * self.buy_lim_coef
                self.stop_loss = None
                return "sell"
            else:
                return "do_nothing"
        else:
            raise ValueError
