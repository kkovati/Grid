import numpy as np
import pandas as pd


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


class GbgsTrader:
    """
    Grid Buy Grid Sell Trader
    """

    def __init__(self, step):
        self.status = "uninitialized"
        self.allow_trade = True  # controls instant rebuy after a stop-loss sell

        self.step = 1 + step

        self.curr_lower_grid_value = None
        self.lower_grid_values_timeline = []

        self.stop_loss_grid_value = None

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
                self.stop_loss_grid_value = self.curr_lower_grid_value / self.step
                return "buy"
            else:
                return "do_nothing"
        elif self.status == "in_trade":
            if price <= self.stop_loss_grid_value:
                self.status = "out_of_trade"
                self.allow_trade = False
                self.stop_loss_grid_value = None
                return "sell"
            else:
                return "do_nothing"
        else:
            raise ValueError

    @staticmethod
    def get_random_trader_population(n_traders, step_lo, step_hi):
        trader_params = pd.DataFrame(columns=('step', 'wallet', 'n_trades'))
        for i in range(n_traders):
            step = np.random.uniform(step_lo, step_hi)
            trader = {
                'step': step,
                'wallet': np.nan,
                'n_trades': np.nan
            }
            trader_params = trader_params.append(trader, ignore_index=True)
        return trader_params


class GblsTrader:
    """
    Grid Buy Limit Sell Trader
    """

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

    @staticmethod
    def get_random_trader_population(n_traders, step_lo, step_hi, stop_loss_coef_lo, stop_loss_coef_hi):
        trader_params = pd.DataFrame(columns=('step', 'stop_loss_coef', 'wallet', 'n_trades'))
        for i in range(n_traders):
            step = np.random.uniform(step_lo, step_hi)
            stop_loss_coef = np.random.uniform(stop_loss_coef_lo, stop_loss_coef_hi)
            trader = {
                'step': step,
                'stop_loss_coef': stop_loss_coef,
                'wallet': np.nan,
                'n_trades': np.nan
            }
            trader_params = trader_params.append(trader, ignore_index=True)
        return trader_params


class LblsTrader:
    """
    Limit Buy Limit Sell Trader
    """

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

    @staticmethod
    def get_random_trader_population(n_traders, buy_lim_coef_lo, buy_lim_coef_hi, stop_loss_coef_lo, stop_loss_coef_hi):
        trader_params = pd.DataFrame(columns=('buy_lim_coef', 'stop_loss_coef', 'wallet', 'n_trades'))
        for i in range(n_traders):
            buy_lim_coef = np.random.uniform(buy_lim_coef_lo, buy_lim_coef_hi)
            stop_loss_coef = np.random.uniform(stop_loss_coef_lo, stop_loss_coef_hi)
            trader = {
                'buy_lim_coef': buy_lim_coef,
                'stop_loss_coef': stop_loss_coef,
                'wallet': np.nan,
                'n_trades': np.nan
            }
            trader_params = trader_params.append(trader, ignore_index=True)
        return trader_params
