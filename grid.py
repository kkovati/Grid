from alive_progress import alive_bar
from dataclasses import dataclass
import pandas as pd
import math
from math import isclose
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

OPEN, HIGH, LOW, CLOSE = 0, 1, 2, 3


def generate_random_timeseries(length, start_price=100.0, mu=.0001, sigma=.01):
    returns = np.random.normal(loc=mu, scale=sigma, size=length)
    timeseries = start_price * (1 + returns).cumprod()
    timeseries += (start_price - timeseries[0])
    return timeseries


def generate_random_interval(length, start_price=100):
    unit = 10
    timeseries = generate_random_timeseries(unit * length, start_price)
    interval = np.zeros((4, length))
    for i in range(length):
        timeseries_unit = timeseries[i * unit: (i + 1) * unit]
        interval[OPEN, i] = timeseries_unit[0]
        interval[HIGH, i] = max(timeseries_unit)
        interval[LOW, i] = min(timeseries_unit)
        interval[CLOSE, i] = timeseries_unit[unit - 1]
    return interval


def convert_interval_to_timeseries(interval):
    assert interval.ndim == 2
    assert interval.shape[0] == 4
    timeseries = np.zeros(4 * interval.shape[1], dtype=float)
    for i in range(interval.shape[1]):
        o = interval[OPEN, i]
        h = interval[HIGH, i]
        l = interval[LOW, i]
        c = interval[CLOSE, i]
        timeseries[4 * i] = o
        if abs(o - h) < abs(o - l):
            timeseries[4 * i + 1] = h
            timeseries[4 * i + 2] = l
        else:
            timeseries[4 * i + 1] = l
            timeseries[4 * i + 2] = h
        timeseries[4 * i + 3] = c
    return timeseries


def plot_results(interval, account, trader, grid):
    fig, axs = plt.subplots(2)
    # axs[0].set_yticks([0, 50, 100, 150, 200], minor=False)
    axs[0].set_yticks(grid, minor=True)
    # axs[0].yaxis.grid(True, which='major')
    axs[0].yaxis.grid(True, which='minor')

    axs[0].set_xticks(account.market_actions, minor=True)
    axs[0].xaxis.grid(True, which='minor')

    axs[0].plot(interval)

    axs[1].plot(account.wallet_timeline)  # , label=trader.label)

    plt.legend()
    plt.show()


def get_random_trader_population(n_traders, base_price):
    # trader_params = pd.DataFrame(columns=('base_price', 'scale', 'step', 'buy_at', 'stop_loss_coef', 'wallet'))
    trader_params = pd.DataFrame(columns=('base_price', 'step', 'stop_loss_coef', 'wallet'))
    for i in range(n_traders):
        # step = 10 ** np.random.uniform(math.log10(0.005), math.log10(0.1))
        step = np.random.uniform(0.005, 0.1)
        # stop_loss_coef = 10 ** np.random.uniform(math.log10(0.005), math.log10(0.1))
        stop_loss_coef = np.random.uniform(0.005, 0.1)
        trader = {
            # 'base_price': np.random.uniform(base_price * 0.9, base_price * 1.1),
            'base_price': base_price,
            # 'scale': np.random.choice(('log', 'lin')),
            'step': step,
            # 'buy_at': np.random.choice(('up', 'down')),
            'stop_loss_coef': stop_loss_coef,
            'wallet': np.nan
        }
        trader_params = trader_params.append(trader, ignore_index=True)
    return trader_params


class Trader:
    def __init__(self, account, base_price, step, stop_loss_coef):
        # assert scale in ('lin', 'log')

        self.account = account

        self.base_price = base_price
        # self.scale = scale
        self.step = step
        self.stop_loss_coef = 1 - stop_loss_coef

        self.grid = self.calc_grid(base_price, step)
        # self.grid_idx = None
        self.grid_idx_list = []

        self.stop_loss_price = None  # stop-loss price of the only one open trade

    def update(self, i, price):
        if not self.grid_idx_list:
            curr_grid_idx = self.get_grid_idx(price, self.grid)
            self.grid_idx_list.append(curr_grid_idx)
            return

        curr_grid_idx = self.get_grid_idx(price, self.grid)
        if self.grid_idx_list[-1] != curr_grid_idx:
            self.grid_idx_list.append(curr_grid_idx)

        if self.stop_loss_price is not None:
            # an open trade exists
            if price <= self.stop_loss_price:
                # sell
                self.account.execute_market_sell(i, price)
                self.stop_loss_price = None
            else:
                # update stop-loss price
                self.stop_loss_price = max(self.stop_loss_price, price * self.stop_loss_coef)
        else:
            # no open trade
            if len(self.grid_idx_list) < 3:
                return
            if self.grid_idx_list[-3] < self.grid_idx_list[-2] < self.grid_idx_list[-1]:
                # buy
                self.account.execute_market_buy(i, price)
                self.stop_loss_price = price * self.stop_loss_coef

    def update_old_2(self, price):
        if self.grid_idx is None:
            self.grid_idx = self.get_grid_idx(price, self.grid)
            self.grid_idx_list.append(self.grid_idx)
            return

        curr_grid_idx = self.get_grid_idx(price, self.grid)

        if self.grid_idx_list[-1] != curr_grid_idx:
            self.grid_idx_list.append(curr_grid_idx)

        if self.check_open_trade_buy_price(curr_grid_idx + 1):
            pass
        else:
            buy_order = {'buy_limit': self.grid[curr_grid_idx + 1],
                         'qty': 1}
            self.place_buy_order(buy_order)

    def update_old_1(self, curr_grid_idx):
        if self.grid_idx < curr_grid_idx:
            self.step_up(curr_grid_idx)
        elif curr_grid_idx < self.grid_idx:
            self.step_down(curr_grid_idx)
        self.grid_idx = curr_grid_idx

    def step_up(self, grid_idx):
        raise NotImplementedError
        buy_up = True
        if buy_up:
            # buy and increase stop_loss
            if not self.check_trade_buy_price(grid_idx):
                self.buy(grid_idx, sell_limit=None, stop_loss=grid_idx - 1)
            for trade in self.open_trades:
                trade['stop_loss'] = grid_idx - 1

        buy_dip = False
        if buy_dip:
            # sell with sell limit order
            pass

    def step_down(self, grid_idx):
        raise NotImplementedError
        buy_up = True
        if buy_up:
            # sell with stop loss order
            pass

        buy_dip = False
        if buy_dip:
            # buy and at step_up() sell one grid higher
            if not self.check_trade_buy_price(grid_idx):
                self.buy(grid_idx + 1, sell_limit=grid_idx + 2)

    def calc_grid(self, base_price, step):
        grid = [base_price]
        grid_up = base_price
        grid_down = base_price
        for i in range(200):  # TODO: major bottleneck
            grid_up *= (1 + step)
            grid_down /= (1 + step)
            grid.append(grid_up)
            grid.append(grid_down)
        grid.sort()
        return grid

    def get_grid_idx(self, price, grid):  # TODO: major bottleneck
        if price <= grid[0] or grid[-1] <= price:
            raise Exception('Price is out of grid bounds')
        for i in range(len(grid)):
            if price < grid[i]:
                return i - 1
        raise Exception('Something went wrong')

    def check_open_buy_order_price(self, grid_idx):
        raise NotImplementedError
        for buy_order in self.account.get_open_buy_orders():
            if isclose(self.grid[grid_idx], buy_order['buy_limit'], rel_tol=0.001):
                return True
        return False

    def check_open_trade_buy_price(self, grid_idx):
        for trade in self.account.get_open_trades():
            if isclose(self.grid[grid_idx], trade['buy_price'], rel_tol=0.001):
                return True
        return False

    def is_open_trade(self):
        pass


class Account:
    def __init__(self):
        self.wallet = 0
        self.wallet_timeline = []
        self.open_buy_orders = []
        self.open_trades = []
        self.market_actions = []

    def update(self, i, price):
        # self.execute_limit_orders(price)
        self.wallet_timeline.append(self.calculate_investment_value(price))

    def get_open_buy_orders(self):
        return self.open_buy_orders

    def get_open_trades(self):
        return self.open_trades

    def execute_market_buy(self, i, price, qty=1, sell_limit=None, stop_loss=None):
        # limited functionality: one open trade at a time
        assert len(self.open_trades) == 0
        self.wallet -= price * qty  # value of trade
        self.wallet -= price * qty * 0.01 * 0.05  # commission
        self.open_trades.append({'buy_price': price,
                                 'qty': qty,
                                 'sell_limit': sell_limit,
                                 'stop_loss': stop_loss})
        assert len(self.open_trades) == 1
        self.market_actions.append(i)
        # print(f"buy @ {price:.4}")

    def execute_market_sell(self, i, price):
        # limited functionality: sell all (one) open trades
        assert len(self.open_trades) == 1
        for trade in self.open_trades:
            self.wallet += price * trade['qty']
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
        open_trade_sum = 0
        for trade in self.open_trades:
            open_trade_sum += price * trade['qty']
        return open_trade_sum + self.wallet

    def get_wallet(self):
        return self.wallet_timeline[-1]


class TraderHold(Trader):
    def __init__(self, base_price, scale, step):
        super().__init__(base_price, scale, step)
        self.enter_flag = False

    def step_up(self, grid_idx):
        if not self.enter_flag:
            self.buy(grid_idx + 1, qty=2)
            self.enter_flag = True

    def step_down(self, grid_idx):
        if not self.enter_flag:
            self.buy(grid_idx + 1, qty=2)
            self.enter_flag = True


class MarketSimulator:
    def __init__(self, trader_params):
        assert isinstance(trader_params, pd.DataFrame)
        self.trader_params = trader_params

    def simulate(self, interval):
        best_wallet = -9999
        for i in tqdm(range(len(self.trader_params))):
            account = Account()
            single_trader_params = self.trader_params.loc[i]
            trader = Trader(
                account=account,
                base_price=single_trader_params['base_price'],
                # scale=single_trader_params['scale'],
                step=single_trader_params['step'],
                stop_loss_coef=single_trader_params['stop_loss_coef'])

            for j, price in enumerate(interval):
                account.update(j, price)
                trader.update(j, price)

            wallet = account.get_wallet()
            self.trader_params.at[i, 'wallet'] = wallet

            if wallet > best_wallet:
                saved_trader = trader
                saved_account = account
                best_wallet = wallet

        plot_results(ts, saved_account, saved_trader, saved_trader.grid)

        print(self.trader_params)
        print(self.trader_params.iloc[self.trader_params['wallet'].argmax()])
        # https://www.statology.org/matplotlib-scatterplot-color-by-value/
        plt.scatter(self.trader_params.step,
                    self.trader_params.stop_loss_coef,
                    s=50,
                    c=self.trader_params.wallet,
                    cmap='gray')
        plt.xlabel("step")
        plt.ylabel("stop_loss_coef")
        plt.show()


def main():
    # ri = generate_random_interval(200, start_price=100)
    # ts = generate_random_timeseries(10000, start_price=100.01)  # , mu=.00005, sigma=.005)
    df = pd.read_csv('data/BTCUSDT-1m-2021-11.csv')
    ts = df.iloc[:, 4].to_numpy()

    trader_params = get_random_trader_population(n_traders=1000, base_price=45000.01)

    me = MarketSimulator(trader_params)
    me.simulate(ts)

    # plot_results(ts, traders, traders[0].grid)


if __name__ == '__main__':
    main()
