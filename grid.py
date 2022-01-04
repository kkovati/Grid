from alive_progress import alive_bar
from dataclasses import dataclass
import pandas as pd
import math
from math import isclose
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from traders import GridTrader, LimitTrader

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


def plot_results(interval, account, trader, grid=None):
    fig, axs = plt.subplots(2)
    # axs[0].set_yticks([0, 50, 100, 150, 200], minor=False)
    if grid is not None:
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


class AccountSimulator:
    def __init__(self):
        self.original_wallet = self.wallet = 1
        self.wallet_timeline = []
        self.open_buy_orders = []
        self.open_trades = []
        self.market_actions = []
        self.n_buys = 0

    def update(self, i, price):
        # self.execute_limit_orders(price)
        self.wallet_timeline.append(self.calculate_investment_value(price))

    def get_open_buy_orders(self):
        return self.open_buy_orders

    def get_open_trades(self):
        return self.open_trades

    def execute_market_buy(self, i, price, qty=1, sell_limit=None, stop_loss=None):
        # limited functionality: one open trade at a time
        # limited functionality: buy with all available money
        assert len(self.open_trades) == 0
        # https://www.binance.com/en/fee/trading
        self.wallet *= 1 - (0.001 * 2)  # buy and sell commission combined
        self.open_trades.append({'buy_price': price,
                                 'qty': qty,
                                 'sell_limit': sell_limit,
                                 'stop_loss': stop_loss})
        assert len(self.open_trades) == 1
        self.market_actions.append(i)
        self.n_buys += 1
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


class MarketSimulator:
    def __init__(self, trader_params):
        assert isinstance(trader_params, pd.DataFrame)
        self.trader_params = trader_params

    def simulate(self, interval):
        best_wallet = -9999
        for i in tqdm(range(len(self.trader_params))):
            account = AccountSimulator()
            single_trader_params = self.trader_params.loc[i]

            trader = GridTrader(
                step=single_trader_params['step'],
                stop_loss_coef=single_trader_params['stop_loss_coef'])

            # trader = LimitTrader(
            #     buy_lim_coef=single_trader_params['step'],
            #     stop_loss_coef=single_trader_params['stop_loss_coef'])

            for j, price in enumerate(interval):
                account.update(j, price)
                action = trader.update(price)
                assert action in ("do_nothing", "buy", "sell")
                if action == "buy":
                    account.execute_market_buy(j, price)
                elif action == "sell":
                    account.execute_market_sell(j, price)

            wallet = account.get_wallet()
            self.trader_params.at[i, 'wallet'] = wallet

            if wallet > best_wallet:
                saved_trader = trader
                saved_account = account
                best_wallet = wallet

        # calculate relevant grid
        # grid = [saved_trader.base_price]
        # price = saved_trader.base_price
        # while price < max(interval):
        #     price *= saved_trader.step
        #     grid.append(price)
        # price = saved_trader.base_price
        # while min(interval) < price:
        #     price /= saved_trader.step
        #     grid.append(price)

        # plot_results(interval, saved_account, saved_trader, grid)
        plot_results(interval, saved_account, saved_trader)

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

    # https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1m/
    # https://github.com/binance/binance-public-data/
    df = pd.read_csv('data/BTCUSDT-1m-2021-11.csv')
    ts = df.iloc[:, 4].to_numpy()

    trader_params = get_random_trader_population(n_traders=200, base_price=45000.01)

    me = MarketSimulator(trader_params)
    me.simulate(ts)

    # plot_results(ts, traders, traders[0].grid)


if __name__ == '__main__':
    main()
