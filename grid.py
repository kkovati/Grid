from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

OPEN, HIGH, LOW, CLOSE = 0, 1, 2, 3


def generate_random_timeseries(length, start_price=100, mu=.0001, sigma=.01):
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


def plot_results(interval, traders_list, grid):
    fig, axs = plt.subplots(2)
    # axs[0].set_yticks([0, 50, 100, 150, 200], minor=False)
    axs[0].set_yticks(grid, minor=True)
    # axs[0].yaxis.grid(True, which='major')
    axs[0].yaxis.grid(True, which='minor')
    axs[0].plot(interval)

    for i, trader in enumerate(traders_list):
        axs[1].plot(trader.wallet_timeline, label=trader.label)

    plt.legend()
    plt.show()


def get_random_trader_population():
    traders = []
    return traders


@dataclass
class TraderConfig:
    id: int
    base_price: float
    scale: str
    step: float
    buy_at: str
    stop_loss: float


def get_random_trader_config(base_price):
    return TraderConfig(
        id=0,
        base_price=base_price,
        scale='log',
        step=np.random.uniform(0.005, 0.1),
        buy_at=np.random.choice('up', 'down'),
        stop_loss=0.1
    )


class Trader:
    def __init__(self, base_price, scale, step, label):
        self.base_price = base_price
        assert scale in ('lin', 'log')
        self.scale = scale
        self.step = step
        self.grid = self.calc_grid(base_price, scale, step)
        self.grid_idx = None

        self.label = label

        self.wallet = 0
        self.wallet_timeline = []
        self.open_trades = []

    def update(self, price):
        if self.grid_idx is None:
            self.grid_idx = self.get_grid_idx(price, self.grid)
            return

        curr_grid_idx = self.get_grid_idx(price, self.grid)

        self.execute_limit_orders(curr_grid_idx)
        self.wallet_timeline.append(self.calculate_investment_value(price))

        if self.grid_idx < curr_grid_idx:
            for inter_grid_idx in range(self.grid_idx + 1, curr_grid_idx + 1):
                self.step_up(inter_grid_idx)
        elif curr_grid_idx < self.grid_idx:
            for inter_grid_idx in range(self.grid_idx, curr_grid_idx, -1):
                self.step_down(inter_grid_idx)
        self.grid_idx = curr_grid_idx

    def step_up(self, grid_idx):
        raise NotImplementedError

    def step_down(self, grid_idx):
        raise NotImplementedError

    def calc_grid(self, base_price, scale, step):
        grid = [base_price]
        grid_up = base_price
        grid_down = base_price
        for i in range(100):
            grid_up *= (1 + step)
            grid_down /= (1 + step)
            grid.append(grid_up)
            grid.append(grid_down)
        grid.sort()
        return grid

    def get_grid_idx(self, price, grid):
        if price <= grid[0] or grid[-1] <= price:
            raise Exception('Price is out of grid bounds')
        for i in range(len(grid)):
            if price < grid[i]:
                return i - 1
        raise Exception('Something went wrong')

    def buy(self, buy_price, sell_limit=None, stop_loss=None, qty=1):
        self.wallet -= self.grid[buy_price] * 0.01 * 0.05
        self.open_trades.append({'buy_price': buy_price,
                                 'sell_limit': sell_limit,
                                 'stop_loss': stop_loss,
                                 'qty': qty})

    def execute_limit_orders(self, grid_idx):
        for trade in self.open_trades:
            if trade['sell_limit'] is not None:
                if trade['sell_limit'] <= grid_idx:
                    self.wallet += (self.grid[trade['sell_limit']] - self.grid[trade['buy_price']]) * trade['qty']
                    self.open_trades.remove(trade)
                    continue
            if trade['stop_loss'] is not None:
                if grid_idx < trade['stop_loss']:
                    self.wallet += (self.grid[trade['stop_loss']] - self.grid[trade['buy_price']]) * trade['qty']
                    self.open_trades.remove(trade)
                    continue

    def calculate_investment_value(self, price):
        open_trade_sum = 0
        for trade in self.open_trades:
            open_trade_sum += (price - self.grid[trade['buy_price']]) * trade['qty']
        return open_trade_sum + self.wallet

    def check_trade_buy_price(self, grid_idx):
        for trade in self.open_trades:
            if grid_idx == trade['buy_price']:
                return True
        return False


class TraderHold(Trader):
    def __init__(self, base_price, scale, step, label):
        super().__init__(base_price, scale, step, label)
        self.enter_flag = False

    def step_up(self, grid_idx):
        if not self.enter_flag:
            self.buy(grid_idx + 1, qty=2)
            self.enter_flag = True

    def step_down(self, grid_idx):
        if not self.enter_flag:
            self.buy(grid_idx + 1, qty=2)
            self.enter_flag = True


class TraderBuyDip(Trader):
    def __init__(self, base_price, scale, step, label):
        super().__init__(base_price, scale, step, label)

    def step_up(self, grid_idx):
        # sell with sell limit order
        pass

    def step_down(self, grid_idx):
        # buy and at step_up() sell one grid higher
        if not self.check_trade_buy_price(grid_idx):
            self.buy(grid_idx + 1, sell_limit=grid_idx + 2)


class TraderBuyUp(Trader):
    def __init__(self, base_price, scale, step, label):
        super().__init__(base_price, scale, step, label)

    def step_up(self, grid_idx):
        # buy and increase stop_loss
        if not self.check_trade_buy_price(grid_idx):
            self.buy(grid_idx, sell_limit=None, stop_loss=grid_idx - 1)
        for trade in self.open_trades:
            trade['stop_loss'] = grid_idx - 1

    def step_down(self, grid_idx):
        # sell with stop loss order
        pass


class MarketEngine:
    def __init__(self, trader_list):
        assert isinstance(trader_list, list)
        self.trader_list = trader_list

    def run(self, interval):
        for close in interval:
            for trader in self.trader_list:
                trader.update(close)


if __name__ == '__main__':
    ri = generate_random_interval(200, start_price=100)

    # traders = [TraderHold(base_price=110.0, scale='log', step=0.1),
    #            TraderBuyDip(base_price=110.0, scale='log', step=0.1),
    #            TraderBuyUp(base_price=110.0, scale='log', step=0.1)]

    traders = [TraderBuyUp(base_price=110.0, scale='log', step=0.01, label='1'),
               TraderBuyUp(base_price=110.0, scale='log', step=0.02, label='2'),
               TraderBuyUp(base_price=110.0, scale='log', step=0.03, label='3'),
               TraderBuyUp(base_price=110.0, scale='log', step=0.04, label='4'),
               TraderBuyUp(base_price=110.0, scale='log', step=0.05, label='5'),
               TraderBuyUp(base_price=110.0, scale='log', step=0.1, label='10')]

    me = MarketEngine(traders)
    me.run(ri[CLOSE])

    plot_results(ri[CLOSE], traders, traders[0].grid)
