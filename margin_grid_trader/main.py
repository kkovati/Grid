import pandas as pd
from tqdm import tqdm

from .simulation import MarginAccountSimulator
from .trader import LongShortTrader


def simulate(self, interval):
    best_wallet = -9999

    trader_params_df = LongShortTrader.get_random_trader_population(n_traders=1000, step_lo=0.01, step_hi=0.1)

    for i in tqdm(range(len(trader_params_df))):
        account = MarginAccountSimulator()
        single_trader_params = trader_params_df.loc[i]

        trader = LongShortTrader(acc=account, initial_price=interval[0], step=single_trader_params['step'])

        for j, price in enumerate(interval):
            account.update(j, price)
            trader.update(price)

        wallet = account.get_investment_value(interval[-1])
        trader_params_df.at[i, 'wallet'] = wallet
        trader_params_df.at[i, 'n_trades'] = account.get_trade_count()

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
    # plot_results(interval, saved_account, saved_trader)

    # print(self.trader_params)
    # print(self.trader_params.iloc[self.trader_params['wallet'].argmax()])
    # # https://www.statology.org/matplotlib-scatterplot-color-by-value/
    # plt.scatter(self.trader_params.step,
    #             self.trader_params.stop_loss_coef,
    #             s=50,
    #             c=self.trader_params.wallet,
    #             cmap='gray')
    # plt.xlabel("step")
    # plt.ylabel("stop_loss_coef")
    # plt.show()

    return trader_params_df


def main():
    df = pd.read_csv('data/BTCUSDT-1m-2021.csv')
    # TODO use high/low values instead close
    ts = df.iloc[:, 4].to_numpy()

    simulate(ts)

    plot_histograms(trader_params_df, x='buy_lim_coef', y='stop_loss_coef', n_bins=10)

    # plot_results(ts, traders, traders[0].grid)


if __name__ == '__main__':
    pass
