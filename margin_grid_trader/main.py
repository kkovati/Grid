import logging
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from margin_grid_trader.account import MarginAccountSimulator
from margin_grid_trader.trader import LongShortTrader


def simulate(interval):
    best_wallet = -9999
    # logging.basicConfig(level=logging.DEBUG)

    trader_params_df = LongShortTrader.get_random_trader_population(n_traders=1, step_lo=0.01, step_hi=0.1)
    trader_params_df.at[0, "step"] = 0.02  # TODO for debug

    for i in tqdm(range(len(trader_params_df))):
        account = MarginAccountSimulator()
        single_trader_params = trader_params_df.loc[i]

        account.update(0, interval[0])
        trader = LongShortTrader(acc=account, initial_price=interval[0], step=single_trader_params['step'])

        for j, price in enumerate(interval[1:]):
            assert price > 0
            j += 1
            account.update(j, price)
            trader.update(j, price)

        wallet = account.get_investment_value(interval[-1])
        trader_params_df.at[i, 'wallet'] = wallet
        trader_params_df.at[i, 'n_trades'] = account.get_trade_count()

        if wallet > best_wallet:
            saved_trader = trader
            saved_account = account
            best_wallet = wallet

    # logging.basicConfig(level=logging.INFO)

    grid = saved_trader.grid_manager.get_grid_list()

    plot_results(interval, saved_account, saved_trader, grid)

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


def plot_histograms(df, x, y, n_bins=10):
    # https://plotly.com/python/2D-Histogram/
    fig = px.density_heatmap(df, x=x, y=y, nbinsx=n_bins, nbinsy=n_bins, text_auto=True)
    fig.show()

    fig = px.density_heatmap(df, x=x, y=y, z="n_trades", histfunc="avg", nbinsx=n_bins, nbinsy=n_bins, text_auto=True)
    fig.show()

    fig = px.density_heatmap(df, x=x, y=y, z="wallet", histfunc="avg", nbinsx=n_bins, nbinsy=n_bins, text_auto=True)
    fig.show()


def plot_results(interval, account: MarginAccountSimulator, trader, grid=None):
    fig, axs = plt.subplots(3)
    if grid is not None:
        axs[0].set_yticks(grid, minor=True)
    axs[0].yaxis.grid(True, which='minor')

    # axs[0].set_xticks(account.market_actions, minor=True) TODO need this
    # axs[0].xaxis.grid(True, which='minor')
    axs[0].xaxis.grid(True, which='both')

    axs[0].plot(interval)

    axs[1].plot(account.wallet_timeline)  # , label=trader.label)
    axs[1].xaxis.grid(True, which='both')

    axs[2].plot(account.borrowed_value_timeline)  # , label=trader.label)
    axs[2].xaxis.grid(True, which='both')

    plt.legend()
    plt.show()


def main():
    df = pd.read_csv('../data/BTCUSDT-1m-2021.csv')
    # TODO use high/low values instead close
    ts = df.iloc[:, 4].to_numpy()

    simulate(ts)

    # plot_histograms(trader_params_df, x='buy_lim_coef', y='stop_loss_coef', n_bins=10) TODO need this

    # plot_results(ts, traders, traders[0].grid)


if __name__ == '__main__':
    main()

