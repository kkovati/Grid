import matplotlib.pyplot as plt
import plotly.express as px


def plot_histograms(df, x, y, n_bins=10):
    # https://plotly.com/python/2D-Histogram/
    fig = px.density_heatmap(df, x=x, y=y, nbinsx=n_bins, nbinsy=n_bins, text_auto=True)
    fig.show()

    fig = px.density_heatmap(df, x=x, y=y, z="n_trades", histfunc="avg", nbinsx=n_bins, nbinsy=n_bins, text_auto=True)
    fig.show()

    fig = px.density_heatmap(df, x=x, y=y, z="wallet", histfunc="avg", nbinsx=n_bins, nbinsy=n_bins, text_auto=True)
    fig.show()


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
