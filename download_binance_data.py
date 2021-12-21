import datetime
import requests


def asd():
    url = 'https://www.facebook.com/favicon.ico'
    r = requests.get(url, allow_redirects=True)

    open('facebook.ico', 'wb').write(r.content)


def get_binance_data(ticker, days):
    ticker = ticker.upper()

    for i in range(days):
        date = datetime.date.today() - datetime.timedelta(days=19)
        url = "https://data.binance.vision/data/spot/daily/trades/"
        url += ticker + "/" + ticker + "-trades-"
        url += str(date) + ".zip"

        r = requests.get(url, allow_redirects=True)
        open('facebook.ico', 'wb').write(r.content)


if __name__ == '__main__':
    get_binance_data('btcusdt', 3)
