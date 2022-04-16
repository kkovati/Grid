
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager


"""
This module is a trial for:

sammchardy/python-binance
This is a much more contributed lib than the binance-connector
https://github.com/sammchardy/python-binance
https://python-binance.readthedocs.io/en/latest/



Note: check which currencies have the lowest interest rates
https://www.binance.com/en/fee/marginFee


"""


if __name__ == '__main__':

    api_key = "asd"
    api_secret = "asd"
    client = Client(api_key, api_secret)
