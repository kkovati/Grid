import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.error import ClientError
import time


"""
This module is a trial for:
binance/binance-connector-python 
This seems to be the official lib
"""

# API Key: sW9XjJLxvqDBiFZtQ9m3FDK1iHYTMxIZttQH8AUL9AzYbJoXrvW8dAgUXcsMgrdD
# Secret Key: CuZWeDhZxmk2G6MqOlBC0I6Zo3yAOtBpKZcIgyDmfLibJTN7ww4kcIVWrJIda9mR
# https://testnet.binance.vision/api


config_logging(logging, logging.DEBUG)

params = {
    "symbol": "BTCUSDT",
    "side": "SELL",
    "type": "LIMIT",
    "timeInForce": "GTC",
    "quantity": 0.002,
    "price": 50000,
}


def get_price(client, ticker):
    response = None
    try:
        # response = client.new_order(**params)
        # response = client.get_open_orders("BTCUSDT")
        # response = client.book_ticker("BTCUSDT")
        response = client.ticker_price(ticker)
        logging.info(response)

    except ClientError as error:
        logging.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )

    return response["price"]


def buy(client, ticker, side, quantity):
    assert side in ("buy", "sell")
    params = {
        "symbol": ticker.uppercase(),
        "side": side.uppercase(),
        "type": "MARKET",
        "quantity": quantity
    }

    try:
        response = client.new_order(**params)
        logging.info(response)

    except ClientError as error:
        logging.error(
            "Found error. status: {}, error code: {}, error message: {}".format(
                error.status_code, error.error_code, error.error_message
            )
        )


def acc_snapshot(client):
    try:
        response = client.account_snapshot("SPOT")
        logging.info(response)
    except ClientError as error:
        logging.error(f"Found error. status: {error.status_code}, error code: {error.error_code}, "
                      f"error message: {error.error_message}")


def acc_status(client):
    response = client.account_status()
    logging.info(response)


def main():
    ticker = "BTCUSDT"

    # Test
    # key = "sW9XjJLxvqDBiFZtQ9m3FDK1iHYTMxIZttQH8AUL9AzYbJoXrvW8dAgUXcsMgrdD"
    # secret = "CuZWeDhZxmk2G6MqOlBC0I6Zo3yAOtBpKZcIgyDmfLibJTN7ww4kcIVWrJIda9mR"
    # client = Client(key, secret, base_url="https://testnet.binance.vision")

    # Live
    key = "Wh89KrKccEHXs3yYAnELU9bPE3EKMncm5im7HAbB4VmHl6wjpBeiOzzybkJhWcSz"
    secret = "f8VMgKFdJT1nDoppeGfVwyVkeJpCvOYyZOZsx6orWYmLPpIOWi3osSACSVBp0ZLw"
    client = Client(key, secret)

    while True:
        logging.info("-----------")
        # price = get_price(client, ticker)
        acc_snapshot(client)
        time.sleep(2)


class BinanceAccountConnector:
    def __init__(self):
        pass


if __name__ == '__main__':
    main()
