import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.error import ClientError
import time

config_logging(logging, logging.DEBUG)

# API Key: sW9XjJLxvqDBiFZtQ9m3FDK1iHYTMxIZttQH8AUL9AzYbJoXrvW8dAgUXcsMgrdD
# Secret Key: CuZWeDhZxmk2G6MqOlBC0I6Zo3yAOtBpKZcIgyDmfLibJTN7ww4kcIVWrJIda9mR
# https://testnet.binance.vision/api


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
        "symbol": ticker,
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


def main():
    ticker = "BTCUSDT"

    key = "sW9XjJLxvqDBiFZtQ9m3FDK1iHYTMxIZttQH8AUL9AzYbJoXrvW8dAgUXcsMgrdD"
    secret = "CuZWeDhZxmk2G6MqOlBC0I6Zo3yAOtBpKZcIgyDmfLibJTN7ww4kcIVWrJIda9mR"

    client = Client(key, secret, base_url="https://testnet.binance.vision")

    while True:
        logging.info("-----------")
        price = get_price(client, ticker)
        time.sleep(2)


if __name__ == '__main__':
    main()
