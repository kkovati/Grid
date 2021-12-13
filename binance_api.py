import logging
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
from binance.error import ClientError

config_logging(logging, logging.DEBUG)

# API Key: sW9XjJLxvqDBiFZtQ9m3FDK1iHYTMxIZttQH8AUL9AzYbJoXrvW8dAgUXcsMgrdD
# Secret Key: CuZWeDhZxmk2G6MqOlBC0I6Zo3yAOtBpKZcIgyDmfLibJTN7ww4kcIVWrJIda9mR
# https://testnet.binance.vision/api

key = "sW9XjJLxvqDBiFZtQ9m3FDK1iHYTMxIZttQH8AUL9AzYbJoXrvW8dAgUXcsMgrdD"
secret = "CuZWeDhZxmk2G6MqOlBC0I6Zo3yAOtBpKZcIgyDmfLibJTN7ww4kcIVWrJIda9mR"

params = {
    "symbol": "BTCUSDT",
    "side": "SELL",
    "type": "LIMIT",
    "timeInForce": "GTC",
    "quantity": 0.002,
    "price": 50000,
}

client = Client(key, secret, base_url="https://testnet.binance.vision")

try:
    # response = client.new_order(**params)
    # response = client.get_open_orders("BTCUSDT")
    response = client.book_ticker("BTCUSDT")
    logging.info(response)
    response = client.ticker_price("BTCUSDT")
    logging.info(response)

except ClientError as error:
    logging.error(
        "Found error. status: {}, error code: {}, error message: {}".format(
            error.status_code, error.error_code, error.error_message
        )
    )
