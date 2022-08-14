import requests
from typing import List


def parse_binance_spot_price():
    data = requests.get('https://api.binance.com/api/v3/ticker/24hr')
    if data.status_code == 200:
        return [x for x in data.json() if x['symbol'].endswith('USDT')]
    return []


def get_data_binance(ticker: str = 'BTCUSDT', interval: str = "4h", limit: int = 1000):
    data = requests.get(
        f"https://api.binance.com/api/v3/klines?symbol={ticker}&interval={interval}&limit={limit}")
    if data.status_code == 200:
        return [
            {
                "openTime": item[0],
                "open": item[1],
                "high": item[2], "low": item[3],
                "close": item[4], "volume": item[5], "closeTime": item[6],
                "quoteAssetVolume": item[7],
                "trades": item[8],
                "takerBaseAssetVolume": item[9],
                "takerQuoteAssetVolume": item[10],
                "ignore": item[11]
            } for item in data.json()
        ]


def get_portfolio_data(tickers: List[str] = None, interval: str = "4h", limit: int = 1000):
    dataset = {}
    for ticker in tickers:
        data = get_data_binance(ticker)
        dataset[ticker] = data
    return dataset
