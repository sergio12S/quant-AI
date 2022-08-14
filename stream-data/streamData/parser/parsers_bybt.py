import requests
import json


def parse_features():
    try:
        data = requests.get('https://fapi.coinglass.com/api/futures/coins/markets')
        if data.status_code == 200:
            return json.loads(data.text).get('data')
    except Exception as e:
        print(e)
        return
