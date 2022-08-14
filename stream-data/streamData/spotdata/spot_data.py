import json
from typing import List
from faust import web
from streamData.app import app
from streamData.db.db import SQLPostgres, SpotPrice
import logging
from datetime import datetime, timezone
import requests
from faust import Record, Schema


logger = logging.getLogger(__name__)


def parse_binance_spot_price():
    try:
        data = requests.get('https://api.binance.com/api/v3/ticker/24hr')
        if data.status_code == 200:
            return [x for x in data.json() if x['symbol'].endswith('USDT')]
    except Exception as e:
        logger.error(e)
        return


class SpotPriceFeatures(Record, coerce=True, serializer='json'):
    timestamp: int
    data: List[dict]


schema = Schema(
    value_type=SpotPriceFeatures,
    value_serializer='json'
)


tumbling_topic = app.topic('spot_price_topic',  schema=schema)


@app.timer(interval=60)
async def save_data():
    timestamp = int(datetime.timestamp(datetime.now(timezone.utc)) // 60 * 60)
    date = datetime.fromtimestamp(timestamp)
    data_parse = parse_binance_spot_price()
    if not data_parse:
        logger.error('Parse binance spot price is None')
        return

    # Add all data features
    try:
        add_features = SpotPrice(
            timestamp=date,
            transaction=json.dumps(data_parse)
        )
        with SQLPostgres() as conn:
            conn.add(add_features)
            conn.commit()
    except Exception as e:
        logger.error(e)

    # Add data to tumbling topic
    try:
        data_btc = SpotPriceFeatures(
            timestamp=date,
            data=data_parse
        )
        await tumbling_topic.send(value=data_btc)
    except Exception as e:
        logger.error(e)

blueprint = web.Blueprint('spotData')


@blueprint.route('/from/{timestamp}', name='from')
class SpotData(web.View):
    async def get(self, request: web.Request, timestamp: str) -> web.Response:
        with SQLPostgres() as conn:
            data = conn.query(SpotPrice).with_entities(
                SpotPrice.timestamp,
                SpotPrice.transaction
            )\
                .filter(SpotPrice.timestamp >=
                        datetime.fromtimestamp(int(timestamp)))\
                .all()
            return self.json({"data": data.transaction})


@blueprint.route('/is/{timestamp}', name='is_exist')
class SpotDataIs(web.View):
    async def get(self, request: web.Request, timestamp: str) -> web.Response:
        with SQLPostgres() as conn:
            data = conn.query(SpotPrice).with_entities(
                SpotPrice.timestamp,
                SpotPrice.transaction
            )\
                .filter(SpotPrice.timestamp ==
                        datetime.fromtimestamp(int(timestamp)))\
                .all()
            return self.json({"data": data.transaction})


@blueprint.route('/last', name='spot_data_last')
class SpotDataLast(web.View):
    async def get(self, request: web.Request) -> web.Response:
        with SQLPostgres() as conn:
            data = conn.query(SpotPrice).with_entities(
                SpotPrice.timestamp,
                SpotPrice.transaction
            )\
                .order_by(SpotPrice.timestamp.desc())\
                .first()
            return self.json({"data": data.transaction})


app.web.blueprints.add('/spot/', blueprint)
