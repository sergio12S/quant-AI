from faust import web
from streamData.parser.parsers_bybt import parse_features
from streamData.app import app
from streamData.db.db import SQLPostgres, FeaturesData, FeturesCrypto
from streamData.db.schema import SqlToDict
from streamData.db.redis_conn import RedisConn
import logging
from datetime import datetime, timezone
from faust import Record, Schema


logger = logging.getLogger(__name__)


class BitcoinDataFeatures(Record, coerce=True, serializer='json'):
    timestamp: int
    symbol: str
    openInterest: float
    openInterestAmount: float
    volUsd: float
    longRate: float
    longVolUsd: float
    shortRate: float
    shortVolUsd: float
    turnoverNumber: float
    priceChange: float
    priceChangePercent: float
    volChangePercent: float
    h1OIChangePercent: float
    h4OIChangePercent: float
    oichangePercent: float


schema = Schema(
    value_type=BitcoinDataFeatures,
    value_serializer='json'
)

tumbling_topic = app.topic('features_topic',  schema=schema)


@app.timer(interval=60)
async def save_data():
    timestamp = int(datetime.timestamp(datetime.now(timezone.utc)) // 60 * 60)
    date = datetime.fromtimestamp(timestamp)
    data_parse = parse_features()
    if not data_parse:
        logger.error('Parse features is None')
        return

    # Add all data features
    try:
        add_features = FeturesCrypto(
            timestamp=date,
            transaction=data_parse
        )
        with SQLPostgres() as conn:
            conn.add(add_features)
            conn.commit()
    except Exception as e:
        logger.error(e)

    # add btc data
    try:
        data_btc = data_parse[0]
        liq_info = data_btc.get('liqInfo')
        long_rate = liq_info['longVolUsd'] / liq_info['totalVolUsd']
        short_rate = liq_info['shortVolUsd'] / liq_info['totalVolUsd']
    except Exception as e:
        logger.error(f'Data btc {e}')
        return

    try:
        add_data = FeaturesData(
            timestamp=date,
            symbol=data_btc.get('symbol'),
            openInterest=data_btc.get('openInterest'),
            openInterestAmount=data_btc.get('openInterestAmount'),
            volUsd=data_btc.get('volUsd'),
            longRate=long_rate,
            longVolUsd=liq_info['longVolUsd'],
            shortRate=short_rate,
            shortVolUsd=liq_info['shortVolUsd'],
            turnoverNumber=liq_info['totalVolUsd'],
            priceChange=data_btc.get('priceChange'),
            priceChangePercent=data_btc.get('priceChangePercent'),
            volChangePercent=data_btc.get('volChangePercent'),
            h1OIChangePercent=data_btc.get('h1OIChangePercent'),
            h4OIChangePercent=data_btc.get('h4OIChangePercent'),
            oichangePercent=data_btc.get('oichangePercent')
        )
        with SQLPostgres() as conn:
            conn.add(add_data)
            conn.commit()
    except Exception as e:
        logger.error(e)

    try:
        with RedisConn() as conn:            
            conn.add(
                key='OPENINTEREST:BTCUSDT',
                timestamp=int(timestamp * 1000),
                value=data_btc.get('openInterest'),
                duplicate_policy='last',
            )
            conn.add(
                key='LONGRATE:BTCUSDT',
                timestamp=int(timestamp * 1000),
                value=long_rate,
                duplicate_policy='last',
            )
            conn.add(
                key='SHORTRATE:BTCUSDT',
                timestamp=int(timestamp * 1000),
                value=short_rate,
                duplicate_policy='last',
            )
    except Exception as e:
        logger.error(e)
    try:
        msg = BitcoinDataFeatures(
            timestamp=timestamp,
            symbol=data_btc.get('symbol'),
            openInterest=data_btc.get('openInterest'),
            openInterestAmount=data_btc.get('openInterestAmount'),
            volUsd=data_btc.get('volUsd'),
            longRate=long_rate,
            longVolUsd=liq_info['longVolUsd'],
            shortRate=short_rate,
            shortVolUsd=liq_info['shortVolUsd'],
            turnoverNumber=liq_info['totalVolUsd'],
            priceChange=data_btc.get('priceChange'),
            priceChangePercent=data_btc.get('priceChangePercent'),
            volChangePercent=data_btc.get('volChangePercent'),
            h1OIChangePercent=data_btc.get('h1OIChangePercent'),
            h4OIChangePercent=data_btc.get('h4OIChangePercent'),
            oichangePercent=data_btc.get('oichangePercent')
        )
        await tumbling_topic.send(value=msg)
    except Exception as e:
        logger.info(e)


blueprint = web.Blueprint('features')


@blueprint.route('/btcusdt/data/{timeframe}/', name='bitcoindata')
class BitcoinData(web.View):
    async def get(self, request: web.Request, timeframe: str) -> web.Response:
        """
        Get bitcoin data
        """
        with SQLPostgres() as conn:
            data = conn.query(FeaturesData).with_entities(
                FeaturesData.timestamp,
                FeaturesData.symbol,
                FeaturesData.openInterest,
                FeaturesData.openInterestAmount,
                FeaturesData.volUsd,
                FeaturesData.longRate,
                FeaturesData.longVolUsd,
                FeaturesData.shortRate,
                FeaturesData.shortVolUsd,
                FeaturesData.turnoverNumber,
                FeaturesData.priceChange,
                FeaturesData.priceChangePercent,
                FeaturesData.volChangePercent,
                FeaturesData.h1OIChangePercent,
                FeaturesData.h4OIChangePercent,
                FeaturesData.oichangePercent
            )\
                .filter(FeaturesData.timestamp >=
                        datetime.fromtimestamp(int(timeframe)))\
                .all()
            return self.json({"data": SqlToDict(data).to_dict()})


@blueprint.route('/last/data/coins/', name='featuresdatalast')
class LastFeaturesData(web.View):
    async def get(self, request: web.Request) -> web.Response:
        with SQLPostgres() as conn:
            data = conn.query(FeturesCrypto).with_entities(
                FeturesCrypto.timestamp,
                FeturesCrypto.transaction
            )\
                .order_by(FeturesCrypto.timestamp.desc())\
                .first()

            return self.json({"data": data[1]})


@blueprint.route('/all/data/transaction/{timeframe}/', name='featuresdata')
class EqualFeaturesData(web.View):
    async def get(self, request: web.Request,
                  timeframe: str) -> web.Response:
        with SQLPostgres() as conn:
            data = conn.query(FeturesCrypto).with_entities(
                FeturesCrypto.timestamp,
                FeturesCrypto.transaction
            )\
                .filter(FeturesCrypto.timestamp ==
                        datetime.fromtimestamp(int(timeframe)))\
                .all()
            return self.json({"data": SqlToDict(data).to_dict()})


@blueprint.route('/all/data/transactions/{timeframe}/',
                 name='fromfeaturesdata')
class FromFeaturesData(web.View):
    async def get(self, request: web.Request, timeframe: str) -> web.Response:
        with SQLPostgres() as conn:
            data = conn.query(FeturesCrypto).with_entities(
                FeturesCrypto.timestamp,
                FeturesCrypto.transaction
            )\
                .filter(FeturesCrypto.timestamp >=
                        datetime.fromtimestamp(int(timeframe)))\
                .all()
            return self.json({"data": SqlToDict(data).to_dict()})


@blueprint.route('/btcusdt/openinterst/{timestamp}/{timeframe}/',
                 name='openinterest')
class BitcoinOpenInterest(web.View):
    async def get(self, request: web.Request, timestamp: int,
                  timeframe: int) -> web.Response:
        with RedisConn() as conn:
            data = dict(conn.range('OPENINTEREST:BTCUSDT', from_time=timestamp,
                                to_time=-1,  aggregation_type='sum',
                                bucket_size_msec=timeframe))
            data = [{'timestamp': i, 'values': v} for i, v in data.items()]
            return self.json({"data": data})


@blueprint.route('/btcusdt/longrate/{timestamp}/{timeframe}/',
                 name='bitcoindata')
class BitcoinLongRate(web.View):
    async def get(self, request: web.Request, timestamp: int,
                  timeframe: int) -> web.Response:
        with RedisConn() as conn:
            data = dict(conn.range('LONGRATE:BTCUSDT', from_time=timestamp,
                                to_time=-1,  aggregation_type='sum',
                                bucket_size_msec=timeframe))
            data = [{'timestamp': i, 'values': v} for i, v in data.items()]
            return self.json({"data": data})


@blueprint.route('/btcusdt/shortrate/{timestamp}/{timeframe}/',
                 name='bitcoindata')
class BitcoinShortRate(web.View):
    async def get(self, request: web.Request,
                  timeframe: int, timestamp: int) -> web.Response:
        with RedisConn() as conn:
            data = dict(conn.range('SHORTRATE:BTCUSDT', from_time=timestamp,
                                to_time=-1,  aggregation_type='sum',
                                bucket_size_msec=timeframe))
            data = [{'timestamp': i, 'values': v} for i, v in data.items()]
            return self.json({"data": data})


app.web.blueprints.add('/features/', blueprint)
