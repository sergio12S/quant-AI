import logging
from streamData.app import app
from streamData.parser.parsers_cot import get_cot_data
from streamData.graph.send_data import send_data_to_gremlin
logger = logging.getLogger(__name__)


topic_cot = 'cotbtc_topic'


stream_topic_cot = app.topic(topic_cot)
table_cot = app.Table('table_cot', partitions=1)


@app.agent(stream_topic_cot)
async def cftc(stream):
    async for p in stream:
        table_cot['ticker'] = p


@app.crontab('5 6 * * FRI')
async def parse_at_6_am_cot():
    msg = get_cot_data()
    send_data_to_gremlin(msg)
    logger.info('cot BTC')
    logger.info(msg)
    await stream_topic_cot.send(
        value=msg
    )


@app.page('/cot/')
@app.table_route(table=table_cot, query_param='ticker')
async def get_cftc(web, request):
    logger.info(request)
    params = request.query['ticker']
    return web.json({
        'data': table_cot[params],
    })
