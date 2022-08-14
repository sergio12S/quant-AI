import logging
from streamData.app import app
from streamData.parser.parsers_grey_scale import pase_grey_scale
from streamData.graph.send_data import send_data_to_gremlin


logger = logging.getLogger(__name__)


topic_greyscale = 'grey_scale_portfolio'

stream_topic_grayscale = app.topic(
    topic_greyscale
)


table_greyscale = app.Table('table_greyscale', partitions=1)
table_cftc = app.Table('table_cftc', partitions=1)


@app.agent(stream_topic_grayscale)
async def grayscale(stream):
    async for p in stream:
        table_greyscale['data'] = p


@app.crontab('15 6 * * *')
async def parse_at_6_am_greyscale():
    print('grey')
    msg = pase_grey_scale()
    if not msg:
        return
    for i in msg:
        i['name'] = f"gs_{i.get('Symbol')}"
    send_data_to_gremlin(msg)
    logger.info('grey_scale')
    logger.info(msg)
    await stream_topic_grayscale.send(
        value=msg
    )


@app.page('/grey-scale/')
@app.table_route(table=table_greyscale, query_param='data')
async def get_portfolio_greyscale(web, request):
    params = request.query['data']
    return web.json({
        'portfolio': table_greyscale[params],
    })
