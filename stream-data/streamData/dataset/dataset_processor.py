from faust import web
from streamData.app import app
from streamData.db.db import SQLPostgres, FeturesCrypto
from streamData.db.schema import SqlToDict
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


blueprint = web.Blueprint('dataset')


@blueprint.route('/dataset/features/{timeframe}/', name='featuresDataset')
class FeaturesDataset(web.View):
    '''
    Create route for get features data from database
    '''
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



app.web.blueprints.add('/dataset/', blueprint)
