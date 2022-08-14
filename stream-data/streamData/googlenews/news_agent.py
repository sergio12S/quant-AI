from GoogleNews import GoogleNews
from faust import web
from streamData.app import app

blueprint = web.Blueprint('whale')
MESSAGES = []


@app.crontab('0 * * * *')
async def parse_at_6_am_cot():
    global MESSAGES
    googlenews = GoogleNews()
    googlenews.set_lang('en')
    googlenews.set_period('1d')
    googlenews.search('Bitcoin')
    MESSAGES = [googlenews.page_at(i) for i in range(1, 5)]


@blueprint.route('/google/{name}/{period}/{page}/', name='googleNews')
class getGoogleNews(web.View):
    '''
    Get data from google news
    '''
    async def get(
        self,
        request: web.Request,
        name: str,
        period: str,
        page: str
    ) -> web.Response:
        googlenews = GoogleNews()
        googlenews.set_lang('en')
        googlenews.set_period(period)
        googlenews.search(name)

        googlenews.total_count()
        result = googlenews.page_at(int(page))

        return self.json({"data": result})


@blueprint.route('/bitcoin/{pages}/', name='bitcoinNews')
class GetSavedNews(web.View):
    async def get(
        self,
        request: web.Request,
        pages: str
    ) -> web.Response:
        if pages == 'all':
            return self.json({"data": MESSAGES})
        else:
            return self.json({"data": []})


app.web.blueprints.add('/news/', blueprint)
