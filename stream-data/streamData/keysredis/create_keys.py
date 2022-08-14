from streamData.app import app
from streamData.db.redis_conn import RedisConn

@app.task
async def create_keys_redis():
    RedisConn().create_keys(retention=60)
        
