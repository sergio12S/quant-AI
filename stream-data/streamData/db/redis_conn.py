from redistimeseries.client import Client


NAME_DB = 'redistimeseries'
PORT_DB = 6379


class RedisConn:
    def __init__(self):
        self.database_name = NAME_DB
        self.port_db = PORT_DB

    def __enter__(self):
        self.conn = Client(
            host=self.database_name, port=self.port_db
            )
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.redis.close()


    def create_keys(self, retention):
        try:
            self.conn.create('INTRADAYPRICES:BTCUSDT', retention_msecs=retention*60*1000,
                    labels={'SYMBOL': 'BTCUSDT',
                            'DESC': 'PRICE',
                            'PRICETYPE': 'INTRADAY',
                            },
                    duplicate_policy='last',
                    uncompressed=False)
        except Exception:
            print('INTRADAYPRICES:BTCUSDT Keys already exist')
        try:
            self.conn.create('OPENINTEREST:BTCUSDT', retention_msecs=retention*60*1000,
                    labels={'SYMBOL': 'BTCUSDT',
                            'DESC': 'OPENINTEREST',
                            'DATATYPE': 'INTRADAY',
                            },
                    duplicate_policy='last',
                    uncompressed=False)
        except Exception:
            print('OPENINTEREST:BTCUSDT Keys already exist')
        try:
            self.conn.create('LONGRATE:BTCUSDT', retention_msecs=retention*60*1000,
                    labels={'SYMBOL': 'BTCUSDT',
                            'DESC': 'LONGRATE',
                            'DATATYPE': 'INTRADAY',
                            },
                    duplicate_policy='last',
                    uncompressed=False)
        except Exception:
            print('LONGRATE:BTCUSDT Keys already exist')
        try:
            self.conn.create('SHORTRATE:BTCUSDT', retention_msecs=retention*60*1000,
                    labels={'SYMBOL': 'BTCUSDT',
                            'DESC': 'SHORTRATE',
                            'DATATYPE': 'INTRADAY',
                            },
                    duplicate_policy='last',
                    uncompressed=False)
        except Exception:
            print('SHORTRATE:BTCUSDT Keys already exist')
        try:
            self.conn.create('TWITTER:SENTIMENT',
                    retention_msecs=retention*60*1000,
                    labels={'DATA': 'SENTIMENT',
                            'DESC': 'SENTIMENT',
                            'PRICETYPE': 'INTRADAY',
                            },
                    duplicate_policy='last',
                    uncompressed=False)
        except Exception:
            print('TWITTER:SENTIMENT Keys already exist')

        try:
            self.conn.create('WHALE:COUNT',
                    retention_msecs=retention*60*1000,
                    labels={'DATA': 'TRANSACTIONCOUNT',
                            'PRICETYPE': 'INTRADAY',
                            },
                    duplicate_policy='last',
                    uncompressed=False)
        except Exception:
            print('WHALE:COUNT Keys already exist')
        try:
            self.conn.create('WHALE:VOLUME',
                    retention_msecs=retention*60*1000,
                    labels={'DATA': 'TRANSACTIONVOLUME',
                            'PRICETYPE': 'INTRADAY',
                            },
                    duplicate_policy='last',
                    uncompressed=False)
        except Exception:
            print('WHALE:VOLUME Keys already exist')
        print('Keys created')

