import faust
from logging.config import dictConfig

# KAFKA_BOOTSTRAP_SERVER = "kafka://kafka:9092"
KAFKA_BOOTSTRAP_SERVER = '135.181.30.253:29092'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
    },
    'loggers': {
        'crypto2data': {
            'handlers': ['console'],
            'level': 'INFO',
        },
    },
}

app = faust.App(
    version=1,
    autodiscover=True,
    # store='rocksdb://',
    origin='crypto2data',
    # debug=True,
    id="1.1",
    broker=KAFKA_BOOTSTRAP_SERVER,
    logging_config=dictConfig(LOGGING)
)


def main() -> None:
    app.main()
