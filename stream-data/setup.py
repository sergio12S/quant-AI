from setuptools import setup, find_packages

requires = [
    "simple-settings",
    "faust",
    "python-binance",
    "requests",
    "tweepy",
    "gremlinpython",
    "pytz",
    "beautifulsoup4",
    "nest-asyncio",
    "pandas",
    "redistimeseries",
    "psycopg2-binary",
    "SQLAlchemy",
    "marshmallow",
    "GoogleNews"
]

setup(
    name='streamData',
    version='1.2.1',
    description='streamData streaming data',
    long_description='''
    streamData running Faust with Docker Compose
    (zookeeper, kafka and schema-registry)
    ''',
    classifiers=[
        "Programming Language :: Python",
    ],
    author='Serhii Ovsiienko',
    author_email='sott095@gmail.com',
    url='',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requires,
    tests_require=[],
    setup_requires=[],
    dependency_links=[],
    entry_points={
        'console_scripts': [
            'streamData = streamData.app:main',
        ]
    },
)
