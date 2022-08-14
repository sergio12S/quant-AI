from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session


NAME_DB = ''
Base = declarative_base()


class FeaturesData(Base):
    __tablename__ = 'features'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    symbol = Column(String)
    openInterest = Column(Float)
    openInterestAmount = Column(Float)
    volUsd = Column(Float)
    longRate = Column(Float)
    longVolUsd = Column(Float)
    shortRate = Column(Float)
    shortVolUsd = Column(Float)
    turnoverNumber = Column(Float)
    priceChange = Column(Float)
    priceChangePercent = Column(Float)
    volChangePercent = Column(Float)
    h1OIChangePercent = Column(Float)
    h4OIChangePercent = Column(Float)
    oichangePercent = Column(Float)


class WhaleAlert(Base):
    __tablename__ = 'whale'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    transaction = Column(JSON)


class FeturesCrypto(Base):
    __tablename__ = 'features_cryptocurrency'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    transaction = Column(JSON)


class SpotPrice(Base):
    __tablename__ = 'spot_price'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    transaction = Column(JSON)


class SQLPostgres:
    def __init__(self):
        self.database_name = NAME_DB

    def __enter__(self):
        engine = create_engine(self.database_name)
        Session = sessionmaker(bind=engine)
        self.conn = scoped_session(Session)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def create_table(self):
        Base.metadata.create_all(
            create_engine(self.database_name),
            checkfirst=True
        )
