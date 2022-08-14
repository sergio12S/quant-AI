from marshmallow import Schema, fields
from datetime import datetime
from typing import List


class WhaleSchema(Schema):
    timestamp = fields.Date()
    transation = fields.Dict()


class SqlToDict:
    def __init__(self, data) -> None:
        self.data = data

    def to_timestamp(self, date):
        if isinstance(date, datetime):
            return int(datetime.timestamp(date))
        else:
            return date

    def to_dict(self) -> List:
        arr = []
        for i in self.data:
            keys = [*i.keys()]
            values = [*i]
            values = [self.to_timestamp(d) for d in values]
            arr.append(dict(zip(keys, values)))
        return arr
