import pandas as pd
import numpy as np
import pandas_ta as ta


class createFeatures:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def _techical_indicators(self):
        return ta.add_all_ta_features(self.data, open="open", high="high", low="low", close="close", volume="volume")

    def create_features(self):
        return self._techical_indicators()

    def create_labels(self):
        return self.data['close']

    def create_data(self):
        return self.create_features(), self.create_labels()
