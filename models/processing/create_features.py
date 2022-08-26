import pandas as pd
import numpy as np
import pandas_ta as ta


class createFeatures:
    def __init__(self) -> None:
        pass

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data.ta.strategy("Momentum")
        cols = data.columns
        cols_filtered = list(filter(lambda x:  x.startswith('Q') is False, cols))
        return data[cols_filtered].dropna()
