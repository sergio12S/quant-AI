import pandas as pd
import numpy as np
import pandas_ta as ta


class createFeatures:
    def __init__(self) -> None:
        pass

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data.ta.strategy("Momentum")
        cols = data.columns
        cols_filtered = list(
            filter(lambda x:  x.startswith('Q') is False, cols))
        return data[cols_filtered].dropna()

    def create_volume_bars(self, trades: pd.DataFrame, frequency: int = 10) -> np.array:
        times = trades.index.values
        prices = trades['close'].values
        volumes = trades['volume'].values
        ans = np.zeros(shape=(len(prices), 6))
        candle_counter = 0
        vol = 0
        lasti = 0
        for i in range(len(prices)):
            vol += volumes[i]
            if vol >= frequency:
                ans[candle_counter][0] = times[i]  # time
                ans[candle_counter][1] = prices[lasti]  # open
                ans[candle_counter][2] = np.max(prices[lasti:i+1])  # high
                ans[candle_counter][3] = np.min(prices[lasti:i+1])  # low
                ans[candle_counter][4] = prices[i]  # close
                ans[candle_counter][5] = np.sum(volumes[lasti:i+1])  # volume
                candle_counter += 1
                lasti = i+1
                vol = 0
        return ans[:candle_counter]
