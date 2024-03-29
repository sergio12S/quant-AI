from typing import List, Dict, Tuple, Union


import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


class ProcessingData:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.data_shape = data.shape
        self.data_size = data.size
        self.data_mean = data.mean()
        self.data_std = data.std()
        self.data_min = data.min()
        self.data_max = data.max()
        self.data_median = np.median(data)
        self.data_mode = np.bincount(data).argmax()
        self.data_range = self.data_max - self.data_min
        self.data_variance = data.var()
        self.data_skew = data.skew()
        self.data_kurtosis = data.kurtosis()
        self.data_entropy = data.entropy()
        self.data_quantiles = np.quantile(data, q=[0.25, 0.5, 0.75])
        self.data_quantiles_diff = self.data_quantiles[1] - \
            self.data_quantiles[0]
        self.data_quantiles_diff_2 = self.data_quantiles[2] - \
            self.data_quantiles[1]
        self.data_quantiles_diff_3 = self.data_quantiles[2] - \
            self.data_quantiles[0]
        self.data_quantiles_diff_4 = self.data_quantiles[1] - \
            self.data_quantiles[0]
        self.data_quantiles_diff_5 = self.data_quantiles[2] - \
            self.data_quantiles[1]
        self.data_quantiles_diff_6 = self.data_quantiles[2] - \
            self.data_quantiles[0]
        self.data_quantiles_diff_7 = self.data_quantiles[1] - \
            self.data_quantiles[0]
        self.data_quantiles_diff_8 = self.data_quantiles[2] - \
            self.data_quantiles[1]

    def get_data_shape(self):
        return self.data_shape

    def get_data_size(self):
        return self.data_size

    def standardize(self):
        return (self.data - self.data_mean) / self.data_std

    def scalling(self):
        return (self.data - self.data_min) / self.data_range

    def normalize(self):
        return (self.data - self.data_mean) / self.data_std


class Transformer:
    def __init__(self) -> None:
        self.scaler_x = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.index = None

    def fit_transform_x(self, data: np.ndarray):
        return self.scaler_x.fit_transform(data)

    def fit_transform_y(self, data: np.ndarray):
        return self.scaler_y.fit_transform(data)

    def min_max_scaler_x(self, data: np.ndarray):
        return self.scaler_x.transform(data)

    def min_max_scaler_y(self, data: np.ndarray):
        return self.scaler_y.transform(data)

    def inverse_min_max_scaler_x(self, data: np.ndarray):
        return self.scaler_x.inverse_transform(data)

    def inverse_min_max_scaler_y(self, data: np.ndarray):
        return self.scaler_y.inverse_transform(data)

    def split_data(self, data: np.ndarray, test_size: float):
        return np.split(data, [int(test_size * data.shape[0])])

    def transform_to_lstm_input(
        self,
        data: pd.DataFrame,
        cols: List,
        window_size: int,
        step_size: int = 0,
        ahead: int = 0,
    ):
        self.index = data.index
        x_scaler = data[cols].values
        y_scaler = data[['open', 'high', 'low', 'close']].values
        x_scaler = self.fit_transform_x(x_scaler)
        y_scaler = self.fit_transform_y(y_scaler)
        X = []
        y = []
        indexes = []
        for i in range(window_size, x_scaler.shape[0] - step_size - ahead):
            X.append(x_scaler[i-window_size:i])
            if ahead > 0:
                y.append(y_scaler[i:i+step_size+ahead])
            else:
                y.append(y_scaler[i + step_size][:4])
            indexes.append(self.index[i])

        return np.array(X), np.array(y), np.array(indexes)
