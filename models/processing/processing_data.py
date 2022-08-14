from operator import imod


import numpy as np
import tensorflow as tf


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
    
