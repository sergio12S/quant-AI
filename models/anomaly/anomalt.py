from tensorflow.keras import Sequential
import numpy as np


class AnomalyDetection():
    def __init__(self, model: Sequential) -> None:
        self.model = model

    def defineTreshold(self, data, loss):
        predict = self.model.predict(data)
        loss_error = np.mean(np.abs(predict - data), axis=1)
        treshold = np.where(loss_error > loss)
        return data[treshold]
