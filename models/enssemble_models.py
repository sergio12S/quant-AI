from models.architecture.rnn_net import RnnGruNetRegressor
from models.processing.processing_data import Transformer
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np


@dataclass
class Store:
    model: RnnGruNetRegressor
    transformer: Transformer
    dataset: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    X_split: np.ndarray
    y_split: np.ndarray
    predictions: Dict


class EnsembleModel:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset
        self.stores = []

    def init_models(self, lag=12):
        for i in range(lag):
            transformer = Transformer()
            X, y = transformer.transform_to_lstm_input(
                data=self.dataset,
                cols=['open', 'high', 'low', 'close'],
                window_size=12,
                step_size=i
            )
            X_split = transformer.split_data(X, 0.2)
            y_split = transformer.split_data(y, 0.2)
            name = f'LSTM_regressor_{i}'
            model = RnnGruNetRegressor(
                input_shape=X.shape[1:],
                output_shape=y.shape[1],
                epochs=1000,
                name=name
            )
            model.build_model()
            self.stores.append(
                Store(model, transformer, self.dataset, X,
                      y, X_split, y_split, {name: None})
            )

    def fit_train_test(self):
        for i in self.stores:
            i.model.fit_train_test(
                i.X_split[0],
                i.y_split[0],
                i.X_split[1],
                i.y_split[1]
            )

    def predict(self):
        for i in self.stores:
            predict = i.model.predict(i.X_split[1])
            i.predictions[i.model.name] = i.transformer.inverse_min_max_scaler(
                predict)

    def get_predicted(self):
        return {i.model.name: i.predictions[i.model.name] for i in self.stores}
