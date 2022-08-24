from operator import index
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
    indexes: np.ndarray
    X_split: np.ndarray
    y_split: np.ndarray
    indexes_split: np.ndarray
    predictions: Dict


@dataclass
class Parameters:
    cols = ['open', 'high', 'low', 'close']
    window_size = 12
    split: float = 0.7
    epochs: int = 1000
    numbers_of_models: int = 12


class EnsembleModel:
    def __init__(self, dataset: pd.DataFrame):
        self.params = Parameters()
        self.dataset = dataset
        self.stores = []

    def init_models(self):
        for i in range(self.params.numbers_of_models):
            transformer = Transformer()
            X, y, indexes = transformer.transform_to_lstm_input(
                data=self.dataset,
                cols=self.params.cols,
                window_size=self.params.window_size,
                step_size=i
            )
            X_split = transformer.split_data(X, self.params.split)
            y_split = transformer.split_data(y, self.params.split)
            indexes_split = transformer.split_data(indexes, self.params.split)
            name = f'LSTM_regressor_{i}'
            model = RnnGruNetRegressor(
                input_shape=X.shape[1:],
                output_shape=y.shape[1],
                epochs=self.params.epochs,
                name=name
            )
            model.build_model()
            self.stores.append(
                Store(
                    model, transformer, self.dataset, X,
                    y, indexes, X_split, y_split,
                    indexes_split, {name: None}
                )
            )

    def fit_train_test(self):
        for i in self.stores:
            i.model.fit_train_test(
                i.X_split[0],
                i.y_split[0],
                i.X_split[1],
                i.y_split[1]
            )

    def predict(self, train: bool = False):
        for i in self.stores:
            predict = i.model.predict(i.X_split[0] if train else i.X_split[1])
            i.predictions[i.model.name] = i.transformer.inverse_min_max_scaler(
                predict)

    def get_predicted(self):
        return {i.model.name: i.predictions[i.model.name] for i in self.stores}
