
from tools.dataset import Dataset
from models.architecture.rnn_net import RnnGruNetRegressor
from models.processing.processing_data import Transformer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


def test_model():
    dataset = Dataset().get_data(days=30, ticker='BTCUSDT', ts='1h')
    transformer = Transformer()
    window_size = 12
    X, y = transformer.transform_to_lstm_input(
        dataset['close'].values, window_size=12)

    model = RnnGruNetRegressor(window_size, 1)
    model.build_model()
    model.fit(X, y)

    predicted = model.predict(X)
    predicted = transformer.inverse_min_max_scaler(predicted)
    print(f'predicted: {predicted}')
    print(np.sum(predicted == transformer.inverse_min_max_scaler(y)))
