
from tools.dataset import Dataset
from models.architecture.rnn_net import RnnGruNetRegressor
from models.processing.processing_data import Transformer
import numpy as np
import pandas as pd
import tensorflow as tf



def test_model():
    dataset = Dataset().get_data(days=30, ticker='BTCUSDT', ts='1h')
    transformer = Transformer()
    X, y = transformer.transform_to_lstm_input(
        data=dataset,
        cols=['open', 'high', 'low', 'close'],
        window_size=12
    )
    X_split = transformer.split_data(X, 0.2)
    y_split = transformer.split_data(y, 0.2)

    model = RnnGruNetRegressor(
        input_shape=X.shape[1:],
        output_shape=y.shape[1],
        epochs=1000
    )
    model.build_model()
    model.fit_train_test(X_split[0], y_split[0], X_split[1], y_split[1])
    # model.fit(X, y)

    predicted = model.predict(X_split[1])
    predicted = transformer.inverse_min_max_scaler(predicted)
    print(f'predicted: {predicted}')


