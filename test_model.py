
from tools.dataset import Dataset
from models.architecture.rnn_net import RnnGruNetRegressor
from sklearn.preprocessing import  MinMaxScaler
import numpy as np
import pandas as pd


def test_model():
    dataset = Dataset().get_data(days=30, ticker='BTCUSDT', ts='1h')
    for i in range(1, 12):
        dataset[f'lag_{i}'] = dataset['close'].shift(-i)
    dataset = dataset.dropna()

    # create dataset X with last 12 bars as features and y as target
    X = []
    y = []
    window_size = 12
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaler = pd.DataFrame(scaler.fit_transform(dataset[['close']]).squeeze(), index=dataset.index, columns=['close'])
    for i in range(window_size, len(X_scaler)):
        X.append(X_scaler['close'][i-window_size:i].values)
        y.append(X_scaler['close'][i])
    # create train and test sets
    X_train = np.array(X[:int(len(X) * 0.8)])
    y_train = np.array(y[:int(len(y) * 0.8)])
    X_test = np.array(X[int(len(X) * 0.8):])
    y_test = np.array(y[int(len(y) * 0.8):])


    model = RnnGruNetRegressor(window_size, 1)
    # model.epochs = 500
    model.build_model()
    model.fit(X_train, y_train, X_test, y_test)
    predicted = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted)
    actual = scaler.inverse_transform(y_test)
    print(f'predicted: {predicted}')
    print(f'actual: {actual}')