
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pickle
import plotly.graph_objects as go
from example.strategies.tools.dataset import Dataset


PATH = 'example/strategies/strategies/deep_learning/saved_model/'


class AgentTrader:
    def __init__(self) -> None:
        self.models = []

    def create_model(self, cols):

        model = Sequential()
        model.add(Dense(64, activation='relu',
                        input_shape=(len(cols),)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy', metrics=['accuracy']
        )
        return model

    def save_model(self, model, name) -> None:
        model\
            .save(f'{PATH}momentum_{name}')

    def save_models(self, list_models):
        names = [f'strategy_{i+1}' for i in range(len(list_models))]
        [self.save_model(list_models[i], names[i])
         for i in range(len(list_models))]

    def load_model(self, name):
        return tf.keras\
            .models\
            .load_model(f'{PATH}momentum_{name}')

    def load_models(self, number_models):
        names = [f'strategy_{i+1}' for i in range(number_models)]
        return [self.load_model(i) for i in names]

    def save_obj(self, agent_models, name):
        self.save_models(agent_models[name]['model'])
        with open(f'{PATH}agent_models_data.pickle', 'wb') as handle:
            pickle.dump(
                {your_key: agent_models[name][your_key]
                 for your_key in [
                    'data', 'mu', 'std']},
                handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    def load_data(self):
        with open(f"{PATH}agent_models_data.pickle", "rb") as f:
            dictname = pickle.load(f)
        return dictname


class DataFeatures:
    def create_features(self, data, lags):
        features = 'volume'
        cols = self._get_indicators(data, features, lags)
        for lag in range(1, lags + 1):
            col = f'corr_vol_lag_{lag}'
            data.loc[:, col] = data['corr_volume'].shift(lag)
            cols.append(col)
        data = data.dropna(inplace=False)
        return data, cols

    def get_last_features(self, data, lags):
        data = data[:-1]
        cols = self._get_indicators(data, 'volume', lags)
        for lag in range(1, lags + 1):
            col = f'corr_vol_lag_{lag}'
            data.loc[:, col] = data['corr_volume'].shift(lag-1)
            cols.append(col)
        data = data.dropna(inplace=False)
        return data.tail(1), cols

    def _get_indicators(self, data, arg1, lags):
        data['returns'] = data['close'].pct_change(1)
        data['ma_vol'] = data[arg1].rolling(24).mean()
        data['corr_volume'] = data['ma_vol'].rolling(24).corr(data[arg1])
        data['direction'] = np.where(data['returns'] > 0, 1, 0)
        data['chg'] = data['close'].pct_change()
        data['corr'] = data['chg'].rolling(
            12).corr(data['chg'].shift(1)).dropna()
        data['cuts_corr'] = pd.cut(data['corr'], bins=12, labels=False)
        result = []
        for lag in range(1, lags + 1):
            col = f'cuts_corr_{lag}'
            data.loc[:, col] = data['cuts_corr'].shift(lag)
            result.append(col)
        return result

    def get_features_fit(self, data, lags):
        data, cols = self.create_features(data, lags)
        data = data[:-1]
        return data.tail(1), cols


class MomentumDenseStrategy:
    def __init__(self) -> None:
        pass

    def plot_results(self, results_strategy,
                     name="Deep Learing for trade BTCUSDT"):

        data_cumsum = results_strategy.cumsum().apply(np.exp)
        fig = go.Figure()
        for i in data_cumsum.columns:
            fig.add_trace(
                go.Scatter(
                    name=i,
                    mode="markers+lines",
                    x=data_cumsum.index,
                    y=data_cumsum[i],
                    marker_symbol="star"
                )
            )
        fig.update_xaxes(showgrid=True, ticklabelmode="period",
                         title=name)
        fig.show()

    def run_backtest_(self, data, length, epochs, cols):
        data['predicted'] = 0
        MODEL_TRAINED = False
        models = []
        for i in range(length - 1, len(data) - 1):
            if not MODEL_TRAINED:
                MODEL_TRAINED = True

                for _ in range(1, 11):
                    print('TRAIN', i)
                    temp = data.iloc[:i].copy()
                    mu, std, model = self.train_model(epochs, cols, temp)
                    models.append(model)
            if MODEL_TRAINED:

                X = data.iloc[i][cols].copy()
                y = data.iloc[i]['direction'].copy()
                X_ = self.scalar_data(X, mu, std)\
                    .values\
                    .reshape(1, len(cols))\
                    .astype(float)
                predict_proba = self.predict_on_batch(models, X_)

                self.train_on_batch(models, y, X_)

                signal = self.make_signal(predict_proba)

                data['predicted'].iloc[i] = signal

        return {'data': data, 'model': models, 'mu': mu, 'std': std}

    def run_backtest_test(self, data, tail):
        data, cols = DataFeatures().create_features(data, lags=50)
        data = data.tail(tail)
        upload_data = AgentTrader().load_data()
        mu = upload_data.get('mu')
        std = upload_data.get('std')
        models = AgentTrader().load_models(number_models=10)
        data['predicted'] = 0
        for i in range(1, len(data)):

            X = data.iloc[i][cols].copy()
            y = data.iloc[i]['direction'].copy()
            X_ = self.scalar_data(X, mu, std)\
                .values\
                .reshape(1, len(cols))\
                .astype(float)
            predict_proba = self.predict_on_batch(models, X_)

            self.train_on_batch(models, y, X_)

            signal = self.make_signal(predict_proba)

            data['predicted'].iloc[i] = signal

        return data

    # @tf.function
    def train_on_batch(self, models, y, X_):
        return [i.fit(X_, np.array(y).reshape(
                1, 1), epochs=1, verbose=False) for i in models]

    # @tf.function
    def predict_on_batch(self, models, X_):
        return np.mean([i.predict(X_) for i in models])

    def make_signal(self, predict_proba, proba=0.5):
        signal = 0
        if predict_proba < proba:
            # print('Short')
            signal = -1
        if predict_proba > proba:
            # print('Long')
            signal = 1
        return signal

    def train_model(self, epochs, cols, temp):
        X = temp[cols]
        y = temp['direction']
        mu, std = X.mean(), X.std()
        X_ = self.scalar_data(X, mu, std)
        model = AgentTrader().create_model(cols)
        callback = tf.keras.callbacks\
            .EarlyStopping(monitor='loss', patience=3)
        model.fit(
            X_.values,
            y.values,
            epochs=epochs,
            verbose=False,
            shuffle=True,
            callbacks=callback,
            use_multiprocessing=True
        )

        return mu, std, model

    def scalar_data(self, X, mu, std):
        return (X - mu) / std

    def calculate_results(self, data, length):
        training_data = data.tail(len(data) - length)
        training_data.loc[:, 'strategy'] = (training_data['predicted'] *
                                            training_data['returns'])

        training_data.loc[:, 'strategy_'] = np.where(
            training_data['predicted'].diff() != 0,
            training_data['strategy'] - 0.001,
            training_data['strategy']
        )
        print(training_data[['returns', 'strategy', 'strategy_']]
              .sum()
              .apply(np.exp))

        return training_data

    def check_stochastic_dynamic(self, data, lags, length, epochs, stoch):
        data, cols = DataFeatures().create_features(data=data, lags=lags)
        results_strategy = pd.DataFrame()
        agent_models = {}
        for i in range(1, stoch + 1):
            data_obj = self.run_backtest_(data, length, epochs, cols)
            agent_models[f'strategy_{i}'] = data_obj
            training_data = self.calculate_results(
                data_obj.get('data'), length)
            results_strategy[f'strategy_{i}'] = training_data['strategy_']
        results_strategy['returns'] = training_data['returns']
        return results_strategy, agent_models

    def run_online(self):
        upload_data = AgentTrader().load_data()
        mu = upload_data.get('mu')
        std = upload_data.get('std')
        data = Dataset().get_data(days=30, ticker='BTCUSDT', ts='6h')
        models = AgentTrader().load_models(number_models=10)

        signal = self.get_signal(data, mu, std, models)
        self.train_batch(data, mu, std, models)
        last_data = data[:-1].tail(1)
        last_data['signal'] = signal
        return last_data

    def train_batch(self, data, mu, std, models):
        data_, cols = DataFeatures().get_features_fit(data, lags=50)
        X = data_[cols]
        y = data_['direction'].values
        X_ = self.scalar_data(X, mu, std)\
            .values\
            .reshape(1, len(cols))\
            .astype(float)

        [i.fit(X_, np.array(y).reshape(
            1, 1), epochs=1, verbose=False) for i in models]
        AgentTrader().save_models(models)

    def get_signal(self, data, mu, std, models):
        X, cols = DataFeatures().get_last_features(data, lags=50)
        X_ = self.scalar_data(X[cols], mu, std)\
            .values\
            .reshape(1, len(cols))\
            .astype(float)

        predict_proba = np.mean([i.predict(X_) for i in models])
        return self.make_signal(predict_proba)
