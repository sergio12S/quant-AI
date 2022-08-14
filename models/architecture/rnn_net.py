import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import numpy as np


class RnnGruNetClassifier:
    def __init__(self, input_shape, output_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.epochs = 100
        self.batch_size = 32
        self.verbose = 1
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=self.input_shape,
                input_shape=(self.input_shape, 1)
            )
        )
        self.model.add(Dropout(0.2))
        self.model.add(RepeatVector(self.input_shape))
        self.model.add(GRU(units=self.input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(
            Dense(units=self.output_shape, activation='sigmoid')))
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, x_train, y_train, x_test, y_test):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        self.history = self.model.fit(
            x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callback, use_multiprocessing=True)
        self.metrics = self.model.evaluate(
            x_test,
            y_test,
            verbose=self.verbose
        )

    def predict(self, x_test):
        return self.model.predict(x_test)


class RnnGruNetRegressor:
    def __init__(self, input_shape, output_shape, learning_rate=0.001):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.epochs = 100
        self.batch_size = 32
        self.verbose = 1
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(
            LSTM(
                units=self.input_shape,
                input_shape=(self.input_shape, 1)
            )
        )
        self.model.add(Dropout(0.2))
        self.model.add(RepeatVector(self.input_shape))
        self.model.add(GRU(units=self.input_shape, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(
            Dense(units=self.output_shape, activation='linear')))
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='mean_squared_error', metrics=['mse'])

    def fit(self, x_train, y_train, x_test, y_test):
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        self.history = self.model.fit(
            x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose, callbacks=callback, use_multiprocessing=True)
        self.metrics = self.model.evaluate(
            x_test,
            y_test,
            verbose=self.verbose
        )

    def predict(self, x_test):
        return self.model.predict(x_test)


# Test the RnnGruNet class for trading with random data
if __name__ == '__main__':
    x_train = np.random.rand(1000, 10)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    y_train = np.random.rand(1000, 1)
    x_test = np.random.rand(1000, 10)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    y_test = np.random.rand(1000, 1)

    # classifier
    rnn_gru_net_classifier = RnnGruNetClassifier(
        input_shape=10, output_shape=1)
    rnn_gru_net_classifier.fit(x_train, y_train, x_test, y_test)
    rnn_gru_net_classifier.predict(x_test)
    print(rnn_gru_net_classifier.metrics)
    # regressor
    rnn_gru_net_regressor = RnnGruNetRegressor(
        input_shape=10, output_shape=1)
    rnn_gru_net_regressor.fit(x_train, y_train, x_test, y_test)
    rnn_gru_net_regressor.predict(x_test)
    print(rnn_gru_net_regressor.metrics)
