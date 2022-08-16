import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.optimizers import Adam
from models.architecture.base_model import BaseModel
import numpy as np


class RnnGruNetClassifier(BaseModel):
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


class RnnGruNetRegressor(BaseModel):
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
        self.model.add(Flatten())
        self.model.add(Dense(units=self.output_shape, activation='linear'))
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='mae', metrics=['mae'])


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
