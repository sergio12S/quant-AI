# crete network architecture using tensorflof 2 for CNN model for trading, using random data

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from base_model import BaseModel
import numpy as np


class ConvNetClassifier(BaseModel):
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Conv1D(
                filters=self.input_shape,
                kernel_size=1,
                activation='relu',
                input_shape=(self.input_shape, 1)
            )
        )
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=self.input_shape,
                       kernel_size=1, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(units=self.input_shape, activation='relu'))
        self.model.add(Dense(units=self.output_shape, activation='sigmoid'))
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='binary_crossentropy', metrics=['accuracy'])


class ConvNetRegressor(BaseModel):
    def build_model(self):
        self.model = Sequential()
        self.model.add(
            Conv1D(
                filters=self.input_shape,
                kernel_size=1,
                activation='relu',
                input_shape=(self.input_shape, 1)
            )
        )
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Conv1D(filters=self.input_shape,
                       kernel_size=1, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(units=self.input_shape, activation='relu'))
        self.model.add(Dense(units=self.output_shape, activation='linear'))
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                           loss='mean_squared_error', metrics=['mse'])


if __name__ == "main":
    # Create random data for testing the ConvNet class
    x_train = np.random.rand(1000, 10)
    x_test = np.random.rand(1000, 10)
    y_train = np.random.rand(1000, 1)
    y_test = np.random.rand(1000, 1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # tess classifier
    model_classifier = ConvNetClassifier(input_shape=10, output_shape=1)
    model_classifier.fit(x_train, y_train, x_test, y_test)
    model_classifier.predict(x_test)
    model_classifier.metrics
    #  test regressor
    model_regressor = ConvNetRegressor(input_shape=10, output_shape=1)
    model_regressor.fit(x_train, y_train, x_test, y_test)
    model_regressor.predict(x_test)
    model_regressor.metrics
