import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from base_model import BaseModel


class ForwardNetClassifier(BaseModel):
    def build_model(self):
        """
        Build the model for the architecture using tensorflow2 framework for trading
        """
        model = Sequential()
        model.add(Dense(
            units=self.input_shape,
            activation='relu',
            input_shape=(self.input_shape,)
        ))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.output_shape, activation='sigmoid'))
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        self.model = model


class ForwardNetRegressor(BaseModel):
    def build_model(self):
        """
        Build the model for the architecture using tensorflow2 framework for trading
        """
        model = Sequential()
        model.add(Dense(
            units=self.input_shape,
            activation='relu',
            input_shape=(self.input_shape,)
        ))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.output_shape, activation='linear'))
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error',
            metrics=['mse']
        )
        self.model = model


if __name__ == '__main__':
    # create random data
    x_train = np.random.rand(100, 10)
    y_train = np.random.randint(2, size=(100, 1))
    x_test = np.random.rand(100, 10)
    y_test = np.random.randint(2, size=(100, 1))

    # Test the class ForwardNetClassifier
    forward_net = ForwardNetClassifier(input_shape=10, output_shape=1)
    forward_net.build_model()
    forward_net.fit(x_train, y_train, x_test, y_test)
    forward_net.predict(x_test)

    # Test the class ForwardNetRegressor
    forward_net = ForwardNetRegressor(input_shape=10, output_shape=1)
    forward_net.build_model()
    forward_net.fit(x_train, y_train, x_test, y_test)
    forward_net.predict(x_test)
