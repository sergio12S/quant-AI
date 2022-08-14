import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, plot_confusion_matrix
import numpy as np


class ForwardNetClassifier:
    def __init__(
        self,
        input_shape,
        output_shape,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        verbose=1
    ):
        """
        Initialize the ForwardNet class
        :param input_shape: shape of the input data
        :param output_shape: shape of the output data
        :param learning_rate: learning rate for the optimizer
        :param epochs: number of epochs for training
        :param batch_size: batch size for training
        :param verbose: verbosity for the training
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = None
        self.history = None
        self.metrics = None
        self.confusion_matrix = None
        self.classification_report = None

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

    def fit(self, x_train, y_train, x_test, y_test):
        """
        Train the model for the architecture using
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        self.history = self.model.fit(
            x_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=(x_test, y_test),
            callbacks=callback,
            use_multiprocessing=True
        )
        self.metrics = self.model.evaluate(
            x_test, y_test, verbose=self.verbose)

    def predict(self, x_test):
        """
        Predict the model for the architecture using
        """
        return self.model.predict(x_test)


class ForwardNetRegressor:
    def __init__(
        self,
        input_shape,
        output_shape,
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        verbose=1
    ):
        """
        Initialize the ForwardNet class
        :param input_shape: shape of the input data
        :param output_shape: shape of the output data
        :param learning_rate: learning rate for the optimizer
        :param epochs: number of epochs for training
        :param batch_size: batch size for training
        :param verbose: verbosity for the training
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.model = None
        self.history = None
        self.metrics = None
        self.confusion_matrix = None
        self.classification_report = None

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

    def fit(self, x_train, y_train, x_test, y_test):
        """
        Train the model for the architecture using
        """
        callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3)
        self.history = self.model.fit(
            x_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_data=(x_test, y_test),
            callbacks=callback,
            use_multiprocessing=True
        )
        self.metrics = self.model.evaluate(
            x_test, y_test, verbose=self.verbose)

    def predict(self, x_test):
        """
        Predict the model for the architecture using
        """
        return self.model.predict(x_test)


# test the class ForwardNet for random data
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
