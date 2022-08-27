from abc import ABC, abstractmethod
from unicodedata import name
import tensorflow as tf
import numpy as np


class BaseModel(ABC):
    def __init__(
        self,
        input_shape,
        output_shape,
        learning_rate=0.001,
        epochs=100,
        batch_size=64,
        verbose=1,
        patient=50,
        name=''
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
        self.patient = patient

        self.model = None
        self.history = None
        self.metrics = None
        self.confusion_matrix = None
        self.classification_report = None
        self.name = name

    @abstractmethod
    def build_model(self):
        """
        Build the model for the architecture using tensorflow2 framework for trading
        """
        pass

    def fit_train_test(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
        """
        Train the model for the architecture using
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patient)
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model for the architecture using
        """
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=self.patient)
        self.history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=callback,
            use_multiprocessing=True
        )
        self.metrics = self.model.evaluate(
            X, y, verbose=self.verbose)

    def predict(self, x: np.ndarray):
        """
        Predict the model for the architecture using
        """
        return self.model.predict(x)
