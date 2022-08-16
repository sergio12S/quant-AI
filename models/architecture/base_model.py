from abc import ABC, abstractmethod


import tensorflow as tf


class BaseModel(ABC):
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

    @abstractmethod
    def build_model(self):
        """
        Build the model for the architecture using tensorflow2 framework for trading
        """
        pass

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