import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, input_shape_data, hidden_layers, output_shape_data):
        super(Autoencoder, self).__init__()
        self.input_shape_data = input_shape_data
        self.hidden_layers = hidden_layers
        self.output_shape_data = output_shape_data
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()

    def build_encoder(self):
        encoder = tf.keras.Sequential()
        encoder.add(tf.keras.layers.InputLayer(input_shape=self.input_shape_data))
        for i in range(len(self.hidden_layers)):
            encoder.add(tf.keras.layers.Dense(
                self.hidden_layers[i], activation='relu'))
        return encoder

    def build_decoder(self):
        decoder = tf.keras.Sequential()
        decoder.add(tf.keras.layers.InputLayer(
            input_shape=self.hidden_layers[-1]))
        for i in range(len(self.hidden_layers)):
            decoder.add(tf.keras.layers.Dense(
                self.hidden_layers[-i-1], activation='relu'))
        decoder.add(tf.keras.layers.Dense(
            self.output_shape, activation='sigmoid'))
        return decoder

    def build_autoencoder(self):
        autoencoder = tf.keras.Sequential()
        autoencoder.add(self.encoder)
        autoencoder.add(self.decoder)
        return autoencoder

    def call(self, inputs):
        return self.autoencoder(inputs)

    def train(self, data, epochs, batch_size, loss):
        self.autoencoder.compile(
            optimizer='adam', loss='mse', metrics=['accuracy'])
        self.history = self.autoencoder.fit(
            data, loss, epochs=epochs, batch_size=batch_size)
        self.metrics = self.autoencoder.evaluate(data, data)

    def predict(self, data):
        return self.autoencoder.predict(data)



def create_features(data, features_shape, hidden_layers, epochs, batch_size, loss):
    autoencoder = Autoencoder(features_shape, hidden_layers, features_shape)
    autoencoder.train(data, epochs, batch_size, loss)
    return autoencoder.predict(data)

# testing the code:
data = np.random.rand(100, 10)
features = create_features(data, 10, [10, 10], 100, 100, 'mse')   
print(features.shape)
print(features)
print(data.shape)
print(data)   
print(features == data)
print(features == data.reshape(100, 10))
print(features == data.reshape(100, 10).T)