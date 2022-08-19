from architecture.rnn_net import RnnGruNetRegressor
from processing.processing_data import Transformer


class Predictor:
    def __init__(self):
        lstm_predictor = []

    def lstm_predictions(self, dataset, window_size, output_shape):
        transformer = Transformer()
        X, y = transformer.transform_to_lstm_input(
            dataset['close'].values, window_size=window_size)

        model = RnnGruNetRegressor(window_size, output_shape)
        model.build_model()
        model.fit(X, y)

        predicted = model.predict(X)
        predicted = transformer.inverse_min_max_scaler(predicted)
        return predicted