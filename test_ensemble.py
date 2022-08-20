from tools.dataset import Dataset
from models.enssemble_models import EnsembleModel


if __name__ == '__main__':
    dataset = Dataset().get_data(days=30, ticker='BTCUSDT', ts='1h')
    ensemble = EnsembleModel(dataset)
    ensemble.init_models()
    ensemble.fit_train_test()
    ensemble.predict()
    predicted = ensemble.get_predicted()
    print(predicted)
