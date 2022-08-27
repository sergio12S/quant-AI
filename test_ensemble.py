from tools.dataset import Dataset
from models.enssemble_models import EnsembleModel
from models.processing.create_features import createFeatures
import pickle


if __name__ == '__main__':
    dataset = Dataset().get_data(days=20, ticker='BTCUSDT', ts='1h')
    features = createFeatures().create_features(dataset[['open', 'high', 'low', 'close']])

    ensemble = EnsembleModel(features)
    ensemble.init_models()
    ensemble.fit_train_test()
    ensemble.predict()
    predicted = ensemble.get_predicted()
    print(predicted)
    # save ensemble pickle
    with open('ensemble.pickle', 'wb') as f:
        pickle.dump(ensemble, f)