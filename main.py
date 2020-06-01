from feature_extraction import Featurizer
from model import Model
from sleeptor import Sleeptor
import pandas as pd
from keras.models import load_model

def main():
    featurizer = Featurizer()
    train = False
    if(train):
        images, labels = featurizer.extrac_images()
        print(images.shape)
        print(labels.shape)
        mode = 'serie'
        type_model = 'lstm'
        model = Model(mode, featurizer.size, featurizer.step, featurizer.n_samples_per_video, type_model)
        model.train_lstm(images, labels)
    else:
        model = load_model('data/models/model_final_normalized.h5')

    sleeptor = Sleeptor(featurizer, model)
    sleeptor.live()


if __name__ == '__main__':
    main()
