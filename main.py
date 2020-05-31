from feature_extraction import Featurizer
from model import Model
from sleeptor import Sleeptor
import pandas as pd

def main():
    featurizer = Featurizer()

    # ######## ****************** first model ******************** ##########
    # data, medias, data_seleccion_final = featurizer.run()

    # df = pd.DataFrame(data_seleccion_final)
    # df.to_csv('data_seleccion_final.csv')
    # mode = 'concat'
    # # mode = 'medias'
    # # mode = 'RNN'
    # # type_model = 'logisticRegression'
    # #type_model = 'ms'
    # type_model = 'rf'
    # #type_model = 'lstm'

    # model = Model(featurizer.n_features, mode, featurizer.size, featurizer.step, featurizer.n_samples_per_video, type_model)
    # if(mode == 'concat'):
    #     X = data[:, :-1]
    #     y = data[:, [-1]]

    #     model.train(X, y)
    # elif(mode == 'medias'):
    #     X = medias[:, :-1]
    #     y = medias[:, [-1]]

    #     model.train(X, y)
    # elif(type_model == 'lstm'):
    #     X = data[:, :-1]
    #     y = data[:, [-1]]
        
    #     model.train(X, y)

    # # sleeptor = Sleeptor(featurizer, model)
    # # sleeptor.live(mode)



    ########## **************** RNNN ******************** ##############
    images, labels = featurizer.extrac_images()
    #print(images.shape)
    #print(labels.shape)
    mode = 'serie'
    type_model = 'lstm'
    model = Model(featurizer.n_features, mode, featurizer.size, featurizer.step, featurizer.n_samples_per_video, type_model)
    model.train_lstm(images, labels)

    #sleeptor = Sleeptor(featurizer, model)
    #sleeptor.live(mode)


if __name__ == '__main__':
    main()
