from feature_extraction import Featurizer
from model import Model
from sleeptor import Sleeptor
import pandas as pd
def main():
    featurizer = Featurizer()
    data, medias = featurizer.run()

    df = pd.DataFrame(data)
    df.to_csv('data.csv')
    # mode = 'concat'
    mode = 'concat'
    type_model = 'logisticRegression'

    model = Model(featurizer.n_features, mode, featurizer.size, featurizer.n_samples_per_video, type_model)
    if(mode == 'concat'):
       X = data[:, :-1]
       y = data[:, [-1]]
       model.train(X, y)
    elif(mode == 'medias'):
        X = medias[:, :-1]
        y = medias[:, [-1]]
        model.train(X, y)

    # sleeptor = Sleeptor(featurizer, model)
    # sleeptor.live(mode)

if __name__ == '__main__':
    main()
