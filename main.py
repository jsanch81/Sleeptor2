from feature_extraction import Featurizer
from model import Model

def main():
    featurizer = Featurizer()
    data, medias = featurizer.run()
    X = data[:, :-1]
    y = data[:, [-1]]
    #X = medias[:, :-1]
    #y = medias[:, [-1]]
    model = Model()
    model.train(X, y)

if __name__ == '__main__':
    main()