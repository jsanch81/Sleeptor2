from feature_extraction import Featurizer
from model import Model

def main():
    featurizer = Featurizer()
    data = featurizer.run()
    X = data[:, :-1]
    y = data[:, [-1]]
    model = Model()
    model.train(X, y)

if __name__ == '__main__':
    main()