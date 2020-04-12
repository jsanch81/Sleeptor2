from feature_extraction import Featurizer
from model import Model

def main():
    featurizer = Featurizer()
    X,y = featurizer.run()
    model = Model()
    model.train(X, y)

if __name__ == '__main__':
    main()