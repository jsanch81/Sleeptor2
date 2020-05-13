from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pickle
import numpy as np
import os

def normalize(data):
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    data_s = (data-means)/stds
    return data_s


class Model():
    def __init__(self, n_features, mode, size, n_samples):
        self.n_features = n_features
        self.mode = mode
        self.size = size
        self.n_samples = n_samples

    def balanced_split(self, X):
        X_train = X[:-2*self.n_samples]
        y_train = y[:-2*self.n_samples]
        X_test = X[-2*self.n_samples:]
        y_test = y[-2*self.n_samples:]

        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        model_filename = 'data/models/model_' + str(self.n_features) + '_' + str(self.size) + '_' + str(len(X)) + '_' + self.mode + '.pkl'
        y = y.ravel()
        X_train, X_test, y_train, y_test = self.balanced_split(X, y)

        if(os.path.isfile(model_filename)):
            self.model = pickle.load(open(model_filename, 'rb'))
        else:
            self.model = RandomForestClassifier()
            #self.model = SVC(probability=True, kernel='sigmoid')
            self.model.fit(X_train, y_train)
            pickle.dump(self.model, open(model_filename, 'wb'))

        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        probas = self.model.predict_proba(X_test)[:,1]
        f1 = metrics.f1_score(y_test, preds)
        roc = metrics.roc_auc_score(y_test, probas)

        print(' acc: {}\n f1 score: {}\n roc_auc score: {}'.format(acc, f1, roc))
        print('confusion matrix')
        print(confusion_matrix(y_test, preds))


