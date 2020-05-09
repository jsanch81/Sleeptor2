from sklearn.ensemble import RandomForestClassifier
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

def balanced_split(X, y, train_size):
    # indcies de cada clase
    idxs_0 = np.where(y==0)[0]
    idxs_1 = np.where(y==1)[0]

    # desordenar indices
    np.random.shuffle(idxs_0)
    np.random.shuffle(idxs_1)

    # separar cada clase en train y test
    wall = int(min(len(idxs_0), len(idxs_1))*train_size)

    idxs_0_train = idxs_0[:wall]
    idxs_1_train = idxs_1[:wall]
    idxs_0_test = idxs_0[wall:]
    idxs_1_test = idxs_1[wall:]

    idxs_train = np.append(idxs_0_train, idxs_1_train, axis=0)
    idxs_test = np.append(idxs_0_test, idxs_1_test, axis=0)

    X_train = X[idxs_train]
    X_test = X[idxs_test]
    y_train = y[idxs_train]
    y_test = y[idxs_test]

    return X_train, X_test, y_train, y_test

class Model():
    def train(self, X,y):
        model_filename = 'data/models/model.pkl'

        if(os.path.isfile(model_filename)):
            self.model = pickle.load(open(model_filename, 'rb'))
        else:
            y = y.ravel()

            X_train, X_test, y_train, y_test = balanced_split(X, y, 0.7)

            self.model = RandomForestClassifier()
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


