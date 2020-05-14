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
        self.n_samples = n_samples - size

    def balanced_split(self, X, y, idx):
        idxs_train = set(np.arange(len(X)))
        idxs_test = set(np.arange(idx*2*self.n_samples, (idx+1)*2*self.n_samples))
        idxs_train = np.array(list(idxs_train - idxs_test))
        idxs_test = np.array(list(idxs_test))
        
        X_train = X[idxs_train]
        y_train = y[idxs_train]
        X_test = X[idxs_test]
        y_test = y[idxs_test]

        return X_train, X_test, y_train, y_test

    def train(self, X, y):
        y = y.ravel()
        print(len(X))

        n_people = int(len(X)/(2*self.n_samples))
        print("n people:", n_people)
        acc_global = 0
        f1_global = 0
        roc_global = 0
        for idx in range(n_people):
            model_filename = 'data/models/model_' + str(self.n_features) + '_' + str(self.size) + '_' + str(len(X)) + '_' + self.mode + '_' + str(idx) + '.pkl'
            X_train, X_test, y_train, y_test = self.balanced_split(X, y, idx)

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

            print('for idx:', idx)
            print(' acc: {}\n f1 score: {}\n roc_auc score: {}'.format(acc, f1, roc))
            print('confusion matrix')
            print(confusion_matrix(y_test, preds))

            acc_global += acc
            f1_global += f1
            roc_global += roc
        
        acc_global /= n_people
        f1_global /= n_people
        roc_global /= n_people
        print('\nglobal:')
        print(' acc: {}\n f1 score: {}\n roc_auc score: {}'.format(acc_global, f1_global, roc_global))


