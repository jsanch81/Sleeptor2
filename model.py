from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

def normalize(data):
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1)
    data_s = (data-means)/stds
    return data_s

def calculate_best_k(X_train, X_test, y_train, y_test, ks):
    best_k = 0
    acc = 0.0
    for k in ks:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(X_train, y_train)
        pred_KN = neigh.predict(X_test)
        # pred_KN = average(pred_KN)
        acc_k = accuracy_score(y_test, pred_KN)
        if(acc_k > acc):
            best_k = k
            acc = acc_k
    return best_k

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
        y = y.ravel()

        X_train, X_test, y_train, y_test = balanced_split(X, y, 0.7)

        #best_k = calculate_best_k(X_train, X_test, y_train, y_test, range(1,10))
        #self.model = KNeighborsClassifier(n_neighbors=best_k)
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        # preds = average(preds)
        probas = self.model.predict_proba(X_test)[:,1]
        f1 = metrics.f1_score(y_test, preds)
        roc = metrics.roc_auc_score(y_test, probas)
        print(' acc: {}\n f1 score: {}\n roc_auc score: {}'.format(acc, f1, roc))
        print('confusion matrix')
        print(confusion_matrix(y_test, preds))
        filename = 'data/models/knn.pkl'
        pickle.dump(self.model, open(filename, 'wb'))


