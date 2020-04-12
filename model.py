from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle

def average(y_pred):
    for i in range(len(y_pred)):
        if i % len(y_pred) == 0 or (i+1) % len(y_pred) == 0:
            pass
        else:
            average = float(y_pred[i-1] +  y_pred[i] + y_pred[i+1])/3
            if average >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
    return y_pred

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
        pred_KN = average(pred_KN)
        acc_k = accuracy_score(y_test, pred_KN)
        if(acc_k > acc):
            best_k = k
            acc = acc_k
    return best_k

class Model():
    def train(self, X,y):
        y = y.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        best_k = calculate_best_k(X_train, X_test, y_train, y_test, range(1,30))
        self.model = KNeighborsClassifier(n_neighbors=best_k)
        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        preds = average(preds)
        probas = self.model.predict_proba(X_test)[:,1]
        f1 = metrics.f1_score(y_test, preds)
        roc = metrics.roc_auc_score(y_test, probas)
        print(' acc: {}\n f1 score: {}\n roc_auc score: {}'.format(acc, f1, roc))
        print('confusion matrix')
        print(confusion_matrix(y_test, preds))
        filename = 'data/models/knn.pkl'
        pickle.dump(self.model, open(filename, 'wb'))
