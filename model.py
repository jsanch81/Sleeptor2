from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.cluster import KMeans, MeanShift
import pickle
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Flatten
from keras.layers import LSTM, ConvLSTM2D, Conv2D
from keras.regularizers import l2

def normalize(data):
    means = np.mean(data, axis=1).reshape(-1,1)
    stds = np.std(data, axis=1).reshape(-1,1)
    data_s = (data-means)/stds
    return data_s


class Model():
    def __init__(self, n_features, mode, size, n_samples, type_model):
        self.n_features = n_features
        self.mode = mode
        self.size = size
        self.n_samples = n_samples - size
        self.type_model = type_model

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
        # X = normalize(X)
        y = y.ravel()
        print(len(X))

        n_people = int(len(X)/(2*self.n_samples))
        print("n people:", n_people)
        acc_global = 0
        f1_global = 0
        roc_global = 0
        for idx in range(n_people):
            model_filename = 'data/models/model_' + str(self.n_features) + '_' + str(self.size) + '_' + str(len(X)) + '_'+self.type_model +'_'+ self.mode + '_' + str(idx) + '.pkl'
            X_train, X_test, y_train, y_test = self.balanced_split(X, y, idx)
            print(X_train.shape)
            if(os.path.isfile(model_filename)):
                self.model = pickle.load(open(model_filename, 'rb'))
                #self.model.fit(X_train, y_train)
                #pickle.dump(self.model, open(model_filename, 'wb'))
            else:
                if self.type_model == 'logisticRegression':
                    self.model = LogisticRegression(random_state=0)
                elif(self.type_model == 'lstm'):
                        self.model = Sequential()
                        self.model.add(LSTM(5, input_shape=(self.size, self.n_features),return_sequences=True))
                        self.model.add(LSTM(5, input_shape=(self.size, self.n_features)))
                        self.model.add(Dense(1))
                        self.model.add(Activation('sigmoid'))
                        self.model.compile(loss='binary_crossentropy', optimizer='adam')
                        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=5000, verbose=2)
                else:
                    #self.model = RandomForestClassifier()
                    #self.model = SVC(probability=True, kernel='sigmoid')
                    #self.model = KMedoids(n_clusters=2)
                    # self.model = AdaBoostClassifier()
                    self.model = MeanShift()
                    self.model.fit(X_train, y_train)
                    pickle.dump(self.model, open(model_filename, 'wb'))

            preds = self.model.predict_classes(X_test)
            acc = accuracy_score(y_test, preds)
            #probas = self.model.predict_proba(X_test)[:,1]
            f1 = metrics.f1_score(y_test, preds)
            #roc = metrics.roc_auc_score(y_test, probas)

            print('for idx:', idx)
            #print(' acc: {}\n f1 score: {}\n roc_auc score: {}'.format(acc, f1, roc))
            print(' acc: {}\n f1 score: {}'.format(acc, f1))
            print('confusion matrix')
            print(confusion_matrix(y_test, preds))

            acc_global += acc
            f1_global += f1
            #roc_global += roc

        acc_global /= n_people
        f1_global /= n_people
        #roc_global /= n_people
        print('\nglobal:')
        #print(' acc: {}\n f1 score: {}\n roc_auc score: {}'.format(acc_global, f1_global, roc_global))
        print(' acc: {}\n f1 score: {}\n'.format(acc_global, f1_global))

    def train_lstm(self, X, y):
        X_train, X_test, y_train, y_test = self.balanced_split(X, y, 0)
        self.model = Sequential()
        self.model.add(ConvLSTM2D(filters=40, kernel_size=(5, 5), strides=(3,3), input_shape=(None, 64, 48, 1), padding='valid', bias_regularizer=l2(1e-4), return_sequences=False))
        #self.model.add(Conv2D(filters=30, kernel_size=(5, 5), strides=(3,3), input_shape=(None, 22, 16), padding='same'))
        #self.model.add(Conv2D(filters=20, kernel_size=(5, 5), strides=(2,2), input_shape=(None, 8, 6), padding='same'))
        #self.model.add(Conv2D(filters=15, kernel_size=(5, 5), strides=(2,2), input_shape=(None, 16, 12), padding='same'))
        #self.model.add(Conv2D(filters=10, kernel_size=(5, 5), strides=(2,2), input_shape=(None, 8, 6), padding='same'))
        #self.model.add(Conv2D(filters=5, kernel_size=(5, 5), strides=(2,2), input_shape=(None, 4, 3), padding='same'))
        # for layer in self.model.layers:
        #     print(layer.output_shape)

        self.model.add(Flatten())
        self.model.add(Dense(1, bias_regularizer=l2(1e-4)))
        self.model.add(Activation('sigmoid'))
        # print(self.model.count_params())
        # print(get_model_memory_usage(40, self.model))

        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=40, verbose=2)

        preds = self.model.predict_classes(X_test)
        acc = accuracy_score(y_test, preds)
        #probas = self.model.predict_proba(X_test)[:,1]
        f1 = metrics.f1_score(y_test, preds)

        print(acc, f1)

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes