from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np
import os
from utils import normalize
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout
from keras.layers import LSTM, ConvLSTM2D, Conv2D
from keras.regularizers import l2
from keras.models import load_model
from keras.optimizers import Adam


class Model():
    def __init__(self, n_features, mode, size, step, n_samples, type_model):
        self.n_features = n_features
        self.mode = mode
        self.size = size
        self.step = step
        self.n_samples = (n_samples - size)/step
        self.type_model = type_model

    def balanced_split(self, X, y, idx, n):
        idxs_train = set(np.arange(len(X)))
        idxs_test = set(np.arange(idx*2*self.n_samples, (idx+n)*2*self.n_samples))
        idxs_train = np.array(list(idxs_train - idxs_test), dtype=np.int32)
        idxs_test = np.array(list(idxs_test), dtype=np.int32)

        X_train = X[idxs_train]
        y_train = y[idxs_train]
        X_test = X[idxs_test]
        y_test = y[idxs_test]

        return X_train, X_test, y_train, y_test

    def train_lstm(self, X, y):
        n_test = 1
        n_people = int(len(X)/(2*self.n_samples))
        print("n people:", n_people)
        acc_global = 0
        f1_global = 0
        roc_global = 0
        
        batch_size=1
        for idx in range(n_people):
        # for idx in range(0):
            model_filename = 'data/models/model_' + str(self.n_features) + '_' + str(self.size) + '_' + str(self.step) + '_' + str(len(X)) + '_'+self.type_model +'_'+ self.mode + '_' + str(n_test) + '_' + str(idx) + '_normalized.h5'
            X_train, X_test, y_train, y_test = self.balanced_split(X, y, idx, n_test)
            if(os.path.isfile(model_filename)):
                model = load_model(model_filename)
            else:
                model = Sequential()
                model.add(ConvLSTM2D(filters=40, kernel_size=(5, 5), strides=(3,3), input_shape=(None, 64, 64, 1), padding='valid', bias_regularizer=l2(1e-3), return_sequences=False, dropout=0.0, kernel_regularizer=l2(1e-3), recurrent_regularizer=l2(1e-3)))
                #self.model.add(Dropout(0.5))
                model.add(Flatten())
                model.add(Dense(1, bias_regularizer=l2(1e-3), kernel_regularizer=l2(1e-3)))
                model.add(Activation('sigmoid'))
                
                opt = Adam(learning_rate=0.00001)
                model.compile(loss='binary_crossentropy', optimizer=opt)
                
                tmp_x = []
                tmp_y = []
                for i in range((n_people-n_test)*2):
                    tmp_x.append(X_train[int(i*self.n_samples):int((i+1)*self.n_samples)])
                    tmp_y.append(y_train[int(i*self.n_samples):int((i+1)*self.n_samples)])
                loss_m2 = np.inf
                loss_m1 = np.inf
                while(True):
                    i = np.random.randint(self.n_samples-batch_size)
                    batch_x = []
                    batch_y = []
                    for cx, cy in zip(tmp_x, tmp_y):
                        batch_x.extend(cx[int(i%self.n_samples):int((i+batch_size)%self.n_samples)])
                        batch_y.extend(cy[int(i%self.n_samples):int((i+batch_size)%self.n_samples)])
                    batch_x = np.array(batch_x)
                    batch_y = np.array(batch_y)
                    model.train_on_batch(batch_x, batch_y)
                    loss = model.test_on_batch(X_test, y_test)
                    print(loss)
                    if(loss > loss_m1 and loss_m1>loss_m2):
                        break
                    loss_m2 = loss_m1
                    loss_m1 = loss
                
                # self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=50, verbose=2, steps_per_epoch=10, validation_steps=1)

                model.save(model_filename)

            preds = model.predict_classes(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = metrics.f1_score(y_test, preds)
            probas = model.predict(X_test)
            roc = metrics.roc_auc_score(y_test, probas)

            acc_global += acc
            f1_global += f1
            roc_global += roc

            print(idx, acc, f1, roc)
        acc_global/=n_people
        f1_global/=n_people
        roc_global/=n_people
        print("acc global:",acc_global)
        print("f1 global:",f1_global)
        print("roc global:", roc_global)

        
        
        
        ############## ************* train final model **************** ################

        n_test = 9
        idx = np.random.randint(n_people-n_test)
        batch_size=2
        model_filename = 'data/models/model_' + str(self.n_features) + '_' + str(self.size) + '_' + str(self.step) + '_' + str(len(X)) + '_'+self.type_model +'_'+ self.mode + '_' + str(n_test) + '_' + str(idx) + '_normalized.h5'
        X_train, X_test, y_train, y_test = self.balanced_split(X, y, idx, n_test)
        self.model = Sequential()
        self.model.add(ConvLSTM2D(filters=40, kernel_size=(5, 5), strides=(3,3), input_shape=(None, 64, 64, 1), padding='valid', bias_regularizer=l2(1e-3), return_sequences=False, dropout=0.0, kernel_regularizer=l2(1e-3), recurrent_regularizer=l2(1e-3)))
        #self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(1, bias_regularizer=l2(1e-3), kernel_regularizer=l2(1e-3)))
        self.model.add(Activation('sigmoid'))
        
        opt = Adam(learning_rate=0.00001)
        self.model.compile(loss='binary_crossentropy', optimizer=opt)
        
        tmp_x = []
        tmp_y = []
        for i in range((n_people-n_test)*2):
            tmp_x.append(X_train[int(i*self.n_samples):int((i+1)*self.n_samples)])
            tmp_y.append(y_train[int(i*self.n_samples):int((i+1)*self.n_samples)])
        loss_m2 = np.inf
        loss_m1 = np.inf
        while(True):
            i = np.random.randint(self.n_samples-batch_size)
            batch_x = []
            batch_y = []
            for cx, cy in zip(tmp_x, tmp_y):
                batch_x.extend(cx[int(i%self.n_samples):int((i+batch_size)%self.n_samples)])
                batch_y.extend(cy[int(i%self.n_samples):int((i+batch_size)%self.n_samples)])
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y)
            self.model.train_on_batch(batch_x, batch_y)
            loss = self.model.test_on_batch(X_test, y_test)
            print(loss)
            if(loss > loss_m1 and loss_m1>loss_m2):
                break
            loss_m2 = loss_m1
            loss_m1 = loss
        
        self.model.save(model_filename)
        preds = self.model.predict_classes(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = metrics.f1_score(y_test, preds)
        probas = self.model.predict(X_test)
        roc = metrics.roc_auc_score(y_test, probas)

        print('final model:')
        print('acc:', acc)
        print('f1:', f1)
        print('roc', roc)