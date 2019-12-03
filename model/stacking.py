import os
import sys
import tensorflow as tf
import numpy as np
from preprocess import *
from sklearn.model_selection import KFold
import pandas as pd

from LSTM import *
from CNN import *
from CNN_LSTM import *

class Stacking():
    def __init__(self, n_folds, base_models):
        self.n_folds = n_folds          # number of folds
        self.base_models = base_models  # list of models

    def get_stacking_data(self, X_train_id, y_train):
        folds = list(KFold(n_splits=self.n_folds, shuffle=True, random_state=0).split(X_train_id))
        S_train = np.zeros((X_train_id.shape[0], 3*len(self.base_models)))
        S_test = []
        for i, clf in enumerate(self.base_models):
            for j, (train_idx, test_idx) in enumerate(folds):
                # for training on this fold
                X_train = X_train_id[train_idx]
                Y_train = y_train[train_idx]
                # for test in this fold
                X_holdout = X_train_id[test_idx]
                Y_holdout = y_train[test_idx]

                clf.fit(X_train, Y_train, batch_size=20, epochs=1, verbose=1)
                y_pred = clf.predict(X_holdout)
                print (y_pred.shape)
                S_train[test_idx, i*3:i*3+3] = y_pred
                S_test += Y_holdout.tolist()
        return S_train, S_test

def train_meta_learner(model, S_train, S_test):
    pass

if __name__ == '__main__':
    X_train_id, X_test_id, y_train, y_test, word_dict = get_data()

    X_train_id = np.asarray(X_train_id)
    X_test_id = np.asarray(X_test_id)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    num_unit = 50
    num_window =  X_train_id.shape[1]
    vocab_size = len(word_dict)

    model_LSTM = get_LSTM_model(num_unit, num_window, vocab_size+1)
    model_CNN = get_CNN_model(num_unit, num_window, vocab_size+1)
    model_CNN_LSTM = get_CNNLSTM_model(num_unit, num_window, vocab_size+1)

    stacking = Stacking(5, [model_LSTM, model_CNN, model_CNN_LSTM])

    S_train, S_test = stacking.get_stacking_data(X_train_id, y_train)
    print (S_train)
