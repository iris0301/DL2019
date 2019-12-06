import os
import sys
import tensorflow as tf
import numpy as np
from preprocess import *

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Conv1D, MaxPooling1D,GlobalMaxPooling1D
from tensorflow.keras.losses import CategoricalCrossentropy

# tweets.db dictionary

def get_CNN_model(num_unit, num_window, vocab_size):
    """
    Return the keras LSTM model for tweet sentiment analysis
    :param num_unit: dimensionality of the output space of LSTM
    :param num_window: the length of each padded tweet
    :param vocab_size: the number of vocabularies
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=num_unit, input_length=num_window))
    model.add(Conv1D(filters=64, kernel_size=5, padding='causal', activation='relu', strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(0.01) # optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    return model


if __name__ == '__main__':
    X_train_id, X_test_id, y_train, y_test, word_dict = get_data('data1.csv')

    X_train_id = np.asarray(X_train_id)
    X_test_id = np.asarray(X_test_id)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    num_unit = 50
    num_window =  X_train_id.shape[1]
    vocab_size = len(word_dict)

    model = get_CNN_model(num_unit, num_window, vocab_size+1)
    model.fit(X_train_id, y_train, batch_size=20, 
                epochs=5, verbose=1)
    res = model.predict_classes(X_test_id)

    m_acc = tf.keras.metrics.Accuracy()
    m_acc.update_state(res, y_test)
    print (m_acc.result().numpy())

