import os
import sys
import tensorflow as tf
import numpy as np
from preprocess import *

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Flatten, Attention, Concatenate
from tensorflow.keras.losses import CategoricalCrossentropy

# tweets.db dictionary

class Attention_LSTM_Model(tf.keras.Model):
    def __init__(self, num_unit, num_window, vocab_size, head_num):
        super(Attention_LSTM_Model, self).__init__()
        self.num_unit = num_unit # dimension of LSTM output
        self.num_window = num_window # window size
        self.vocab_size = vocab_size # vocabulary size
        self.head_num = head_num # number of attention heads

        self.embedding = Embedding(input_dim=vocab_size, output_dim=num_unit, input_length=num_window)
        self.att_layer = [Attention() for _ in range(head_num)] # multiple attention layers
        self.LSTM_layer = LSTM(num_unit)
        self.bidirectional_layer = Bidirectional(self.LSTM_layer) # bidirectional LSTM layer
        self.dense_1 = Dense(150, activation='relu') # dense layer
        self.dense_2 = Dense(3, activation='softmax')
    
    def call(self, inputs):
        emb = self.embedding(inputs)

        att_list = []
        for i in range(self.head_num):
            att = self.att_layer[i]([emb, emb, emb])
            att_list.append(att)
        attention = tf.concat(att_list, 2) # concatenate all the attention heads

        lstm = self.bidirectional_layer(attention)

        logits = self.dense_1(lstm)
        prbs = self.dense_2(logits)
        return prbs


if __name__ == '__main__':
    # get data
    X_train_id, X_test_id, y_train, y_test, word_dict = get_data('smalldata.csv')

    X_train_id = np.asarray(X_train_id)
    X_test_id = np.asarray(X_test_id)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # hyper parameters
    num_unit = 50
    num_window =  X_train_id.shape[1]
    vocab_size = len(word_dict)

    # get model
    model = Attention_LSTM_Model(num_unit, num_window, vocab_size+1, 5)
    optimizer = tf.keras.optimizers.Adam(0.01) # optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    # train
    model.fit(X_train_id, y_train, batch_size=100, 
                epochs=1, verbose=1)
    res = model.predict(X_test_id)
    res = tf.argmax(res, 1)

    # metrics
    m_precision = tf.keras.metrics.Precision()
    m_precision.update_state(y_test, res)
    m_recall = tf.keras.metrics.Recall()
    m_recall.update_state(y_test, res)
    print (m_precision.result().numpy())
    print (m_recall.result().numpy())
    m_F1 = (2*m_precision*m_recall) / (m_precision + m_recall)
    print (m_F1)

