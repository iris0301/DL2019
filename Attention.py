import os
import sys
import tensorflow as tf
import numpy as np
from preprocess import *

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, Flatten, Attention
from tensorflow.keras.losses import CategoricalCrossentropy
from keras_multi_head import MultiHeadAttention

# tweets.db dictionary

class Attention_Model(tf.keras.Model):
    def __init__(self, num_unit, num_window, vocab_size, head_num):
        super(Attention_Model, self).__init__()
        self.num_unit = num_unit
        self.num_window = num_window
        self.vocab_size = vocab_size
        self.head_num = head_num

        self.embedding = Embedding(input_dim=vocab_size, output_dim=num_unit, input_length=num_window)
        self.att_layer = [Attention() for _ in range(head_num)]
        self.flatten = Flatten()
        self.dense_1 = Dense(150, activation='relu')
        self.dense_2 = Dense(3, activation='softmax')
    
    @tf.function
    def call(self, inputs):
        emb = self.embedding(inputs)

        att_list = []
        for i in range(self.head_num):
            att = self.att_layer[i]([emb, emb, emb])
            att_list.append(att)
        attention = tf.concat(att_list, 2)

        flat = self.flatten(attention)
        logits = self.dense_1(flat)
        prbs = self.dense_2(logits)
        return prbs


if __name__ == '__main__':
    X_train_id, X_test_id, y_train, y_test, word_dict = get_data('data1.csv')

    X_train_id = np.asarray(X_train_id)
    X_test_id = np.asarray(X_test_id)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    num_unit = 50
    num_window =  X_train_id.shape[1]
    vocab_size = len(word_dict)

    model = Attention_Model(num_unit, num_window, vocab_size+1, 5)
    optimizer = tf.keras.optimizers.Adam(0.01) # optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    model.fit(X_train_id, y_train, batch_size=20, 
                epochs=1, verbose=1)
    res = model.predict(X_test_id)
    res = tf.argmax(res, 1)

    m_acc = tf.keras.metrics.Accuracy()
    m_acc.update_state(res, y_test)
    print (m_acc.result().numpy())

