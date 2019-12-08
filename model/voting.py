import os
import sys
import tensorflow as tf
import numpy as np
from preprocess import *
import csv

from sklearn.metrics import precision_recall_fscore_support as score
from Attention_LSTM import *
from Attention import *
from CNN_LSTM import *
from CNN import *
from LSTM import *

import argparse

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

parser = argparse.ArgumentParser(description='Voting')
parser.add_argument('--mode', type=str, default='test',
                    help='Can be "train" or "test"')
parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')
args = parser.parse_args()

# tweets.db dictionary

def voting(inputs, model_LSTM, model_Attention_LSTM, model_CNN_LSTM, model_Attention, model_CNN):
    """
    Getting categorical predictions via voting
    """
    pred1 = model_LSTM.predict_classes(inputs)
    pred2 = model_Attention_LSTM.predict(inputs)
    pred2 = tf.argmax(pred2, 1)
    pred3 = model_CNN_LSTM.predict_classes(inputs)
    pred4 = model_Attention.predict(inputs)
    pred4 = tf.argmax(pred4, 1)
    pred5 = model_CNN.predict_classes(inputs)

    res = [0 for _ in range(pred1.shape[0])]
    for i in range(pred1.shape[0]):
        tmp_lis = [pred1[i], pred2[i], pred3[i], pred4[i], pred5[i]]
        res[i] = np.argmax(np.bincount(tmp_lis))
    
    return tf.convert_to_tensor(res)

def test(inputs, y_true, model_LSTM, model_Attention_LSTM, model_CNN_LSTM, model_Attention, model_CNN):
    print("model_LSTM: ")
    res = model_LSTM.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    m_acc = tf.keras.metrics.Accuracy()
    m_acc.update_state(res, y_true)
    print ("Accuracy: %f"%m_acc.result().numpy())
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_Attention_LSTM: ")
    res = model_Attention_LSTM.predict(inputs)
    res = tf.argmax(res, 1)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    m_acc = tf.keras.metrics.Accuracy()
    m_acc.update_state(res, y_true)
    print ("Accuracy: %f"%m_acc.result().numpy())
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_CNN_LSTM: ")
    res = model_CNN_LSTM.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    m_acc = tf.keras.metrics.Accuracy()
    m_acc.update_state(res, y_true)
    print ("Accuracy: %f"%m_acc.result().numpy())
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_Attention: ")
    res = model_Attention.predict(inputs)
    res = tf.argmax(res, 1)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    m_acc = tf.keras.metrics.Accuracy()
    m_acc.update_state(res, y_true)
    print ("Accuracy: %f"%m_acc.result().numpy())
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_CNN: ")
    res = model_CNN.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    m_acc = tf.keras.metrics.Accuracy()
    m_acc.update_state(res, y_true)
    print ("Accuracy: %f"%m_acc.result().numpy())
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")


if __name__ == '__main__':
    X_train_id, X_test_id, y_train, y_test, word_dict = get_data('data250k.csv')
    X_train_id = np.asarray(X_train_id)
    X_test_id = np.asarray(X_test_id)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # hyper parameters
    num_unit = 80
    num_window =  X_train_id.shape[1]
    vocab_size = len(word_dict)
    batch_size = 120
    epoch = 1
    learning_rate = 0.001

    optimizer = tf.keras.optimizers.Adam(learning_rate) # optimizer

    # Models
    model_LSTM = get_LSTM_model(num_unit, num_window, vocab_size+1)
    
    model_Attention_LSTM = Attention_LSTM_Model(num_unit, num_window, vocab_size+1, 5) # 5 heads
    model_Attention_LSTM.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    model_CNN_LSTM = get_CNNLSTM_model(num_unit, num_window, vocab_size+1)

    model_Attention = Attention_Model(num_unit, num_window, vocab_size+1, 5)
    model_Attention.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)

    model_CNN = get_CNN_model(num_unit, num_window, vocab_size+1)


    # model saving management
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    def save():
        checkpoint = tf.train.Checkpoint(model_LSTM=model_LSTM, model_Attention_LSTM=model_Attention_LSTM,
                                        model_CNN_LSTM=model_CNN_LSTM, model_Attention=model_Attention, model_CNN=model_CNN)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        manager.save()

    if args.mode == 'train':
        # Training
        with tf.device('/device:' + args.device):
            model_LSTM.fit(X_train_id, y_train, batch_size=batch_size, 
                        epochs=epoch, verbose=1)
            save()

            model_Attention_LSTM.fit(X_train_id, y_train, batch_size=batch_size, 
                        epochs=epoch, verbose=1)
            save()

            model_CNN_LSTM.fit(X_train_id, y_train, batch_size=batch_size, 
                        epochs=epoch, verbose=1)
            save()

            model_Attention.fit(X_train_id, y_train, batch_size=batch_size, 
                        epochs=epoch, verbose=1)
            save()

            model_CNN.fit(X_train_id, y_train, batch_size=batch_size, 
                        epochs=epoch, verbose=1)
            save()

    if args.mode == 'test':
        checkpoint = tf.train.Checkpoint(model_LSTM=model_LSTM, model_Attention_LSTM=model_Attention_LSTM,
                                        model_CNN_LSTM=model_CNN_LSTM, model_Attention=model_Attention, model_CNN=model_CNN)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        checkpoint.restore(manager.latest_checkpoint)
        test(X_test_id, y_test, model_LSTM, model_Attention_LSTM, model_CNN_LSTM, model_Attention, model_CNN)
    
    if args.mode == 'trump':
        checkpoint = tf.train.Checkpoint(model_LSTM=model_LSTM, model_Attention_LSTM=model_Attention_LSTM,
                                        model_CNN_LSTM=model_CNN_LSTM, model_Attention=model_Attention, model_CNN=model_CNN)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
        checkpoint.restore(manager.latest_checkpoint)

        x = get_trump(word_dict)
        
        res = voting(x, model_LSTM, model_Attention_LSTM, model_CNN_LSTM, model_Attention, model_CNN)
        with open('trump_output.csv', 'w', newline='') as csvfile:
            fieldnames = ['Text', 'Sentiment']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(res)):
                output_sentiment = tf.dtypes.cast(res[i], tf.int32)
                output_text = ''
                for ele in x[i]:
                    if ele == 0:
                        break
                    output_text = output_text+' '+ list(word_dict.keys())[list(word_dict.values()).index(ele)]
                writer.writerow({'Text': output_text, 'Sentiment': output_sentiment})
        
        pos = tf.reduce_sum(tf.cast(tf.equal(res, 2), tf.int32))
        neu = tf.reduce_sum(tf.cast(tf.equal(res, 1), tf.int32))
        neg = tf.reduce_sum(tf.cast(tf.equal(res, 0), tf.int32))
        print ("For tweets about Donald Trump:")
        print ("Positive: %f"%(pos/(pos+neu+neg)))
        print ("Neutral: %f"%(neu/(pos+neu+neg)))
        print ("Negative: %f"%(neg/(pos+neu+neg)))

    