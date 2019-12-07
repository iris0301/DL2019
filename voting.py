import os
import sys
import tensorflow as tf
import numpy as np
from preprocess import *

from sklearn.metrics import precision_recall_fscore_support as score
from Attention_LSTM import *
from Attention import *
from CNN_LSTM import *
from CNN import *
from LSTM import *

import argparse

parser = argparse.ArgumentParser(description='Voting')
parser.add_argument('--mode', type=str, default='test',
                    help='Can be "train" or "test"')
args = parser.parse_args()

# tweets.db dictionary

def voting(inputs, model1, model2, model3, model4, model5):
    """
    Getting categorical predictions via voting
    """
    pred1 = model1.predict_classes(inputs)
    pred2 = model2.predict_classes(inputs)
    pred3 = model3.predict_classes(inputs)
    pred4 = model4.predict_classes(inputs)
    pred5 = model5.predict_classes(inputs)

    res = [0 for _ in range(pred1.shape[0])]
    for i in range(pred1.shape[0]):
        tmp_lis = [pred1, pred2, pred3, pred4, pred5]
        res[i] = np.argmax(np.bincount(tmp_lis))
    
    return tf.convert_to_tensor(res)

def test(inputs, y_true, model_LSTM, model_Attention_LSTM, model_CNN_LSTM, model_Attention, model_CNN):
    print("model_LSTM: ")
    res = model_LSTM.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_Attention_LSTM: ")
    res = model_Attention_LSTM.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_CNN_LSTM: ")
    res = model_CNN_LSTM.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_Attention: ")
    res = model_Attention.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
    print ("Precision: %f"%precision)
    print ("Recall: %f"%recall)
    print ("F1 score: %f"%f1)
    print ("\n")

    print("model_CNN: ")
    res = model_CNN.predict_classes(inputs)
    precision, recall, f1, _ = score(y_true, res, average='weighted')
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
    checkpoint = tf.train.Checkpoint(model_LSTM=model_LSTM, model_Attention_LSTM=model_Attention_LSTM,
                                    model_CNN_LSTM=model_CNN_LSTM, model_Attention=model_Attention, model_CNN=model_CNN)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)

    if args.mode == 'train':
        # Training
        model_LSTM.fit(X_train_id, y_train, batch_size=batch_size, 
                    epochs=epoch, verbose=1)
        manager.save()

        model_Attention_LSTM.fit(X_train_id, y_train, batch_size=batch_size, 
                    epochs=epoch, verbose=1)
        manager.save()

        model_CNN_LSTM.fit(X_train_id, y_train, batch_size=batch_size, 
                    epochs=epoch, verbose=1)
        manager.save()

        model_Attention.fit(X_train_id, y_train, batch_size=batch_size, 
                    epochs=epoch, verbose=1)
        manager.save()

        model_CNN.fit(X_train_id, y_train, batch_size=batch_size, 
                    epochs=epoch, verbose=1)
        manager.save()
    if args.mode == 'test':
        checkpoint.restore(manager.latest_checkpoint)
        test(X_test_id, y_test, model_LSTM, model_Attention_LSTM, model_CNN_LSTM, model_Attention, model_CNN)
    