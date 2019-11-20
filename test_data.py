#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:40:43 2019

@author: iris0301
"""

import numpy as np
from Data_processing import get_data
import math



def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
   
    print("Get_data")
    X_train_id, X_test_id, y_train, y_test, word_dict = get_data()
    print(len(X_train_id))
    print(X_train_id[:10])
    print(len(X_test_id))
    print(len(y_train))
    print(y_train[:10])
    print(len(y_test))
    print(len(word_dict))
    
    # TO-DO:  Separate your train and test data into inputs and labels
    

    
    
    
if __name__ == '__main__':
    main()
