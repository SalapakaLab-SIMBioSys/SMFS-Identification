#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 19:36:04 2025

@author: cailonghua
"""

# Step 1: Imports
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../')

from APIs.utils import utils
from APIs.PemNN import PemNN
from APIs.PemNN import Classifier_PemNN


# Step 2: Initialize the toolbox
UtObj = utils()
PemNN = PemNN()

# Step 3: Read data
# Choose moelcule from Titin', 'UtrNR3', 'UtrNR3_bact', 'DysNR3_bact', 
molecule = 'Titin'
df_save_path = '../Data/ML_Dataset/' + molecule +'/' # data save path

file_name = 'Fu_' + molecule + '_sim'
Fu_data_df =  pd.read_pickle(df_save_path + file_name + '_data' + '.csv', )

file_name = 'xp_' + molecule + '_sim'
xp_data_df =  pd.read_pickle(df_save_path + file_name + '_data' + '.csv', )

# model save path
output_directory = 'ML_models/' + molecule + '/saved_model_physics/'
os.makedirs(output_directory, exist_ok=True)

    
# Step 4: Pre-process and train/test split
[
 x_train_phy, x_test_phy, # physical branch
 x_train, x_test,  # non-physical branch
 y_train, y_test, # label
 y_train_oh, y_test_oh, # one-hot encoding label
 ] = PemNN.pre_process(Fu_data_df, xp_data_df, test_size = 0.8,) 


# Step 5: Initialize PemNN classifier 
nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0))) # No. of classes
clf = Classifier_PemNN(input_shape_pairs = x_train_phy.shape[1:], 
                        input_shape_traces = x_train.shape[1:],
                        nb_classes = nb_classes,
                        output_directory = output_directory,
                        )

# Step 6: Train PemNN
clf.fit([x_train_phy, x_train], y_train_oh,
        batch_size = 16,
        nb_epochs = 2,
        diagonistic = False)


# Step 7: Test PemNN and print metric
[acc, f1, roc_auc, _] = clf.predict([x_test_phy, x_test],y_test)
print('The accuracy is : ' + str(round(acc,4)))

















