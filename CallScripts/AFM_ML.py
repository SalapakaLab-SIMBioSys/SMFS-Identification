# =============================================================================
# Description
# =============================================================================

# Call script for detecting single molecule curves with deep learning models
# =============================================================================

# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import sys
from matplotlib import pyplot as plt
import pandas as pd
import os
from datetime import datetime

# sklearn
from sklearn.model_selection import train_test_split
import sklearn as sk

# tensorflow
import tensorflow.keras as keras
import tensorflow as tf
import random

#
# sktime
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.deep_learning import InceptionTimeClassifier
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier  
from sktime.classification.deep_learning.fcn import FCNClassifier

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.append('../')

from APIs.utils import utils
from APIs.PemNN import PemNN
from APIs.PemNN import Classifier_PemNN
#%%
################################################
#             Global parameters
################################################
UtObj = utils()
PemNN = PemNN()

# Seed for traning_test datasets
random_seed = random.randint(0, 1000000) # random seeds
print('The current seed = ' + str(random_seed))
molecule = 'Titin' # Choose from 'Titin', 'UtrNR3', 'UtrNR3_bact', 'DysNR3_bact', 

# Polymer elastic model params
thres = 1000 # countour length threshold
Lp = 300 # Persistence length
Lk = 300 # Kuhn length
Lb = 150 # bond lenght

# Check the gpu status
print("Num Devices Available: ", tf.config.list_physical_devices())


# data setups
build_exp_dataset = False # read exp data that consisting of unfolding events
build_sim_dataset = False # read sim data 
use_exp_or_sim = False # True with exp data, False with sim data
visualization = False
verbose = False # verbose for training status

test_size = 0.8
no_of_pairs = 400 # No. of pairs (100, 200, 400, 800,...)
resampling = True
resample_size = 400  # resample size (100, 200, 400, 800,...)
minmax_normalization = True


# paths
df_save_path = '../Data/ML_Dataset/' + molecule +'/' # dataset save path
os.makedirs(df_save_path, exist_ok=True)

data_save_path = 'ML_models/' + molecule + '/'
os.makedirs(data_save_path, exist_ok=True)

output_directory = 'ML_models/' + molecule + '/saved_model_physics/' # model save path
os.makedirs(output_directory, exist_ok=True)

fig_save_path = output_directory + 'plots/'
os.makedirs(fig_save_path, exist_ok=True)

results_save_path = output_directory + 'results/'
os.makedirs(results_save_path, exist_ok=True)

#%%
################################################
#        Build (Fu,xp) from raw data
################################################
if (build_exp_dataset):
    [Fu_data_df, xp_data_df] = UtObj.get_Fu_xp_exp(molecule, df_save_path)
    
    file_name = 'Fu_' + molecule + '_exp'
    Fu_data_df.to_pickle(df_save_path + file_name + '_data.csv',)
    
    file_name = 'xp_' + molecule + '_exp'
    xp_data_df.to_pickle(df_save_path + file_name + '_data.csv',)
    
if (build_sim_dataset):
    [Fu_data_df, xp_data_df] = UtObj.get_Fu_xp_sim(molecule, df_save_path)

    file_name = 'Fu_' + molecule + '_sim'
    Fu_data_df.to_pickle(df_save_path + file_name + '_data.csv',)
    
    file_name = 'xp_' + molecule + '_sim'
    xp_data_df.to_pickle(df_save_path + file_name + '_data.csv',)
#%%
################################################
#        Load saved (Fu,xp) data
################################################
if (use_exp_or_sim):
    file_name = 'Fu_' + molecule + '_exp'
    Fu_data_df =  pd.read_pickle(df_save_path + file_name + '_data' + '.csv', )

    file_name = 'xp_' + molecule + '_exp'
    xp_data_df =  pd.read_pickle(df_save_path + file_name + '_data' + '.csv', )
else:
    file_name = 'Fu_' + molecule + '_sim'
    Fu_data_df =  pd.read_pickle(df_save_path + file_name + '_data' + '.csv', )

    file_name = 'xp_' + molecule + '_sim'
    xp_data_df =  pd.read_pickle(df_save_path + file_name + '_data' + '.csv', )

        
#%%
################################################
#            Build train test data 
################################################
[Fu_train_exp, xp_train_exp, 
 Fu_train_exp_norm, xp_train_exp_norm, 
 y_train_exp] = UtObj.get_trace_pair_data(Fu_data_df, xp_data_df, 
                         no_of_pairs, resample_size,
                         molecule, resampling, )
                                          
[Fu_train_exp, Fu_test_exp, 
  xp_train_exp, xp_test_exp,
  y_train_exp, y_test_exp,
  Fu_train_exp_norm, Fu_test_exp_norm, 
  xp_train_exp_norm, xp_test_exp_norm,] = train_test_split(Fu_train_exp, 
                                              xp_train_exp, 
                                              y_train_exp, 
                                              Fu_train_exp_norm,
                                              xp_train_exp_norm,
                                              test_size = test_size, 
                                              random_state=random_seed,
                                              stratify=y_train_exp)
    
enc = sk.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train_exp, y_test_exp), axis=0).reshape(-1, 1))

y_train_exp_oh = enc.transform(y_train_exp.reshape(-1, 1)).toarray()
y_test_exp_oh = enc.transform(y_test_exp.reshape(-1, 1)).toarray()

print('The length of experimental training data = ' + str(len(y_train_exp)))
print('The length of experimental testing data = ' + str(len(y_test_exp)))

#%%
################################################
#   Baselines with Fu trace channel only
################################################
# choose data to use when training models
if (len(Fu_train_exp_norm.shape)<3):
    x_train_nn = Fu_train_exp_norm.reshape((Fu_train_exp_norm.shape[0],Fu_train_exp_norm.shape[1], 1))
else:
    x_train_nn = Fu_train_exp_norm
x_train_nn = x_train_nn.transpose(0,2,1) # sktime uses (n_instances (n), n_dimensions (d), series_length (m)) 

if (len(Fu_test_exp_norm.shape)<3):
    x_test_nn = Fu_test_exp_norm.reshape((Fu_test_exp_norm.shape[0],Fu_test_exp_norm.shape[1], 1))
else:
    x_test_nn = Fu_test_exp_norm
x_test_nn = x_test_nn.transpose(0,2,1) # sktime uses (n_instances (n), n_dimensions (d), series_length (m)) 

y_train_nn = y_train_exp
y_test_nn = y_test_exp

#%%
# ResNet
n_epochs = 200
batch_size = 16


# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
  min_lr=0.0001)

file_name = 'ResNet_traces_Fu'
file_path = output_directory + file_name + '_best.keras'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
  save_best_only=True)

callbacks = [reduce_lr,model_checkpoint]

print('Training with' + str(file_name))
clf = ResNetClassifier(n_epochs=n_epochs, 
                       batch_size = batch_size, 
                       verbose = verbose, 
                       loss='categorical_crossentropy',
                       metrics=['accuracy'],
                       optimizer = keras.optimizers.Adam(),
                       callbacks = callbacks,
                       activation='softmax'
                       ) 
# train
clf.fit(x_train_nn, y_train_nn) 

# test
PemNN.test_models(clf, x_test_nn, y_test_nn, test_size, batch_size, n_epochs,
                random_seed, no_of_pairs, resample_size, molecule,
                file_path, results_save_path, file_name, 
                use_exp_or_sim, 
                )

#%%
# InceptionTime
n_epochs = 200
batch_size = 64 #16


# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
  min_lr=0.0001)

file_name = 'InceptionTime_traces_Fu'
file_path = output_directory + file_name + '_best.keras'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
  save_best_only=True)

callbacks = [reduce_lr,model_checkpoint]
print('Training with' + str(file_name))
clf = InceptionTimeClassifier(n_epochs=n_epochs, 
                            batch_size = batch_size, 
                            verbose = verbose, 
                            loss='categorical_crossentropy',
                            metrics=['accuracy'],
                            callbacks = callbacks,
                            ) 
# train
clf.fit(x_train_nn, y_train_nn) 

# test
PemNN.test_models(clf, x_test_nn, y_test_nn, test_size, batch_size, n_epochs,
                random_seed, no_of_pairs, resample_size, molecule,
                file_path, results_save_path, file_name, 
                use_exp_or_sim, 
                )


#%%
# FCN
n_epochs = 200
batch_size = 16


# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
  min_lr=0.0001)

file_name = 'FCN_traces_Fu'
file_path = output_directory + file_name + '_best.keras'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
  save_best_only=True)

callbacks = [reduce_lr,model_checkpoint]
print('Training with' + str(file_name))
clf = FCNClassifier(n_epochs=n_epochs, 
                    batch_size = batch_size, 
                    verbose = verbose, 
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    optimizer = keras.optimizers.Adam(),
                    callbacks = callbacks,
                    activation='softmax'
                    ) 
# train
clf.fit(x_train_nn, y_train_nn) 

# test
PemNN.test_models(clf, x_test_nn, y_test_nn, test_size, batch_size, n_epochs,
                random_seed, no_of_pairs, resample_size, molecule,
                file_path, results_save_path, file_name, 
                use_exp_or_sim, 
                )

#%%
# LSTMFCN
n_epochs = 200
batch_size = 16


# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
  min_lr=0.0001)

file_name = 'LSTMFCN_traces_Fu'
file_path = output_directory + file_name + '_best.keras'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
  save_best_only=True)

callbacks = [reduce_lr,model_checkpoint]
print('Training with' + str(file_name))
clf = LSTMFCNClassifier(n_epochs=n_epochs, 
                    batch_size = batch_size, 
                    verbose = verbose, 
                    callbacks = callbacks,
                    ) 
# train
clf.fit(x_train_nn, y_train_nn) 
    
# test
PemNN.test_models(clf, x_test_nn, y_test_nn, test_size, batch_size, n_epochs,
                random_seed, no_of_pairs, resample_size, molecule,
                file_path, results_save_path, file_name, 
                use_exp_or_sim, 
                )

#%%
# dimension swap into [batch, time length, channels]
if(x_train_nn.shape[1]<x_train_nn.shape[2]):
    x_train_nn = x_train_nn.transpose(0,2,1)
    x_test_nn = x_test_nn.transpose(0,2,1)
#%%
# Triplet Network
#  Build dataset for Triplet network
X_train = x_train_nn
X_test = x_test_nn
y_train = y_train_nn
y_test = y_test_nn

X_train_pos_all = np.empty((0,X_train.shape[1],X_train.shape[2]), int)
X_train_neg_all = np.empty((0,X_train.shape[1],X_train.shape[2]), int)

y_train_pos_all = []
y_train_neg_all = []
for ii in range(len(y_train)):
    y_pos_chk = (y_train == y_train[ii])
    y_neg_chk =np.logical_not(y_pos_chk)
    
    X_train_pos = X_train[y_pos_chk]
    X_train_neg = X_train[y_neg_chk]
    
    pos_random_choice = np.random.choice(X_train_pos.shape[0], size = 1)
    neg_random_choice = np.random.choice(X_train_neg.shape[0], size = 1)
    
    X_train_pos_sel = X_train_pos[pos_random_choice,:,:]
    X_train_neg_sel = X_train_neg[neg_random_choice,:,:]
    
    X_train_pos_all = np.vstack((X_train_pos_all, X_train_pos_sel))
    X_train_neg_all = np.vstack((X_train_neg_all, X_train_neg_sel))
    
    y_train_pos_all.append(y_train[pos_random_choice])
    y_train_neg_all.append(y_train[neg_random_choice])

y_train_pos_all = np.array(y_train_pos_all)
y_train_neg_all = np.array(y_train_neg_all)


y_train_oh = enc.transform(y_train.reshape(-1, 1)).toarray()
y_train_pos_all_oh = enc.transform(y_train_pos_all.reshape(-1, 1)).toarray()
y_train_neg_all_oh = enc.transform(y_train_neg_all.reshape(-1, 1)).toarray()

# Build model with keras
input_shape = X_train.shape[1:]
nb_classes = len(np.unique(np.concatenate((y_train, y_test_nn), axis=0)))

# file path to save model
file_name = 'Triplet'
file_path = output_directory + file_name + '_last.keras'

# create model
[embed_model, model] = PemNN.create_Triplet(input_shape)


if (visualization):
    # Check triplet dataset
    plt.figure()
    check_no = np.random.choice(X_train.shape[0], size = 1)[0]
    plt.plot(X_train[check_no,:,0], label = 'anchor')
    plt.plot(X_train_pos_all[check_no,:,0], label = 'pos')
    plt.plot(X_train_neg_all[check_no,:,0], label = 'neg')
    plt.legend()
    
    plt.savefig(fname = fig_save_path + 'visual_Triplet_dataset' + '.png')
    plt.savefig(fname = fig_save_path + 'visual_Triplet_dataset' + '.svg') 
 



# Train Triplet
batch_size = 16
nb_epochs = 200
[embed_model, model] = PemNN.train_triplet(X_train, X_train_pos_all, X_train_neg_all, y_train, embed_model, model, batch_size, 
                 nb_epochs, output_directory, file_name, fig_save_path, verbose = verbose, diagonistic = visualization)


# Test triplet
X_train_feature = embed_model(X_train)
input_shape = X_train_feature.shape[1:]
nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

file_name = 'MLP_Triplet'

[input_layer, output_layer] = PemNN.create_MLP(input_shape, nb_classes)

model_class = keras.models.Model(inputs=input_layer, outputs=output_layer)

# Loss function and optimizer
model_class.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(),
  metrics=['accuracy'])

# Callbacks
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

file_path = output_directory+file_name+'_best.keras'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
  save_best_only=True)

callbacks = [reduce_lr,model_checkpoint]

batch_size = 16
nb_epochs = 20

model_class = PemNN.train_models(X_train_feature, y_train_oh, model_class, callbacks, batch_size, 
                 nb_epochs, output_directory, file_name, fig_save_path, 
                 verbose = verbose, diagonistic = visualization)  


file_name = 'Triplet_traces_Fu'
batch_size = 16
n_epochs = 200
clf = 0
PemNN.test_models(clf, x_test_nn, y_test_nn, test_size, batch_size, n_epochs,
                random_seed, no_of_pairs, resample_size, molecule,
                file_path, results_save_path, file_name, 
                use_exp_or_sim, 
                load_saved_model = False,
                embed_model = embed_model,
                model_class = model_class,
                X_test = X_test)

#%%
################################################
#  Polymer Elastic Model neural networks (PemNN)
################################################
# Build physcial data (pair channel)
model_method = 'WLC'     # 'FRC', 'FJC', 'WLC'
data_use = 'Lc_Fu' #'Lc_Fu_xp', 'Lc_Fu', 'Lc'
[x_train_nn_pairs_WLC_Lc_Fu, x_test_nn_pairs_WLC_Lc_Fu] = PemNN.get_pem_data(xp_train_exp, Fu_train_exp, 
                                                                             xp_test_exp, Fu_test_exp, 
                                                                             model_method, data_use, thres = thres,
                                                                             Lp = Lp)

print('The shape of training and testing data are '+ str(x_train_nn_pairs_WLC_Lc_Fu.shape) + 'and ' + str(x_test_nn_pairs_WLC_Lc_Fu.shape))

# swap dimensions
if (x_test_nn_pairs_WLC_Lc_Fu.shape[1]<x_test_nn_pairs_WLC_Lc_Fu.shape[2]):
    x_train_nn_pairs_WLC_Lc_Fu = x_train_nn_pairs_WLC_Lc_Fu.transpose(0,2,1)
    x_test_nn_pairs_WLC_Lc_Fu = x_test_nn_pairs_WLC_Lc_Fu.transpose(0,2,1)
    
# PemNN
filters_arr = np.array([128,256,128])
kernel_size_arr = np.array([8,5,3])
lstm_size = 8
lstm_dropout = 0.8
use_lstm_traces = True
use_lstm_pairs = True
fused_pos = 'conv' # 'conv', 'gap'
fused_method = 'conv'
# fused_method: 'add', 'average', 'max', 'weighted_average','conv'
# 'gated', 'concat'

file_name = 'Fused_traces_pairs'                 
input_shape_pairs = x_train_nn_pairs_WLC_Lc_Fu.shape[1:]
input_shape_traces = x_train_nn.shape[1:]
nb_classes = len(np.unique(np.concatenate((y_train_nn, y_test_nn), axis=0)))
batch_size = 16#16#64
nb_epochs = 200#400#300
diagonistic = False


print('Training with' + str(file_name))
clf = Classifier_PemNN(input_shape_pairs = input_shape_pairs, 
                        input_shape_traces = input_shape_traces,
                        nb_classes = nb_classes,
                        filters_arr = filters_arr,
                        kernel_size_arr = kernel_size_arr,
                        use_lstm_traces = use_lstm_traces,
                        use_lstm_pairs = use_lstm_pairs,
                        verbose = verbose,
                        file_name = file_name,
                        output_directory = output_directory,
                        fused_method = fused_method,
                        fused_pos = fused_pos,
                        )
model = clf.model

# fit

clf.fit([x_train_nn_pairs_WLC_Lc_Fu, x_train_nn], y_train_exp_oh,
        batch_size = batch_size,
        nb_epochs = nb_epochs,
        fig_save_path = fig_save_path,
        file_name = file_name,
        validation_split = 1/1000,
        diagonistic = diagonistic)


# test
[acc_FCNN, f1_score_FCNN, 
  roc_auc_score_FCNN, cm_arr] = clf.predict([x_test_nn_pairs_WLC_Lc_Fu, x_test_nn], 
                                            y_test_nn,)
                                            
#%%
# log results
result_file_name = results_save_path + 'results_log_ours' + '.csv'
if (os.path.isfile(result_file_name)):
    result_df = pd.read_csv(result_file_name)
    result_df_ind = result_df.index[-1] + 1
else:
    result_df = pd.DataFrame([])
    result_df_ind = 0
    
if (use_exp_or_sim):
    result_df.loc[result_df_ind,'molecule'] = molecule
else:
    result_df.loc[result_df_ind,'molecule'] = molecule + '_sim'
result_df.loc[result_df_ind,'models'] = 'PemNN'
result_df.loc[result_df_ind,'data_type'] = 'pairs_WLC_Lc_Fu_Traces_Fu'
result_df.loc[result_df_ind,'accuracy'] = round(acc_FCNN,4)
result_df.loc[result_df_ind, 'f1'] = round(f1_score_FCNN,4)
result_df.loc[result_df_ind,'ROC_AUC'] = round(roc_auc_score_FCNN,4)
result_df.loc[result_df_ind,'CM'] = str(cm_arr)
result_df.loc[result_df_ind,'test_size'] = test_size
result_df.loc[result_df_ind,'batch_size'] = batch_size
result_df.loc[result_df_ind,'n_epochs'] = nb_epochs
result_df.loc[result_df_ind,'seed'] = random_seed
result_df.loc[result_df_ind,'no_of_pairs'] = no_of_pairs
result_df.loc[result_df_ind,'resample_size'] = resample_size
result_df.loc[result_df_ind,'date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
result_df.loc[result_df_ind,'Lc_thres'] = thres
result_df.loc[result_df_ind,'Lp'] = Lp

    
# parameters of PemNN
result_df.loc[result_df_ind,'filters_arr'] = str(filters_arr)
result_df.loc[result_df_ind,'kernel_size_arr'] = str(kernel_size_arr)
result_df.loc[result_df_ind,'lstm_size'] = lstm_size
result_df.loc[result_df_ind,'lstm_dropout'] = lstm_dropout
result_df.loc[result_df_ind,'use_lstm_traces'] = use_lstm_traces
result_df.loc[result_df_ind,'use_lstm_pairs'] = use_lstm_pairs
result_df.loc[result_df_ind,'fused_pos'] = fused_pos
result_df.loc[result_df_ind,'fused_method'] = fused_method



result_df.to_csv(result_file_name, index = False)

print_stats = result_df[['accuracy', 'f1', 'ROC_AUC']].loc[result_df_ind]
print('The metrics for current run of ' + file_name + ' is : ' + str(print_stats))

#%%















