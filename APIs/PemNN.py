# =============================================================================
# Description
# =============================================================================
# Class to classify single molecular data using deep learning models
# =============================================================================


# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesResampler
import tensorflow.keras as keras
import tensorflow as tf
import time
from datetime import datetime
import os
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
# =============================================================================


#%%
class PemNN():
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self):         
        self.kb = 1.38064*(10**-23)# m^2 kg s^-2 K^-1 Boltzmann constant
        self.T = 300 # T
        self.kT = self.kb*self.T
    #--------------------------------------------------------------------------       
    
    
    #--------------------------------------------------------------------------
    # protein elastic models: WLC, FRC, FJC
    #--------------------------------------------------------------------------
    def get_FJC_Lc(self, xp, Fu, Lk = 300):
        Fu = Fu*1e-3 # nN
        
        y_physical_fjc = np.zeros_like(xp)
        for ii in range(xp.shape[0]):
            xp_cur = xp[ii,:,:]
            Fu_cur = Fu[ii,:,:]
            c = (Fu_cur*Lk)/(self.kb*self.T*1e9*1e12)
            a1 = np.cosh(c)/np.sinh(c)  # coth = cosh/sinh
            a2 = 1/c
            y_physical_fjc_cur = xp_cur/(a1-a2)
            
            y_physical_fjc[ii,:,:] = y_physical_fjc_cur.reshape((y_physical_fjc_cur.shape[0],1))
        
        return y_physical_fjc

    def get_WLC_Lc(self, xp, Fu, Lp = 300):
        Fu = Fu*1e-3 # nN
        
        y_physical_wlc = np.zeros_like(xp)
        for ii in range(xp.shape[0]):
            xp_cur = xp[ii,:,:]
            Fu_cur = Fu[ii,:,:]
            u = (Fu_cur*Lp)/(self.kT*1e9*1e12)
            u = u.astype(np.complex128)
            xp_cur = xp_cur.astype(dtype=np.complex128)
            
            gu_1 = 27 - (27/2*u) + 36*(u**2) - 8*(u**3)
            gu_2 = 3*np.sqrt(3.0) * 0.5
            gu_3 = -(u**2)*((4*u - 3)**3-108)

            gu_1 =  gu_1.astype(dtype=np.complex128)
            gu_2 =  gu_2.astype(dtype=np.complex128)
            gu_3 =  gu_3.astype(dtype=np.complex128)
            gu_4 = gu_1+gu_2*np.sqrt(gu_3)
            # gu = tf.experimental.numpy.cbrt(gu_4)
            gu = gu_4**(1/3)
            Lxu_1 = 3 + 4*u + ((9-3*u+4*u**2)/gu) + gu
            Lxu_2 = xp_cur/(6*u)
            y_physical_wlc_cur = Lxu_2*Lxu_1
            y_physical_wlc_cur = np.real(y_physical_wlc_cur)
            y_physical_wlc_cur = y_physical_wlc_cur.astype(dtype=np.float32)
            # proportion between two modesl
            
            y_physical_wlc[ii,:,:] = y_physical_wlc_cur.reshape((y_physical_wlc_cur.shape[0],1))
        
        return y_physical_wlc


    # In the infinite-chain limit, contour length can be computed via the following
    def get_FRC_Lc_per_sample(self, xp, Fu, Lb):
        gamma = 41/180*np.pi # 22C in radian (from "Comparing Proteins by Their Unfolding Pattern")
        # 41 C in https://blog.tremily.us/posts/FRC/ 
        a = Lb*(1+np.cos(gamma))/((1-np.cos(gamma))*np.cos(gamma/2))
        p = Lb*(np.cos(gamma/2))/(np.abs(np.log(np.cos(gamma))))
        
        y_physical_frc = [] 
        for ii in range(xp.shape[0]):
            xp_cur = xp[ii]
            Fu_cur = Fu[ii]
            v_cur = (Fu_cur*Lb)/(self.kT*1e9*1e12)
            if (v_cur < Lb/p):
                a1 = (Fu_cur*a)/(3*self.kT*1e9*1e12)
                y_physical_frc_cur = xp_cur/a1
            elif (v_cur > Lb/p) & (v_cur < p/Lb):
                a1 = np.power((4*Fu_cur*p)/(self.kT*1e9*1e12), -1/2)
                a2 = 1 - a1
                y_physical_frc_cur = xp_cur/a2
            elif (v_cur > p/Lb):
                a1 = np.power((2*Fu_cur*Lb)/(self.kT*1e9*1e12), -1)
                a2 = 1 - a1
                y_physical_frc_cur = xp_cur/a2
            y_physical_frc.append(y_physical_frc_cur)
            
        y_physical_frc = np.array(y_physical_frc)
        return y_physical_frc
    
    def get_FRC_Lc(self, xp, Fu, Lb = 300):
        # xp in nm, Fu in pN
        # FRC
        Fu = Fu*1e-3 # nN
        
        y_physical_frc = np.zeros_like(xp)
        
        
        for ii in range(xp.shape[0]):
            xp_cur = xp[ii,:,:]
            Fu_cur = Fu[ii,:,:]
            y_physical_frc_cur = self.get_FRC_Lc_per_sample(xp_cur, Fu_cur, Lb)
                    
            y_physical_frc[ii,:,:] = y_physical_frc_cur.reshape((y_physical_frc_cur.shape[0],1))
        
        return y_physical_frc
    #-------------------------------------------------------------------------- 
    
    #--------------------------------------------------------------------------   
    # Get data with countour length extracted from polymer elastic models
    
    # model_method: choose from [WLC, FRC, FJC]
    # data_use: choose Lc, xp, Fu to use
    #--------------------------------------------------------------------------   
    def get_pem_data(self, xp_train_exp, Fu_train_exp, xp_test_exp, Fu_test_exp, 
                        model_method, data_use, thres = 500, Lp = 300,
                        Lb = 150, Lk = 300):
        # model_method 'FRC', 'FJC', 'WLC'
        # data_use: which data to use 'Lc_Fu_xp', 'Lc_Fu', 'Lc'
        # thres: filter on Lc
        xp_train_exp = xp_train_exp.reshape((xp_train_exp.shape[0],xp_train_exp.shape[1], 1))
        Fu_train_exp = Fu_train_exp.reshape((Fu_train_exp.shape[0],Fu_train_exp.shape[1], 1))
        if (model_method == 'FRC'):
            Lc_train_exp = self.get_FRC_Lc(xp_train_exp, Fu_train_exp, Lb = Lb)
        if (model_method == 'FJC'):
            Lc_train_exp = self.get_FJC_Lc(xp_train_exp, Fu_train_exp, Lk = Lk)
        if (model_method == 'WLC'):
            Lc_train_exp = self.get_WLC_Lc(xp_train_exp, Fu_train_exp, Lp = Lp)   
        
        x_train_nn_resnet = np.concatenate((Lc_train_exp, Fu_train_exp, xp_train_exp), axis = 2)
        
        xp_test_exp = xp_test_exp.reshape((xp_test_exp.shape[0],xp_test_exp.shape[1], 1))
        Fu_test_exp = Fu_test_exp.reshape((Fu_test_exp.shape[0],Fu_test_exp.shape[1], 1))
        if (model_method == 'FRC'):
            Lc_test_exp = self.get_FRC_Lc(xp_test_exp, Fu_test_exp, Lb = Lb)
        if (model_method == 'FJC'):
            Lc_test_exp = self.get_FJC_Lc(xp_test_exp, Fu_test_exp, Lk = Lk)
        if (model_method == 'WLC'):
            Lc_test_exp = self.get_WLC_Lc(xp_test_exp, Fu_test_exp, Lp = Lp)   

        x_test_nn_resnet = np.concatenate((Lc_test_exp, Fu_test_exp, xp_test_exp), axis = 2)
        
        
        no_pick_data = 50 # pick minimum data out of Lc (only FL_mUtr needs this)
        # filter on x_train_nn_resnet
        x_train_nn_resnet_filterd = np.zeros_like(x_train_nn_resnet)
        for ii in range(x_train_nn_resnet.shape[0]):
            xp_cur = x_train_nn_resnet[ii,:,2]
            Fu_cur = x_train_nn_resnet[ii,:,1]
            Lc_cur = x_train_nn_resnet[ii,:,0]
            
            Lc_cur_filtered_ind = np.where([(Lc_cur<thres) & (Lc_cur>0)])[1]
            if (Lc_cur_filtered_ind.size == 0):
                # find minimum 10 points
                Lc_cur_filtered_ind = np.argpartition(Lc_cur, no_pick_data)[:no_pick_data]
            Lc_cur_filtered_ind_sel = np.random.choice(Lc_cur_filtered_ind, 
                                                        size = x_train_nn_resnet.shape[1],
                                                        replace = True)
            
            # sort index to give time series into data
            Lc_cur_filtered_ind_sel = np.sort(Lc_cur_filtered_ind_sel)
            Lc_cur_filtered = Lc_cur[Lc_cur_filtered_ind_sel]
            Fu_filtered = Fu_cur[Lc_cur_filtered_ind_sel]
            xp_filterd = xp_cur[Lc_cur_filtered_ind_sel]
            
            if (data_use == 'Lc_Fu_xp'):
                x_train_nn_resnet_filterd[ii,:,2] = xp_filterd
                x_train_nn_resnet_filterd[ii,:,1] = Fu_filtered
                x_train_nn_resnet_filterd[ii,:,0] = Lc_cur_filtered
                
            if (data_use == 'Lc_Fu'):
                x_train_nn_resnet_filterd[ii,:,1] = Fu_filtered
                x_train_nn_resnet_filterd[ii,:,0] = Lc_cur_filtered
        
            if (data_use == 'Lc'):
                x_train_nn_resnet_filterd[ii,:,0] = Lc_cur_filtered
        
        if (data_use == 'Lc_Fu'):
            x_train_nn_resnet_filterd = np.delete(x_train_nn_resnet_filterd, (2), axis=2)
        if (data_use == 'Lc'):
            x_train_nn_resnet_filterd = np.delete(x_train_nn_resnet_filterd, (1,2), axis=2)
            
            
        x_test_nn_resnet_filterd = np.zeros_like(x_test_nn_resnet)
        for ii in range(x_test_nn_resnet.shape[0]):
            xp_cur = x_test_nn_resnet[ii,:,2]
            Fu_cur = x_test_nn_resnet[ii,:,1]
            Lc_cur = x_test_nn_resnet[ii,:,0]
            Lc_cur_filtered_ind = np.where([(Lc_cur<thres) & (Lc_cur>0)])[1]
            if (Lc_cur_filtered_ind.size == 0):
                # find minimum 10 points
                Lc_cur_filtered_ind = np.argpartition(Lc_cur, no_pick_data)[:no_pick_data]
            Lc_cur_filtered_ind_sel = np.random.choice(Lc_cur_filtered_ind, 
                                                        size = x_test_nn_resnet.shape[1],
                                                        replace = True)
            # sort index to give time series into data
            Lc_cur_filtered_ind_sel = np.sort(Lc_cur_filtered_ind_sel)
            Lc_cur_filtered = Lc_cur[Lc_cur_filtered_ind_sel]
            Fu_filtered = Fu_cur[Lc_cur_filtered_ind_sel]
            xp_filterd = xp_cur[Lc_cur_filtered_ind_sel]
            
            if (data_use == 'Lc_Fu_xp'):
                x_test_nn_resnet_filterd[ii,:,2] = xp_filterd
                x_test_nn_resnet_filterd[ii,:,1] = Fu_filtered
                x_test_nn_resnet_filterd[ii,:,0] = Lc_cur_filtered
                
            if (data_use == 'Lc_Fu'):
                x_test_nn_resnet_filterd[ii,:,1] = Fu_filtered
                x_test_nn_resnet_filterd[ii,:,0] = Lc_cur_filtered
        
            if (data_use == 'Lc'):
                x_test_nn_resnet_filterd[ii,:,0] = Lc_cur_filtered

        if (data_use == 'Lc_Fu'):
            x_test_nn_resnet_filterd = np.delete(x_test_nn_resnet_filterd, (2), axis=2)
        if (data_use == 'Lc'):
            x_test_nn_resnet_filterd = np.delete(x_test_nn_resnet_filterd, (1,2), axis=2)
        
        x_train_nn_pairs = x_train_nn_resnet_filterd#x_train_nn_resnet_filterd #x_train_nn_resnet
        x_test_nn_pairs = x_test_nn_resnet_filterd#x_test_nn_resnet_filterd #x_test_nn_resnet
        
        x_train_nn_pairs = x_train_nn_pairs.transpose(0,2,1)
        x_test_nn_pairs = x_test_nn_pairs.transpose(0,2,1)
        
        return (x_train_nn_pairs, x_test_nn_pairs)
    #--------------------------------------------------------------------------   
    
    
    
    #--------------------------------------------------------------------------        
    # Test models and log results
    #--------------------------------------------------------------------------  
    def test_models(self, clf, x_test_nn, y_test_nn, test_size, batch_size, n_epochs,
                    random_seed, no_of_pairs, resample_size, molecule,
                    file_path, results_save_path, file_name, 
                    use_exp_or_sim, thres = None, Lp = None,
                    test_molecule = None, 
                    chrono_order = None, 
                    load_saved_model = False,
                    train_size = None,
                    embed_model = None, model_class = None, X_test = None,):
        
        if (model_class is not None) and (embed_model is not None):
            # testing for triplet model
            y_pred_feature = embed_model.predict(X_test)
            y_pred_multi = model_class.predict(y_pred_feature)
        else:
            # if load saved model, transpose the data and use predict to get probability
            if (load_saved_model):
                clf = keras.models.load_model(file_path)
                y_pred_multi = clf.predict(x_test_nn.transpose(0,2,1))
            else:
                y_pred_multi = clf.predict_proba(x_test_nn)
            
        
        y_pred_multi_norm = y_pred_multi/y_pred_multi.sum(axis=1,keepdims=1)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred_multi , axis=1)
        acc_FCNN = accuracy_score(y_test_nn, y_pred, normalize=True)
        f1_score_FCNN = f1_score(y_test_nn, y_pred, average = 'weighted')
        roc_auc_score_FCNN = roc_auc_score(y_test_nn, y_pred_multi_norm, average = 'weighted', multi_class= 'ovr')
        cm = confusion_matrix(y_test_nn, y_pred)
        cm_arr = [cm[0,0], cm[0,1], cm[0,2],cm[1,0],
                  cm[1,1],cm[1,2],cm[2,0], cm[2,1],cm[2,2]]

        # log results
        result_file_name = results_save_path + 'results_log' + '.csv'
        if (os.path.isfile(result_file_name)):
            result_df = pd.read_csv(result_file_name)
            result_df_ind = result_df.index[-1] + 1
        else:
            result_df = pd.DataFrame([])
            result_df_ind = 0
        
        model_name = file_name.split('_')[0]
        data_type = file_name.split('_',1)[1]
        
        if (use_exp_or_sim):
            result_df.loc[result_df_ind,'molecule'] = molecule
        else:
            result_df.loc[result_df_ind,'molecule'] = molecule + '_sim'
        result_df.loc[result_df_ind,'models'] = model_name
        result_df.loc[result_df_ind,'data_type'] = data_type
        result_df.loc[result_df_ind,'accuracy'] = round(acc_FCNN,4)
        result_df.loc[result_df_ind, 'f1'] = round(f1_score_FCNN,4)
        result_df.loc[result_df_ind,'ROC_AUC'] = round(roc_auc_score_FCNN,4)
        result_df.loc[result_df_ind,'CM'] = str(cm_arr)
        result_df.loc[result_df_ind,'test_size'] = test_size
        result_df.loc[result_df_ind,'train_size'] = train_size
        result_df.loc[result_df_ind,'batch_size'] = batch_size
        result_df.loc[result_df_ind,'n_epochs'] = n_epochs
        result_df.loc[result_df_ind,'seed'] = random_seed
        result_df.loc[result_df_ind,'no_of_pairs'] = no_of_pairs
        result_df.loc[result_df_ind,'resample_size'] = resample_size
        result_df.loc[result_df_ind,'date'] = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        result_df.loc[result_df_ind,'Lc_thres'] = thres
        result_df.loc[result_df_ind,'Lp'] = Lp
        result_df.loc[result_df_ind,'test_molecule'] = test_molecule
        result_df.loc[result_df_ind,'chrono_order'] = chrono_order
        
        
        result_df.to_csv(result_file_name, index = False)

        print_stats = result_df[['accuracy', 'f1', 'ROC_AUC']].loc[result_df_ind]
        print('The metrics for current run of ' + file_name + ' is : ' + str(print_stats))
        
        return None
    #--------------------------------------------------------------------------     
    

    

    
    #--------------------------------------------------------------------------        
    # Create MLP 
    # input_shape: the number of neurons in the input layer
    # nb_classes: the number of neurons in the output layer
    #--------------------------------------------------------------------------  
    def create_MLP(self, input_shape, nb_classes):
        # Create model
        input_layer = keras.layers.Input(input_shape)
        
        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)
        
        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)
        
        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)
        
        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)
        
        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)
    
        return (input_layer, output_layer)
    #--------------------------------------------------------------------------  

    
    #--------------------------------------------------------------------------        
    # Create Triplet 
    # input_shape: the number of neurons in the input layer
    #--------------------------------------------------------------------------  
    def create_Triplet(self, input_shape):
        # Create embed model: the model consists of three embeded models, while parameters are the same
        def create_embedding(input_shape):
            n_feature_maps = 128#128 #64# 64
            input_layer = keras.layers.Input(input_shape)
            
            # BLOCK 1
            conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            
            conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            
            conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)
            
            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
            
            output_block_1 = keras.layers.add([shortcut_y, conv_z])
            output_block_1 = keras.layers.Activation('relu')(output_block_1)
            
            # BLOCK 2
            conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            
            conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            
            conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)
            
            # expand channels for the sum
            shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
            shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
            
            output_block_2 = keras.layers.add([shortcut_y, conv_z])
            output_block_2 = keras.layers.Activation('relu')(output_block_2)
            
            # BLOCK 3
            conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
            conv_x = keras.layers.BatchNormalization()(conv_x)
            conv_x = keras.layers.Activation('relu')(conv_x)
            
            conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
            conv_y = keras.layers.BatchNormalization()(conv_y)
            conv_y = keras.layers.Activation('relu')(conv_y)
            
            conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
            conv_z = keras.layers.BatchNormalization()(conv_z)
            
            # no need to expand channels because they are equal
            shortcut_y = keras.layers.BatchNormalization()(output_block_2)
            
            output_block_3 = keras.layers.add([shortcut_y, conv_z])
            
            output_layer = output_block_3
            
            model = keras.models.Model(inputs=input_layer, outputs=output_layer)

            return model


        # build the anchor, positive and negative input layer
        anchorInput = keras.Input(input_shape , name="anchor")
        positiveInput = keras.Input(input_shape, name="positive")
        negativeInput = keras.Input(input_shape, name="negative")

        # Create embedding model
        embeddingModel = create_embedding(input_shape)

        # embed the anchor, positive and negative images
        anchorEmbedding = embeddingModel(anchorInput)
        positiveEmbedding = embeddingModel(positiveInput)
        negativeEmbedding = embeddingModel(negativeInput)
        # build the siamese network and return it
        siamese_network = keras.Model(
            inputs=[anchorInput, positiveInput, negativeInput],
            outputs=[anchorEmbedding, positiveEmbedding, negativeEmbedding]
        )
        
        return (embeddingModel, siamese_network)
    #--------------------------------------------------------------------------        
    
    
        
    #--------------------------------------------------------------------------        
    # Train models
    #--------------------------------------------------------------------------  
    def train_models(self, X_train, y_train_oh, model, callbacks, batch_size, 
                     nb_epochs, output_directory, file_name, fig_save_path, 
                     validation_split = 0.2, verbose = 1, diagonistic = False):
        mini_batch_size = int(min(X_train.shape[0]/10, batch_size))

        start_time = time.time()

        hist = model.fit(X_train, y_train_oh, batch_size=mini_batch_size, epochs=nb_epochs,
          validation_split = validation_split, verbose=verbose, callbacks=callbacks)

        duration = time.time() - start_time

        model.save(output_directory + file_name + '_last.keras')

        # Plotting tranining statistics
        history_dict = hist.history

        acc = history_dict['accuracy']
        loss = history_dict['loss']

        val_acc = history_dict['val_accuracy']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        # Plot loss
        if (diagonistic == True):
            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'ro', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.savefig(fname = fig_save_path + file_name + '_loss' + '.png')
            plt.savefig(fname = fig_save_path + file_name + '_loss' + '.svg')
    
            # Plot accuracy
            plt.figure()
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'ro', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.show()
            
            plt.savefig(fname = fig_save_path + file_name + '_acc' + '.png')
            plt.savefig(fname = fig_save_path + file_name + '_acc' + '.svg')
        return (model)
    #--------------------------------------------------------------------------  


    #--------------------------------------------------------------------------        
    # Train models: different ways of training triplet models
    #--------------------------------------------------------------------------  
    def train_triplet(self, X_train, X_train_pos_all, X_train_neg_all, y_train, embed_model, model, batch_size, 
                     nb_epochs, output_directory, file_name, fig_save_path, verbose = 1, diagonistic = False):
    
        # Define loss function and training function
        def loss(model, x, y, training, margin = 10.0):
          # training=training is needed only if there are layers with different
          # behavior during training versus inference (e.g. Dropout).
          all_outputs = model(x)
    
          anchor_output = all_outputs[0]
          positive_output = all_outputs[1]
          negative_output = all_outputs[2]
    
          d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
          d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)
    
    
          loss = tf.nn.relu(margin + d_pos - d_neg)
          loss = tf.reduce_mean(loss)
    
          # loss = batch_hard_triplet_loss(y, all_outputs, margin, squared=False)
    
          return loss
    
        def grad(model, inputs, targets):
          with tf.GradientTape() as tape:
            loss_value = loss(model, inputs, targets, training=True)
          return loss_value, tape.gradient(loss_value, model.trainable_variables)
    
    
        # Making training faster
        epoch_loss_avg = tf.keras.metrics.Mean()
    
        @tf.function
        def train_step(x, y):
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_avg.update_state(loss_value)
            return loss_value
        
        file_path = output_directory + file_name + '_last.keras'

        train_loss_results = []
        optimizer = tf.keras.optimizers.Adam()

        train_data = [X_train, X_train_pos_all, X_train_neg_all]

        train_label = y_train

        pre_loss = tf.zeros(1)
        for epoch in range(nb_epochs):

            # Training loop - using batches of 32
            for ii in range(int(X_train.shape[0]/batch_size)):
                x = [X_train[32*ii:32*(ii+1),:,:], X_train_pos_all[32*ii:32*(ii+1),:,:], X_train_neg_all[32*ii:32*(ii+1),:,:]]
                y = train_label[32*ii:32*(ii+1)]
                # Optimize the model
                loss_value = train_step(x, y)

            # Display metrics at the end of each epoch.
            pre_loss = loss_value
            loss = epoch_loss_avg.result()
            if (verbose == 1):
                print("Epoch {:03d}: Loss: {:.20f}".format(epoch,epoch_loss_avg.result()))
            train_loss_results.append(loss)
  
            if (tf.math.abs(loss-pre_loss) <= 1e-8):
              break # quit training if loss is zero, no further updates
  
            # Reset training metrics at the end of each epoch
            # epoch_loss_avg.reset_states()
            epoch_loss_avg.reset_state()

        # Save model
        embed_model.save(file_path)

        epochs = range(1, len(train_loss_results) + 1)

        # Plot loss
        if (diagonistic == True):
            plt.figure()
            plt.plot(epochs, train_loss_results, 'bo', label='Training loss')
            plt.title('Training loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.ylim([0,10])
            plt.show()
            
            plt.savefig(fname = fig_save_path + file_name + 'acc_loss' + '.png')
            plt.savefig(fname = fig_save_path + file_name + 'acc_loss' + '.svg')
        
        return (embed_model, model)
    #--------------------------------------------------------------------------  
 
    
 
    

    #--------------------------------------------------------------------------        
    # Test models with improved thresholds -- use FCN and ResNet only
    #--------------------------------------------------------------------------  
    def test_models_thresholds(self, X_test_exp, y_test_exp, model,
                     file_name, fig_save_path, diagonistic = False):
        
        # Modify labels in class 1 vs the rest
        y_test_exp_one_vs_r = [1 if y_test_exp[i] == 1 else 0 for i in range(len(y_test_exp))]
        y_test_exp_one_vs_r = np.array(y_test_exp_one_vs_r)
        
        # get roc curvces per classes, using one over rest
        y_pred_exp_multi = model.predict(X_test_exp)
        y_pred_exp_multi_norm = y_pred_exp_multi/y_pred_exp_multi.sum(axis=1,keepdims=1)
          
        fpr = {}
        tpr = {}
        thresh ={}
        n_class = 3
        for i in range(n_class):    
            fpr[i], tpr[i], thresh[i] = roc_curve(y_test_exp, y_pred_exp_multi_norm[:,i], pos_label=i)
            
        # plotting  
        if (diagonistic == True):
            plt.figure()
            plt.plot(fpr[0], tpr[0], linestyle='--',color='orange', label='Class 0 vs Rest')
            plt.plot(fpr[1], tpr[1], linestyle='--',color='green', label='Class 1 vs Rest')
            plt.plot(fpr[2], tpr[2], linestyle='--',color='blue', label='Class 2 vs Rest')
            plt.title('Multiclass ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive rate')
            plt.legend(loc='best')
            
        # get the optimal threshold
        class_no = 1
        # Youden's J statistic https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
        optimal_fcn = tpr[class_no] - fpr[class_no]

        
        optimal_idx = np.argmax(optimal_fcn)
        optimal_threshold = thresh[class_no][optimal_idx]
        if (diagonistic == True):
            print("Threshold value is:", optimal_threshold)
        
        
        # Metrics before changing threshold
        y_pred_exp = np.argmax(y_pred_exp_multi , axis=1)
        y_pred_exp_one_vs_r = [1 if y_pred_exp[i] == 1 else 0 for i in range(len(y_pred_exp))]
        y_pred_exp_one_vs_r = np.array(y_pred_exp_one_vs_r)

        acc_before = accuracy_score(y_test_exp_one_vs_r, y_pred_exp_one_vs_r, normalize=True)
        f1_score_before = f1_score(y_test_exp_one_vs_r, y_pred_exp_one_vs_r, average = 'weighted')

        print('The percentage accuracy for ' + file_name + ' with exp data before thresholding is :' +  str(acc_before),)
        print('The F1 score (weighted) for ' + file_name + ' with exp data before thresholding is :' +  str(f1_score_before),)

        # Confusion matrix and plot
        cm_before = confusion_matrix(y_test_exp_one_vs_r, y_pred_exp_one_vs_r)
        cm_before_arr = [cm_before[0,0],cm_before[0,1],cm_before[1,0],cm_before[1,1]]

        # Metrics after chaning threshold
        y_pred_thres = [1 if y_pred_exp_multi[i,1] > optimal_threshold else 0 for i in range(len(y_test_exp_one_vs_r))]
        y_pred_thres = np.array(y_pred_thres)
        acc_after = accuracy_score(y_test_exp_one_vs_r, y_pred_thres, normalize=True)
        f1_score_after = f1_score(y_test_exp_one_vs_r, y_pred_thres, pos_label = 0, average = 'weighted')

        print('The percentage accuracy for ' + file_name + ' with exp data after thresholding is :' +  str(acc_after),)
        print('The F1 score (weighted) for ' + file_name + ' with exp data after thresholding is :' +  str(f1_score_after),)

        
        # Confusion matrix and plot
        cm_after = confusion_matrix(y_test_exp_one_vs_r, y_pred_thres)
        cm_after_arr = [cm_after[0,0],cm_after[0,1],cm_after[1,0],cm_after[1,1]]

        # writing to text file
        txt_name = fig_save_path + file_name + '_threshold_logs.txt'
        with open(txt_name, "a") as f:
            print('The percentage accuracy for ' + file_name + ' with exp data before thresholding is :' +  str(acc_before),
                  file = f)
            print('The F1 score (weighted) for ' + file_name + ' with exp data before thresholding is :' +  str(f1_score_before),
                  file = f)
            print('Confusion_matrix before thresholding = ' + str(cm_before_arr), 
                  file = f)
    
    
            print('The percentage accuracy for ' + file_name + ' with exp data after thresholding is :' +  str(acc_after),
                  file = f)
            print('The F1 score (weighted) for ' + file_name + ' with exp data after thresholding is :' +  str(f1_score_after),
                  file = f)
            print('Confusion_matrix after thresholding = ' + str(cm_after_arr), 
                  file = f)
    
        return None
    #--------------------------------------------------------------------------        
    
  


#%%
#--------------------------------------------------------------------------
# classifier of PemNN 
#--------------------------------------------------------------------------
import keras
keras.saving.get_custom_objects().clear()

# Upon registration, you can optionally specify a package or a name.
# If left blank, the package defaults to `Custom` and the name defaults to
# the class name.
@keras.saving.register_keras_serializable(package="WeightedAverageLayer")
class WeightedAverageLayer(keras.layers.Layer):
    def __init__(self, num_inputs):
        super().__init__()
        self.w = self.add_weight(
            shape=(int(num_inputs),),
            initializer="random_normal",
            trainable=True,
            # constraint=tf.keras.constraints.min_max_norm(max_value=1.0, min_value=0.0),
            dtype='float32',
        )
        self.num_inputs = num_inputs
    def call(self, inputs):
        # Ensure weights sum to 1 using softmax
        normalized_weights = tf.nn.softmax(self.w)

        # Apply weighted average
        weighted_sum = tf.reduce_sum(
            [inputs[i] * normalized_weights[i] for i in range(len(inputs))],
            axis=0
        )

        return weighted_sum
    
    def get_config(self):
        return {"num_inputs": self.num_inputs}
    
class Classifier_PemNN:
    def __init__(self, input_shape_pairs, input_shape_traces, nb_classes, 
                 filters_arr = np.array([128,256,128]), 
                 kernel_size_arr = np.array([8,5,3]),
                 lstm_size = 8,
                 lstm_dropout = 0.8, 
                 use_lstm_traces = True,
                 use_lstm_pairs = True, 
                 verbose=False,
                 output_directory = None,
                 file_name = None,
                 fused_pos = 'conv',
                 fused_method = 'average'):
        # fused_pos: 'conv', 'gap'
        # fused_method: 'add', 'average', 'max', 'weighted_average','conv',
        # 'gated', 'concat' used for 'gap' 
        if verbose:
            print('Creating PemNN Classifier')
        self.verbose = verbose
        self.filters_arr = filters_arr
        self.kernel_size_arr = kernel_size_arr
        self.lstm_size = lstm_size
        self.lstm_dropout = lstm_dropout
        self.conv1_pairs = keras.layers.Conv1D(filters=self.filters_arr[0], 
                                             kernel_size=int(self.kernel_size_arr[0]), 
                                             padding='same')
        
        self.conv1_traces = keras.layers.Conv1D(filters=self.filters_arr[0], 
                                             kernel_size=int(self.kernel_size_arr[0]), 
                                             padding='same')
        
        # shared parameters
        self.conv2 = keras.layers.Conv1D(filters=self.filters_arr[1], 
                                             kernel_size=int(self.kernel_size_arr[1]), 
                                             padding='same')
        
        self.conv3 = keras.layers.Conv1D(filters=self.filters_arr[2], 
                                             kernel_size=int(self.kernel_size_arr[2]), 
                                             padding='same')
        
        self.bn2 = keras.layers.BatchNormalization()
        self.bn3 = keras.layers.BatchNormalization()
        
        self.gap = keras.layers.GlobalAveragePooling1D()
        
        self.use_lstm_traces = use_lstm_traces
        self.use_lstm_pairs = use_lstm_pairs
        self.output_directory = output_directory
        self.file_name = file_name
        self.fused_method = fused_method
        self.fused_pos = fused_pos
        
        # Build Model -----------------------------------------------------------
        self.model = self.build_model(input_shape_pairs, input_shape_traces, nb_classes)
        # -----------------------------------------------------------------------
        if verbose:
            self.model.summary()
            

    def build_model(self, input_shape_pairs, input_shape_traces, nb_classes):
        
        # input
        x_pairs = keras.layers.Input(input_shape_pairs)
        x_traces = keras.layers.Input(input_shape_traces)
        input_layer = [x_pairs, x_traces]
        
        # conv1_pairs
        pairs_conv1 = self.conv1_pairs(x_pairs)
        pairs_conv1 = keras.layers.BatchNormalization()(pairs_conv1)
        pairs_conv1 = keras.layers.Activation(activation='relu')(pairs_conv1)
        
        # conv1_traces
        traces_conv1 = self.conv1_traces(x_traces)
        traces_conv1 = keras.layers.BatchNormalization()(traces_conv1)
        traces_conv1 = keras.layers.Activation(activation='relu')(traces_conv1)
        
        if (self.fused_pos == 'conv'): # fuse at conv layer
            # fuse conv1_paris and conv1_traces
            if (self.fused_method == 'add'):
                fused_conv1 = keras.layers.add([pairs_conv1, traces_conv1]) # add
            elif (self.fused_method == 'average'):
                fused_conv1 = keras.layers.average([pairs_conv1, traces_conv1]) # average
            elif (self.fused_method == 'max'):
                fused_conv1 = keras.layers.Maximum()([pairs_conv1, traces_conv1]) # Maximum
            elif (self.fused_method == 'weighted_average'):
                wt_add = WeightedAverageLayer(2)
                fused_conv1 = wt_add([pairs_conv1, traces_conv1])
            elif (self.fused_method == 'conv'):
                # using nn to fuse
                fused_conv1 = keras.layers.Concatenate(axis = 2)([pairs_conv1, traces_conv1])
                fused_conv1 = keras.layers.Conv1D(filters=self.filters_arr[1], kernel_size=1, padding='same')(fused_conv1)
                fused_conv1 = keras.layers.BatchNormalization()(fused_conv1)
                fused_conv1 = keras.layers.Activation(activation='relu')(fused_conv1)
            else:
                print('Please select valid fusing method!')
            
            # conv2
            fused_conv2 = self.conv2(fused_conv1)
            fused_conv2 = self.bn2(fused_conv2)
            fused_conv2 = keras.layers.Activation(activation='relu')(fused_conv2)
            
            # conv3
            fused_conv3 = self.conv3(fused_conv2)
            fused_conv3 = self.bn3(fused_conv3)
            fused_conv3 = keras.layers.Activation(activation='relu')(fused_conv3)
            
            # gap
            gap_fcn = self.gap(fused_conv3)
            
        elif(self.fused_pos == 'gap'): # fuse at gap layer
            # conv2_pairs
            pairs_conv2 = self.conv2(pairs_conv1)
            pairs_conv2 = self.bn2(pairs_conv2)
            pairs_conv2 = keras.layers.Activation(activation='relu')(pairs_conv2)
            
            # conv2_traces
            traces_conv2 = self.conv2(traces_conv1)
            traces_conv2 = self.bn2(traces_conv2)
            traces_conv2 = keras.layers.Activation(activation='relu')(traces_conv2)
            
            # conv3_pairs
            pairs_conv3 = self.conv3(pairs_conv2)
            pairs_conv3 = self.bn3(pairs_conv3)
            pairs_conv3 = keras.layers.Activation(activation='relu')(pairs_conv3)
            
            # conv3_traces
            traces_conv3 = self.conv3(traces_conv2)
            traces_conv3 = self.bn3(traces_conv3)
            traces_conv3 = keras.layers.Activation(activation='relu')(traces_conv3)
            
            # gap
            gap_pairs = self.gap(pairs_conv3)
            gap_traces = self.gap(traces_conv3)
            gap_fcn = keras.layers.concatenate([gap_pairs, gap_traces])
            if (self.fused_method == 'concat'):
                gap_fcn = gap_fcn
            elif (self.fused_method == 'gated'):
                # https://github.com/ZZUFaceBookDL/GTN/blob/master/Gated%20Transformer%20%E8%AE%BA%E6%96%87IJCAI%E7%89%88/module/transformer.py
                gate = keras.layers.Dense(2, use_bias = True, activation='softmax')(gap_fcn)
                gap_fcn = keras.layers.concatenate([gap_pairs*gate[:,0:1], gap_traces*gate[:,1:2]])
                
            else:
                print('Please select valid fusing method!')
                
        else:
            print('Please select valid fusing postion!')
            
        # lstm
        if (self.use_lstm_traces):
            lstm_traces = keras.layers.Permute((2, 1))(x_traces)
            lstm_traces = keras.layers.LSTM(self.lstm_size)(lstm_traces)
            lstm_traces = keras.layers.Dropout(self.lstm_dropout)(lstm_traces)
            
        if (self.use_lstm_pairs):
            lstm_pairs = keras.layers.Permute((2, 1))(x_pairs)
            lstm_pairs = keras.layers.LSTM(self.lstm_size)(lstm_pairs)
            lstm_pairs = keras.layers.Dropout(self.lstm_dropout)(lstm_pairs)
            
        # concat gap
        if (self.use_lstm_traces) and (self.use_lstm_pairs):
            concat_gap = keras.layers.concatenate([lstm_traces, lstm_pairs, gap_fcn])
        elif(self.use_lstm_traces) and (~self.use_lstm_pairs):
            concat_gap = keras.layers.concatenate([lstm_traces, gap_fcn])
        elif(~self.use_lstm_traces) and (self.use_lstm_pairs):
            concat_gap = keras.layers.concatenate([lstm_pairs, gap_fcn])
        else:
            concat_gap = gap_fcn
            
        # output
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(concat_gap)
        
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        
        
        # Loss function and optimizer
        model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(),
          metrics=['accuracy'])
        
        # Callbacks
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
          min_lr=0.0001)
        
        file_path = self.output_directory + self.file_name + '_best.keras'
        
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
          save_best_only=True)
        
        self.callbacks = [reduce_lr,model_checkpoint]
        return model

    def fit(self, x_train, y_train,
            batch_size, nb_epochs, fig_save_path,
            file_name, 
            validation_split = 0.2, diagonistic = False):

        mini_batch_size = int(min(x_train[0].shape[0]/10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, 
                              epochs=nb_epochs,validation_split = validation_split,
                              verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + self.file_name + '_last.keras')

        # Plotting tranining statistics
        history_dict = hist.history

        acc = history_dict['accuracy']
        loss = history_dict['loss']

        val_acc = history_dict['val_accuracy']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        # Plot loss
        if (diagonistic == True):
            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'ro', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.savefig(fname = fig_save_path + file_name + '_loss' + '.png')
            plt.savefig(fname = fig_save_path + file_name + '_loss' + '.svg')
    
            # Plot accuracy
            plt.figure()
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'ro', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.show()
            
            plt.savefig(fname = fig_save_path + file_name + '_acc' + '.png')
            plt.savefig(fname = fig_save_path + file_name + '_acc' + '.svg')
        keras.backend.clear_session()
            
    def predict(self, x_test, y_test_nn):
        file_path = self.output_directory + self.file_name + '_best.keras'
        model = keras.models.load_model(file_path)
        
        y_pred_multi = model.predict(x_test)
        y_pred_multi_norm = y_pred_multi/y_pred_multi.sum(axis=1,keepdims=1)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred_multi , axis=1)
    
    
        acc_FCNN = accuracy_score(y_test_nn, y_pred, normalize=True)
        f1_score_FCNN = f1_score(y_test_nn, y_pred, average = 'weighted')
        roc_auc_score_FCNN = roc_auc_score(y_test_nn, y_pred_multi_norm, average = 'weighted', multi_class= 'ovr')
        cm = confusion_matrix(y_test_nn, y_pred)
        cm_arr = [cm[0,0], cm[0,1], cm[0,2],cm[1,0],
                  cm[1,1],cm[1,2],cm[2,0], cm[2,1],cm[2,2]]
        
        return (acc_FCNN, f1_score_FCNN, roc_auc_score_FCNN, cm_arr)
    
    def fit_and_predict(self, model, x_train, y_train,
                        x_test, y_test_nn, 
                        batch_size, nb_epochs, fig_save_path,
                        file_name, 
                        validation_split = 0.2, diagonistic = False):
        # given model and do fit + prediction

        mini_batch_size = int(min(x_train[0].shape[0]/10, batch_size))

        start_time = time.time()

        hist = model.fit(x_train, y_train, batch_size=mini_batch_size, 
                              epochs=nb_epochs,validation_split = validation_split,
                              verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        # model.save(self.output_directory + self.file_name + '_last.keras')

        # Plotting tranining statistics
        history_dict = hist.history

        acc = history_dict['accuracy']
        loss = history_dict['loss']

        val_acc = history_dict['val_accuracy']
        val_loss = history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        # Plot loss
        if (diagonistic == True):
            plt.figure()
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'ro', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            plt.savefig(fname = fig_save_path + file_name + '_loss' + '.png')
            plt.savefig(fname = fig_save_path + file_name + '_loss' + '.svg')
    
            # Plot accuracy
            plt.figure()
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'ro', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend(loc='lower right')
            plt.show()
            
            plt.savefig(fname = fig_save_path + file_name + '_acc' + '.png')
            plt.savefig(fname = fig_save_path + file_name + '_acc' + '.svg')
            
        y_pred_multi = model.predict(x_test)
        y_pred_multi_norm = y_pred_multi/y_pred_multi.sum(axis=1,keepdims=1)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred_multi , axis=1)
    
    
        acc_FCNN = accuracy_score(y_test_nn, y_pred, normalize=True)
        f1_score_FCNN = f1_score(y_test_nn, y_pred, average = 'weighted')
        roc_auc_score_FCNN = roc_auc_score(y_test_nn, y_pred_multi_norm, average = 'weighted', multi_class= 'ovr')
        cm = confusion_matrix(y_test_nn, y_pred)
        cm_arr = [cm[0,0], cm[0,1], cm[0,2],cm[1,0],
                  cm[1,1],cm[1,2],cm[2,0], cm[2,1],cm[2,2]]
        
        return (acc_FCNN, f1_score_FCNN, roc_auc_score_FCNN, cm_arr)    
    


    
    
    
    
    
#--------------------------------------------------------------------------   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    