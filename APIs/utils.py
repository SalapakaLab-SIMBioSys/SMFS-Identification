# =============================================================================
# Description
# =============================================================================

# Utils to visualize data

# =============================================================================



# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import random
import tensorflow as tf
import os
import pandas as pd
import scipy

from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesResampler, TimeSeriesScalerMeanVariance

import sys
sys.path.append('../')


# =============================================================================

#%%
# =============================================================================
# Class containing utils functions
# =============================================================================
class utils():
    
    
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self):         
        self.color_set = ([109/255, 1/255, 31/255], [183/255, 34/255, 48/255], [220/255, 109/255, 87/255], [246/255, 178/255, 147/255],
                     [251/255, 227/255, 213/255], [233/255, 241/255, 244/255], [182/255, 215/255, 232/255], [182/255, 215/255, 232/255],
                      [109/255, 173/255, 209/255], [49/255, 124/255, 183/255], [16/255, 70/255, 128/255])

        self.color_set_hist = ([255/255, 220/255, 0/255], [180/255, 204/255, 0/255], [86/255, 187/255, 85/255],
                          [0/255, 159/255, 105/255], [0/255, 131/255, 125/255],[255/255, 220/255, 0/255], [180/255, 204/255, 0/255], [86/255, 187/255, 85/255],
                          [0/255, 159/255, 105/255], [0/255, 131/255, 125/255],[255/255, 220/255, 0/255], [180/255, 204/255, 0/255], [86/255, 187/255, 85/255],
                          [0/255, 159/255, 105/255], [0/255, 131/255, 125/255])

        self.color_set_hist_force = ([255/255, 220/255, 0/255], [180/255, 204/255, 0/255], [86/255, 187/255, 85/255],
                          [0/255, 159/255, 105/255], [0/255, 131/255, 125/255])

        self.plot_fonts = {'font.family':'Arial',
                              'font.size': 20,
                              'figure.figsize': (8,6),#(15.428,11),#(15.428,9), #(12,7),
                              'mathtext.fontset': 'cm',
                              'xtick.labelsize':26,
                              'ytick.labelsize':26,
                              'axes.titlesize' : 26,
                              'axes.labelsize' : 26,
                              }    
        self.marker_set = ['o', '^', 's', '*', '+', 'o', '^', 's', '*', '+']

        self.fmt_set = [':', '--', '-', '-.']
    #--------------------------------------------------------------------------       
    
    
    #--------------------------------------------------------------------------        
    # Seed all 
    #--------------------------------------------------------------------------  
    def seed_all(self, random_seed):
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)        
        return None
    #--------------------------------------------------------------------------  
    
    #--------------------------------------------------------------------------
    # Trimming data after detachment to make alignment of signals
    #--------------------------------------------------------------------------
    def trimming_data(self, data, data_xp, No_molecule, trim_percent = 0.15):
        # trim_with_time: using time as trimming criterion, otherwise use xp 
        if (int(No_molecule)>0):
            # brute force to find detachment event
            data_non_zero = data[data!=0]
            data_actual_length = len(data_non_zero)
            
            data_flat = data_non_zero[data_actual_length-int(data_actual_length*0.1):data_actual_length]
            y_mean = np.mean(data_flat)
            y_std =np.std(data_flat)
            
            detachment_index = data_actual_length
            thres = y_mean + 5*y_std
            for ii in range(data_actual_length-1, 0, -1):
                if (data_non_zero[ii]>thres):
                    detachment_index = ii
                    break
                
            # do brute force again --- increase the accuracy
            data_flat = data_non_zero[detachment_index:data_actual_length]
            y_mean = np.mean(data_flat)
            y_std =np.std(data_flat)
            
            thres = y_mean + 5*y_std
            for ii in range(data_actual_length-1, 0, -1):
                if (data_non_zero[ii]>thres):
                    detachment_index = ii
                    break
                            

            stop_index = detachment_index + int(trim_percent*data_actual_length)
                
            if(stop_index == 0):
                stop_index = len(data)
            
            
            data_out = np.zeros_like(data)
            data_out[0:stop_index] = data[0:stop_index]
        else:
            data_out = data
            
        return data_out
    
    #--------------------------------------------------------------------------
    # Build simulation dataset for ML classificaiton
    
    # Choose properties to build dataset
    # molecule: Choose from ['Titin', 'UtrNR3','DysNR3_bact', 'UtrNR3_bact']
    # No_molecule_arr = np.array(['0', '1', '2'])  #No. of Molecule being attached to cantilver tips. 
    # koff_method_arr = np.array(['DHS'])  # 'DHS' 
    # Energy_shape_arr = np.array([0]) # 0 for cusp like, 1 for linear cubic
    # Noise_data = True # Read noised data
    # property_name_arr = np.array(['xp','Fwlc','time']) # Data properties to read
    # trimming = True # Trim Fwlc to zero after detachment
    # df_save_path  # Saving path
    #--------------------------------------------------------------------------
    def build_sim_dataset_ML(self, molecule, No_molecule_arr = np.array(['0', '1', '2']), 
                             speed = None, 
                             Noise_data = True, property_name_arr = np.array(['xp','Fwlc','time','Fwlc_ori']),
                             trimming = True, df_save_path = '../Data/ML_Dataset/', 
                             koff_method_arr = np.array(['DHS']), 
                             Energy_shape_arr = np.array([0]),):
        
        molecule_arr = np.array([molecule])
        
        # The number of data to read under each situations, to make dataset balance
        if (molecule == 'Titin'):
            if (speed is None):
                speed = 2000e-9
            
            no_data_arr = np.array([512,128,32])*2 #np.array([256,64,32])# No. of data to data under a specific situation, choose between [0,1000] (keep data balance)
            # no_data_arr = np.array([512,128,14,13]) # for three molecules
        elif (molecule == 'UtrNR3') or (molecule == 'DysNR3_bact') or (molecule == 'UtrNR3_bact'):
            if (speed is None):
                speed = 1000e-9
            no_data_arr = np.array([512,128,32])*4

            
        speed_arr = np.array([speed]) # Constant pulling speeds
        
        os.makedirs(df_save_path, exist_ok=True)
        
        # Initialize dataframe to save data
        data_pd = pd.DataFrame() 
        data_pd['molecule'] = []
        data_pd['koff_method'] = []
        data_pd['Energy_shape'] = []
        data_pd['No_molecule'] = []
        data_pd['speed'] = []
        data_pd['Initial_Pos'] = []
        
        # specifying astype
        data_pd['molecule'] = data_pd['molecule'].astype('string')
        data_pd['koff_method'] = data_pd['koff_method'].astype('string')
        data_pd['Energy_shape'] = data_pd['Energy_shape'].astype('float')
        data_pd['No_molecule'] = data_pd['No_molecule'].astype('string')
        data_pd['Initial_Pos'] = data_pd['Initial_Pos'].astype('string')
        data_pd['speed'] = data_pd['speed'].astype('float')
        
        for property_name in property_name_arr:
            data_pd[property_name] = []
            data_pd[property_name] = data_pd[property_name].astype(object)
    
    
        data_pd_ind = 0 # index of dataset
        for molecule in molecule_arr:
            for koff_method in koff_method_arr:
                for Energy_shape in Energy_shape_arr:
                    for No_molecule in No_molecule_arr:
                        for speed in speed_arr:
                            root_path =  '../Data/' + molecule + '_data/Constant_Speed/'  + koff_method + '_Shape_' + str(Energy_shape) + '/'
                            # Find all subfolders in root_path
                            common_path_list = [ item for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item)) ]
                            
                            # Reading noise data
                            if (Noise_data == True):
                                common_path_list = [common_path_sel for common_path_sel in common_path_list if '_noise' in common_path_sel]
                            else:
                                common_path_list = [common_path_sel for common_path_sel in common_path_list if '_noise' not in common_path_sel]
                                
                            for common_path in common_path_list:
                                if (('NoMol_' + No_molecule) in common_path):
                                    
                                    # Find Initial position array and load to dataframe
                                    common_path_split = common_path.replace('NoMol_' + No_molecule, '').split('_')
                                    ini_pos_arr =  [int(s) for s in common_path_split if s.isdigit()]
                                    ini_pos_arr = str(ini_pos_arr)
                                    
                                    
                                    full_path = root_path + common_path + '/'
                                    
                                    xp_pd = pd.read_csv(full_path + 'xp_Speed_' + str(speed) + '.csv')
                                    Fwlc_pd = pd.read_csv(full_path + 'Fwlc_Speed_' + str(speed) + '.csv')
                                    time_pd = pd.read_csv(full_path + 'Time_arr_Speed_' + str(speed) + '.csv')
                                    ext_pd = pd.read_csv(full_path + 'ext_Speed_' + str(speed) + '.csv')
                                    work_pd = pd.read_csv(full_path + 'work_Speed_' + str(speed) + '.csv')
                                    prob_unfold_pd = pd.read_csv(full_path + 'prob_unfold_Speed_' + str(speed) + '.csv')
                                    rup_no_pd = pd.read_csv(full_path + 'rup_no_Speed_' + str(speed) + '.csv')
    
                                    
                                    # Choose no_data randomly without replacement
                                    no_data = no_data_arr[int(No_molecule)]
                                    
                                    no_domains = (rup_no_pd.max()).max()
                                    succ_no = rup_no_pd[rup_no_pd.iloc[:,-1] == int(no_domains)].index
                                    if (len(succ_no)>10):
                                        no_data_choice = np.random.choice(succ_no, min(no_data,len(succ_no)), replace = False)
                                    else:
                                        no_data_choice = np.random.choice(succ_no, min(no_data,10), replace = True)
    
                                    # Loading selected properies into dataset
                                    for no_data_ind in no_data_choice:
                                        
                                        # Built dataet
                                        data_pd.loc[data_pd_ind, 'molecule'] = molecule
                                        data_pd.loc[data_pd_ind, 'koff_method'] = koff_method
                                        data_pd.loc[data_pd_ind, 'Energy_shape'] = Energy_shape
                                        data_pd.loc[data_pd_ind, 'No_molecule'] = No_molecule
                                        data_pd.loc[data_pd_ind, 'speed'] = speed
                                        data_pd.loc[data_pd_ind, 'Initial_Pos'] = ini_pos_arr
                                        
                                        # Adding desired properies into dataset
                                        if ('xp' in property_name_arr):
                                            data_pd.at[data_pd_ind, 'xp'] = np.array(xp_pd.loc[no_data_ind,:]) # use 'at' to log arrays
                
                                        if ('Fwlc' in property_name_arr):
                                            Fwlc_load = np.array(Fwlc_pd.loc[no_data_ind,:])
                                            xp = np.array(xp_pd.loc[no_data_ind,:])
                                            data_pd.at[data_pd_ind, 'Fwlc_ori'] = Fwlc_load
                                            if(trimming ==True):
                                                Fwlc_load = self.trimming_data(Fwlc_load, xp, No_molecule, trim_with_time = True, trim_std_multiplier = 0.2, trim_percent = 0.1)
                                            else:
                                                Fwlc_load = Fwlc_load
                                            data_pd.at[data_pd_ind, 'Fwlc'] = Fwlc_load # use 'at' to log arrays
                                            
                                                    
                                        if ('time' in property_name_arr):
                                            data_pd.at[data_pd_ind, 'time'] = np.array(time_pd) # use 'at' to log arrays
                
                                        if ('ext' in property_name_arr):
                                            data_pd.at[data_pd_ind, 'ext'] = np.array(ext_pd.loc[no_data_ind,:]) # use 'at' to log arrays
                                            
                                        if ('work' in property_name_arr):
                                            data_pd.at[data_pd_ind, 'work'] = np.array(work_pd.loc[no_data_ind,:]) # use 'at' to log arrays
                                            
                                        if ('prob_unfold' in property_name_arr):
                                            data_pd.at[data_pd_ind, 'prob_unfold'] = np.array(prob_unfold_pd.loc[no_data_ind,:]) # use 'at' to log arrays
                                            
                                        if ('rup_no' in property_name_arr):
                                            data_pd.at[data_pd_ind, 'rup_no'] = np.array(rup_no_pd.loc[no_data_ind,:]) # use 'at' to log arrays
                                            
                                        data_pd_ind = data_pd_ind + 1
    
    
    
        # data_pd.to_csv(df_save_path + 'ML_data' + '.csv', index = None)
        data_pd.to_pickle(df_save_path + 'ML_data_' + molecule  + '.csv') # To preserve the exact structure of the DataFrame
        
        
        # Build reference data 
        data_pd_ref = pd.DataFrame() 
    
        
        if (molecule == 'Titin'):
            no_sample_arr = np.array([16,4,1]) 
        elif (molecule == 'UtrNR3') or (molecule == 'DysNR3_bact') or (molecule == 'UtrNR3_bact') :
            no_sample_arr = np.array([16,8,3]) 
        elif (molecule == 'FL_Dys') or (molecule == 'FL_mUtr'):
            no_sample_arr = np.array([28,6,1]) 
        ii = 0
        for No_mol in data_pd['No_molecule'].unique():
          data_pd_cur_mol =  data_pd.loc[data_pd['No_molecule'] == No_mol]
          no_sample = no_sample_arr[ii]
          ii = ii + 1
          for ini_pos in data_pd_cur_mol['Initial_Pos'].unique():
            data_pd_cur_mol_cur_ini_pos = data_pd_cur_mol.loc[data_pd_cur_mol['Initial_Pos']==ini_pos]
            data_selected = data_pd_cur_mol_cur_ini_pos.sample( n = no_sample )
            data_pd_ref = pd.concat([data_pd_ref, data_selected])
              
        data_pd_ref.to_pickle(df_save_path + 'ML_data_refer_' + molecule + '.csv') 
    
        return (data_pd, data_pd_ref)
    #--------------------------------------------------------------------------
    
    
    
    
    #--------------------------------------------------------------------------
    # Build experimental dataset for ML classificaiton
    
    # Choose properties to build dataset
    # molecule: Choose from ['Titin', 'UtrNR3', 'DysNR3_bact','UtrNR3_bact']
    # No_molecule_arr = np.array(['0', '1', '2'])  #No. of Molecule being attached to cantilver tips. Choose from: 1,2,3
    # no_data = 200 # No. of data to data under a specific situation, choose between [0,1000]
    # property_name_arr = np.array(['xp','Fwlc','time','file_name','label', 'Fwlc_ori']) # Data properties to read
    # trimming = True # Trim Fwlc to zero after detachment
    # df_save_path  # Saving path
    #--------------------------------------------------------------------------
    def build_exp_dataset_ML(self, molecule, No_molecule_arr = np.array(['0', '1', '2']), 
                             property_name_arr = np.array(['xp','Fwlc','time','file_name','label', 'Fwlc_ori']) ,
                             trimming = True, df_save_path = '../Data/ML_Dataset/', 
                             no_data_arr = np.array([200, 200, 200]),):
        
        # smoothing methods -- not so useful
        def move_mean_preprocess(data, winsz=13):
            data = pd.DataFrame(data).rolling(window=winsz).mean().dropna().values
            # data = pd.DataFrame(data).rolling(window=winsz).mean().dropna().values
            # data = pd.DataFrame(data).rolling(window=winsz).mean().dropna().values
            data = data.reshape(-1)
            return data

        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        
        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w

        # Fast Fourier Transform & Denoising
        def filter_signal(signal, threshold=1e8):
            fourier = np.fft.rfft(signal)
            frequencies = np.fft.rfftfreq(signal.size, d=20e-3/signal.size)
            fourier[frequencies > threshold] = 0
            return np.fft.irfft(fourier)


        molecule_arr = np.array([molecule]) 
        speed = 2000e-9 # dummy variable here
        speed_arr = np.array([speed]) # Constant pulling speeds
        

        # Initialize dataframe to save data
        data_pd = pd.DataFrame() 
        data_pd['molecule'] = []
        data_pd['No_molecule'] = []
        data_pd['speed'] = []
        
        data_pd['molecule'] = data_pd['molecule'].astype('string')
        data_pd['No_molecule'] = data_pd['No_molecule'].astype('string')
        data_pd['speed'] = data_pd['speed'].astype('float')
        
        for property_name in property_name_arr:
            data_pd[property_name] = []
            data_pd[property_name] = data_pd[property_name].astype(object)
        
        
        data_pd_ind = 0 # index of dataset
        for molecule in molecule_arr:
            for No_molecule in No_molecule_arr:
                for speed in speed_arr:
                    
                    # Read saved data
                    common_path = '../Data/' + molecule + '_data/Exp_ibw_data/'
                    if (molecule == 'Titin'):
                        if (No_molecule == '1'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataTitin_mol_1TitinI270.mat') # 2e-6 m/s
                        elif (No_molecule == '0'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataTitin_mol_0TitinI270.mat') # 2e-6 m/s
                        elif (No_molecule == '2'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataTitin_mol_2TitinI270.mat') # 2e-6 m/s
                        else:
                            print('No available dataset for current speed and No_molecule')
                            continue
                    elif (molecule == 'UtrNR3'):
                        if (No_molecule == '1'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataUtrNR3_mol_1UtrN_R3.mat') # 2e-6 m/s
                        elif (No_molecule == '0'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataUtrNR3_mol_0UtrN_R3.mat') # 2e-6 m/s
                        elif (No_molecule == '2'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataUtrNR3_mol_2UtrN_R3.mat') # 2e-6 m/s
                        else:
                            print('No available dataset for current speed and No_molecule')
                            continue
                        
                    elif (molecule == 'DysNR3_bact'):
                        if (No_molecule == '1'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataDysNR3_mol_1DysN-R3.mat') # 2e-6 m/s
                        elif (No_molecule == '0'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataDysNR3_mol_0DysN-R3.mat') # 2e-6 m/s
                        elif (No_molecule == '2'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataDysNR3_mol_2DysN-R3.mat') # 2e-6 m/s
                        else:
                            print('No available dataset for current speed and No_molecule')
                            continue
                        
                    elif (molecule == 'UtrNR3_bact'):
                        if (No_molecule == '1'):
                            Exp_data = scipy.io.loadmat(common_path + 'Expdatabact_UtrNR3_mol_1Bact_UtrNR3.mat') # 2e-6 m/s
                        elif (No_molecule == '0'):
                            Exp_data = scipy.io.loadmat(common_path + 'Expdatabact_UtrNR3_mol_0Bact_UtrNR3.mat') # 2e-6 m/s
                        elif (No_molecule == '2'):
                            Exp_data = scipy.io.loadmat(common_path + 'Expdatabact_UtrNR3_mol_2Bact_UtrNR3.mat') # 2e-6 m/s
                        else:
                            print('No available dataset for current speed and No_molecule')
                            continue
                    elif (molecule == 'FL_Dys'):
                        if (No_molecule == '1'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataFL_mDys_mol_1FL_mDys.mat') # 2e-6 m/s
                        elif (No_molecule == '0'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataFL_mDys_mol_0FL_mDys.mat') # 2e-6 m/s
                        elif (No_molecule == '2'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataFL_mDys_mol_2FL_mDys.mat') # 2e-6 m/s
                        else:
                            print('No available dataset for current speed and No_molecule')
                            continue
                        
                    elif (molecule == 'FL_mUtr'):
                        if (No_molecule == '1'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataFL_mUtr_mol_1FL_mUtr.mat') # 2e-6 m/s
                        elif (No_molecule == '0'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataFL_mUtr_mol_0FL_mUtr.mat') # 2e-6 m/s
                        elif (No_molecule == '2'):
                            Exp_data = scipy.io.loadmat(common_path + 'ExpdataFL_mUtr_mol_2FL_mUtr.mat') # 2e-6 m/s
                        else:
                            print('No available dataset for current speed and No_molecule')
                            continue
                            
                    Exp_data = Exp_data['Exp_data']
                    
                    # Finding largest size of data
                    exp_data_size = 0
                    exp_data_no = np.size(Exp_data,0)
                    for ii in range(exp_data_no):
                        dfl = Exp_data[ii,0]
                        cur_exp_data_size = np.size(dfl,0)
                        if(cur_exp_data_size > exp_data_size):
                            exp_data_size = cur_exp_data_size
                            
                    # Initialize dataframe to save data
                    xp_pd = pd.DataFrame(0, index = np.arange(exp_data_no), columns = np.arange(exp_data_size), dtype = float) # molecular extension 
                    Fu_pd = pd.DataFrame(0, index = np.arange(exp_data_no), columns = np.arange(exp_data_size), dtype = float) # Force
                    ext_pd = pd.DataFrame(0, index = np.arange(exp_data_no), columns = np.arange(exp_data_size), dtype = float) # extension
                    time_pd = pd.DataFrame(0, index = np.arange(exp_data_no), columns = np.arange(exp_data_size), dtype = float)  # time
                    file_name_pd = pd.DataFrame(0, index = np.arange(exp_data_no), columns = np.arange(1), dtype = 'string') # extension # file name
                    label_pd = pd.DataFrame(0, index = np.arange(exp_data_no), columns = np.arange(1), dtype = 'string') # extension# label
                    
                    # Organize data into dataframe
                    for ii in range(exp_data_no):
                        # molecular extension 
                        xp = Exp_data[ii,0]
                        xp_pd.loc[ii,0:len(xp)-1] = xp.reshape(-1)
                        # force
                        Fu = Exp_data[ii,1]
                        Fu_pd.loc[ii,0:len(Fu)-1] = Fu.reshape(-1)
                        
                        # extension
                        ext = Exp_data[ii,2]
                        ext_pd.loc[ii,0:len(ext)-1] = ext.reshape(-1)
                        # time
                        time = Exp_data[ii,3]
                        time_pd.loc[ii,0:len(time)-1] = time.reshape(-1)
                        # file name
                        file_name = Exp_data[ii,4]
                        file_name_pd.loc[ii,0] = file_name[0]
                        # label
                        label = Exp_data[ii,5]
                        label_pd.loc[ii,0] = str(label[0][0])
                        
                        
                    # Choose no_data randomly without replacement
                    succ_no = xp_pd.index
                    no_data = no_data_arr[int(No_molecule)]
                    no_data_choice = np.random.choice(succ_no, min(no_data,exp_data_no), replace = False)
                
                    # Loading selected properies into dataset
                    for no_data_ind in no_data_choice:
                        
                        # Built dataet
                        data_pd.loc[data_pd_ind, 'molecule'] = molecule
                        data_pd.loc[data_pd_ind, 'No_molecule'] = No_molecule
                        data_pd.loc[data_pd_ind, 'speed'] = speed
                        
                        # Adding desired properies into dataset
                        if ('xp' in property_name_arr):
                            data_pd.at[data_pd_ind, 'xp'] = np.array(xp_pd.loc[no_data_ind,:]) # use 'at' to log arrays
        
                        if ('Fwlc' in property_name_arr):
                            Fwlc_load = np.array(Fu_pd.loc[no_data_ind,:])
                            xp = np.array(xp_pd.loc[no_data_ind,:])
                            
                            data_pd.at[data_pd_ind, 'Fwlc_ori'] = Fwlc_load
                            
                            if(trimming == True):
                                Fwlc_load = self.trimming_data(Fwlc_load, xp, No_molecule, trim_percent = 0.1)
                                # Fwlc_load = self.trimming_data(Fwlc_load, xp, No_molecule, trim_with_time = True, trim_std_multiplier = 0.2, trim_percent = 0.1)
                            else:
                                Fwlc_load = Fwlc_load
                            # smoothing data here -- not so useful
                            # Fwlc_load = move_mean_preprocess(Fwlc_load, winsz=11)
                            # Fwlc_load = savgol_filter(Fwlc_load, 31, 16) # window size 51, polynomial order 3
                            # Fwlc_load = medfilt(Fwlc_load, kernel_size = 13)
                            # Fwlc_load = smooth(Fwlc_load,5)
                            data_pd.at[data_pd_ind, 'Fwlc'] = Fwlc_load # use 'at' to log arrays     

                        if ('time' in property_name_arr):
                            data_pd.at[data_pd_ind, 'time'] = np.array(time_pd.loc[no_data_ind,:]) # use 'at' to log arrays
        
                        if ('ext' in property_name_arr):
                            data_pd.at[data_pd_ind, 'ext'] = np.array(ext_pd.loc[no_data_ind,:]) # use 'at' to log arrays
                            
                        if ('file_name' in property_name_arr):
                            data_pd.at[data_pd_ind, 'file_name'] = np.array(file_name_pd.loc[no_data_ind,0]) # use 'at' to log arrays
                        
                        if ('label' in property_name_arr):
                            data_pd.at[data_pd_ind, 'label'] = np.array(label_pd.loc[no_data_ind,0]) # use 'at' to log arrays
                            
                        data_pd_ind = data_pd_ind + 1
        
        data_pd.to_pickle(df_save_path + 'ML_data_exp_' + molecule + '.csv') # To preserve the exact structure of the DataFrame
        
        return data_pd
    
    #--------------------------------------------------------------------------
    # Build Fu and xp from experimetnal data
    
    # molecule: Choose from ['Titin', 'UtrNR3', 'DysNR3_bact','UtrNR3_bact']
    # df_save_path  # Saving path
    #--------------------------------------------------------------------------
    def get_Fu_xp_exp(self, molecule, df_save_path):
        data_df_exp_cs = self.build_exp_dataset_ML(molecule = molecule, No_molecule_arr = np.array(['0', '1', '2']), 
                                 property_name_arr = np.array(['xp','Fwlc','time','file_name','label', 'Fwlc_ori']),
                                 trimming = True, df_save_path = df_save_path, 
                                 no_data_arr = np.array([1000, 1000, 1000]),)
        
        # find exp_data_size
        exp_data_size = 0
        for ii in data_df_exp_cs.index:
            # Need to padding for experimental data
            Fwlc = data_df_exp_cs['Fwlc'][ii]*1e12
            exp_data_size = np.max([np.size(Fwlc),exp_data_size])
        
        data_df = pd.DataFrame(columns = np.arange(0,exp_data_size))
        data_df_ind = 0
        data_df_label = pd.DataFrame([])
        for ii in data_df_exp_cs.index:  
            Fwlc = data_df_exp_cs['Fwlc'][ii]*1e12
            file_name = data_df_exp_cs['file_name'][ii]
            if (np.size(Fwlc)>2):
                padding_size = exp_data_size - np.size(Fwlc)
                Fwlc = np.pad(Fwlc, (0, padding_size), 'constant', constant_values=(0,0))
                
            data_df.loc[data_df_ind, :] = np.abs(Fwlc)
            
            data_df_label.loc[data_df_ind, 'Label'] = data_df_exp_cs['No_molecule'][ii]
            data_df_label.loc[data_df_ind, 'file_name'] = file_name
            data_df_ind = data_df_ind + 1
            
        data_df = pd.concat([data_df, data_df_label], axis = 1)
        
        Fu_data_df = data_df

        
        # find exp_data_size
        exp_data_size = 0
        for ii in data_df_exp_cs.index:
            # Need to padding for experimental data
            Fwlc = data_df_exp_cs['xp'][ii]*1e9
            exp_data_size = np.max([np.size(Fwlc),exp_data_size])
        
        data_df = pd.DataFrame(columns = np.arange(0,exp_data_size))
        data_df_ind = 0
        data_df_label = pd.DataFrame([])
        for ii in data_df_exp_cs.index:  
            Fwlc = data_df_exp_cs['xp'][ii]*1e9
            file_name = data_df_exp_cs['file_name'][ii]
            if (np.size(Fwlc)>2):
                padding_size = exp_data_size - np.size(Fwlc)
                Fwlc = np.pad(Fwlc, (0, padding_size), 'constant', constant_values=(0,0))
            data_df.loc[data_df_ind, :] = np.abs(Fwlc)
            data_df_label.loc[data_df_ind, 'Label'] = data_df_exp_cs['No_molecule'][ii]
            data_df_label.loc[data_df_ind, 'file_name'] = file_name
            data_df_ind = data_df_ind + 1
            
        data_df = pd.concat([data_df, data_df_label], axis = 1)
        xp_data_df = data_df
        
        return (Fu_data_df, xp_data_df)
    
    #--------------------------------------------------------------------------
    # Build Fu and xp from simluated data
    
    # molecule: Choose from ['Titin', 'UtrNR3', 'DysNR3_bact','UtrNR3_bact']
    # df_save_path  # Saving path
    #--------------------------------------------------------------------------
    def get_Fu_xp_sim(self, molecule, df_save_path):
        [data_pd,_]  = self.build_sim_dataset_ML(molecule, No_molecule_arr = np.array(['0', '1', '2']), 
                                 Noise_data = True, property_name_arr = np.array(['xp','Fwlc','time', 'Fwlc_ori']),
                                 trimming = True, df_save_path = df_save_path) 
        
        no_per_class = 200 # keep 200 of sim data per class (keeping same data amount as exp data)
        data_pd_sample = pd.DataFrame([])
        # balance sample of different initial positions
        for no_molecule in np.unique(data_pd['No_molecule']):
            data_pd_cur = data_pd[data_pd['No_molecule'] == no_molecule]
            sample_no = int((no_per_class/len(np.unique(data_pd_cur['Initial_Pos'])))+1)
            data_pd_cur = data_pd_cur.groupby('Initial_Pos').sample(n = sample_no)
            data_pd_sample = pd.concat([data_pd_sample, data_pd_cur], ignore_index=True)
        
        data_pd = data_pd_sample
        
        # find exp_data_size
        exp_data_size = 0
        for ii in data_pd.index:
            # Need to padding for experimental data
            Fwlc = data_pd['Fwlc'][ii]*1e12
            exp_data_size = np.max([np.size(Fwlc),exp_data_size])
        
        data_df = pd.DataFrame(columns = np.arange(0,exp_data_size))
        data_df_ind = 0
        data_df_label = pd.DataFrame([])
        file_name = 0 # dummy variable to distinguish different data
        for ii in data_pd.index:  
            Fwlc = data_pd['Fwlc'][ii]*1e12
            file_name = file_name + 1
            if (np.size(Fwlc)>2):
                padding_size = exp_data_size - np.size(Fwlc)
                Fwlc = np.pad(Fwlc, (0, padding_size), 'constant', constant_values=(0,0))

                    
            data_df.loc[data_df_ind, :] = np.abs(Fwlc)
            
            data_df_label.loc[data_df_ind, 'Label'] = data_pd['No_molecule'][ii]
            data_df_label.loc[data_df_ind, 'file_name'] = file_name
            data_df_ind = data_df_ind + 1
            
        data_df = pd.concat([data_df, data_df_label], axis = 1)
        
        Fu_data_df = data_df

        
        # find exp_data_size
        exp_data_size = 0
        for ii in data_pd.index:
            # Need to padding for experimental data
            Fwlc = data_pd['xp'][ii]*1e9
            exp_data_size = np.max([np.size(Fwlc),exp_data_size])
        
        data_df = pd.DataFrame(columns = np.arange(0,exp_data_size))
        data_df_ind = 0
        data_df_label = pd.DataFrame([])
        file_name = 0
        for ii in data_pd.index:  
            Fwlc = data_pd['xp'][ii]*1e9
            file_name = file_name+1
            if (np.size(Fwlc)>2):
                padding_size = exp_data_size - np.size(Fwlc)
                Fwlc = np.pad(Fwlc, (0, padding_size), 'constant', constant_values=(0,0))
            data_df.loc[data_df_ind, :] = np.abs(Fwlc)
            data_df_label.loc[data_df_ind, 'Label'] = data_pd['No_molecule'][ii]
            data_df_label.loc[data_df_ind, 'file_name'] = file_name
            data_df_ind = data_df_ind + 1
            
        data_df = pd.concat([data_df, data_df_label], axis = 1)
        xp_data_df = data_df
        
        return (Fu_data_df, xp_data_df)
    
    
    
    #--------------------------------------------------------------------------        
    # Resample and normalization of data
    # X_train, y_train: Data and label to process
    # sim_data: True if simulation data is used
    # resampling: True or False, perform resample or not
    # resample_size: length after resample
    # data_normalization: True or False, perform data minmax normalization or not. 
    #--------------------------------------------------------------------------  
    def resample_normalization_data(self, X_train, y_train, molecule, sim_data,
                                    resampling, resample_size, data_normalization):         
        
        # Get rid of extra zeros at the end
        if (sim_data == True):
            if (molecule != 'fewshot'):
              X_train_index = np.zeros((len(X_train),int(X_train.shape[1]-X_train.shape[1]/5)), dtype=bool)
              X_train_index = np.concatenate((X_train_index, X_train[:,-int(X_train.shape[1]/5+2):-1]==0), axis = 1)
              X_train[X_train_index] = 'nan'
        
        else:
            X_train[X_train==0] = 'nan'
            
        # Resample data
        if (resampling == True):
            X_train = to_time_series_dataset(X_train)
            X_train = TimeSeriesResampler(sz = resample_size).fit_transform(X_train)
        else:
            if len(X_train.shape) == 2:  # if univariate
                # add a dimension to make it multivariate with one dimension
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    
        # Normalize the data
        if(data_normalization == True):
            X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    
        return (X_train, y_train)
    #--------------------------------------------------------------------------     
    
    
    #--------------------------------------------------------------------------        
    # Get refernce data from simulation data
    # data_pd: simulation data
    # molecule: # Choose from 'Titin', 'UtrNR3', 'DysNR3_bact', 'fewshot'
    #--------------------------------------------------------------------------  
    def get_ref_data(self, data_pd, molecule):
        
        # Build reference data 
        data_pd_ref = pd.DataFrame() 
        
        if (molecule == 'Titin'):
            no_sample_arr = np.array([16,4,1]) 
        elif (molecule == 'fewshot'):
            no_sample_arr = np.array([10,10,10]) 
        elif (molecule == 'UtrNR3') or (molecule == 'DysNR3_bact'):
            no_sample_arr = np.array([16,8,3]) 
                
        ii = 0
        for No_mol in data_pd['No_molecule'].unique():
            data_pd_cur_mol =  data_pd.loc[data_pd['No_molecule'] == No_mol]
            no_sample = no_sample_arr[ii]
            ii = ii + 1
            for ini_pos in data_pd_cur_mol['Initial_Pos'].unique():
                data_pd_cur_mol_cur_ini_pos = data_pd_cur_mol.loc[data_pd_cur_mol['Initial_Pos']==ini_pos]
                data_selected = data_pd_cur_mol_cur_ini_pos.sample( n = no_sample )
                data_pd_ref = pd.concat([data_pd_ref, data_selected])
                
        return data_pd_ref
    
    
    
    #--------------------------------------------------------------------------        
    # Augment reference curves to input curve
    # X_train, y_train: Data and label to process
    # data_pd_ref: reference curve dataframe
    # no_refer: the number of reference curves
    #--------------------------------------------------------------------------  
    def add_refer_data(self, X_train, y_train, data_pd_ref, no_refer, resample_size, 
                       data_normalization):
        
        X_train_ref = np.zeros((X_train.shape[0], X_train.shape[1], no_refer))
        
        
        for ii in range((len(y_train))):
            X_data_reference_df = pd.DataFrame()
            X_data_reference_df = data_pd_ref.sample(n = no_refer, replace = False)
      
            cur_refer = 0
            for jj in X_data_reference_df.index:
                X_data_reference_data = X_data_reference_df['Fwlc'][jj]*1e12
                X_data_reference_data = to_time_series_dataset(X_data_reference_data)
                X_data_reference_data = TimeSeriesResampler(sz = resample_size).fit_transform(X_data_reference_data)
                # Adding normalization to reference data
                if(data_normalization == True):
                    X_data_reference_data = TimeSeriesScalerMinMax().fit_transform(X_data_reference_data)
    
                X_data_reference_data_load = X_train[ii,:,0] - X_data_reference_data[0,:,0]
                X_train_ref[ii,:,cur_refer] = X_data_reference_data_load
                cur_refer = cur_refer + 1

        X_train = np.concatenate((X_train,X_train_ref), axis = 2)
        return X_train
    #--------------------------------------------------------------------------     

    
    #--------------------------------------------------------------------------        
    # Get trace and pair data from (Fu,xp)
    
    # Fu_data_df and xp_data_df: data of (Fu, xp)
    # no_of_pairs: number of elements in pair data
    # resample_size: number of elements in trace data 
    #--------------------------------------------------------------------------  
    def get_trace_pair_data(self, Fu_data_df, xp_data_df, 
                            no_of_pairs, resample_size,
                            molecule, resampling = True):
        file_name_arr = np.unique(Fu_data_df['file_name'])
    
        Fu_train_exp =  np.zeros((len(file_name_arr), no_of_pairs))
        xp_train_exp =  np.zeros((len(file_name_arr), no_of_pairs))
    
        # save resampled traces
        Fu_train_exp_norm =  np.zeros((len(file_name_arr), resample_size))
        xp_train_exp_norm =  np.zeros((len(file_name_arr), resample_size))
    
        X_train_exp_ind = 0
        y_train_exp = []
        for file_name in file_name_arr:
            Fu_data_df_cur = Fu_data_df[Fu_data_df['file_name'] == file_name]
            xp_data_df_cur = xp_data_df[xp_data_df['file_name'] == file_name]
            
            y_train_exp_cur = int(np.unique(Fu_data_df_cur['Label'])[0])
            
            Fu_data_arr = np.array([])
            xp_data_arr = np.array([])
            for ii in Fu_data_df_cur.index:
                Fu_data_cur = np.array(Fu_data_df_cur.drop(['Label','file_name'], axis = 1).loc[ii])
    
                xp_data_cur = np.array(xp_data_df_cur.drop(['Label','file_name'], axis = 1).loc[ii])
                
                xp_data_cur = xp_data_cur[Fu_data_cur!=0]
                Fu_data_cur = Fu_data_cur[Fu_data_cur!=0]
                
    
                Fu_data_arr = np.concatenate((Fu_data_arr,Fu_data_cur))
                xp_data_arr = np.concatenate((xp_data_arr,xp_data_cur))
            
            
            # random choose pairs
            random_ind = np.random.randint(0, xp_data_arr.shape[0], size = no_of_pairs)
            random_ind = np.sort(random_ind)
            Fu_data_chose = Fu_data_arr[random_ind]
            xp_data_chose = xp_data_arr[random_ind]
            
            # normizalize and resample data
            [Fu_data_norm, _] = self.resample_normalization_data(Fu_data_arr, y_train = None, 
                                                                    molecule = molecule, sim_data = False,
                                                                    resampling = resampling, resample_size = resample_size, 
                                                                    data_normalization = True,)
            
            # normizalize and resample data
            [xp_data_norm, _] = self.resample_normalization_data(xp_data_arr, y_train = None, 
                                                                    molecule = molecule, sim_data = False,
                                                                    resampling = resampling, resample_size = resample_size, 
                                                                    data_normalization = True,)
 
            Fu_train_exp[X_train_exp_ind, :] = Fu_data_chose
            Fu_train_exp_norm[X_train_exp_ind, :] = Fu_data_norm.reshape(-1)
            xp_train_exp_norm[X_train_exp_ind, :] = xp_data_norm.reshape(-1)
            
            xp_train_exp[X_train_exp_ind, :] = xp_data_chose
            y_train_exp.append(y_train_exp_cur)
            X_train_exp_ind = X_train_exp_ind + 1
            
        y_train_exp = np.array(y_train_exp)
        
        return (Fu_train_exp, xp_train_exp, Fu_train_exp_norm, xp_train_exp_norm, y_train_exp)
    
    
    
    #--------------------------------------------------------------------------
    # Build fewshot dataset
    #--------------------------------------------------------------------------
    def build_fewshot_ML(self, No_molecule_arr = np.array(['0', '1', '2']), 
                             df_save_path = '../Data/ML_Dataset/',
                             ):
        
        def fewshot_preprocess(data, winsz=13):
            data = np.transpose(data)
            data = np.diff(data,axis=0)
            data = pd.DataFrame(data).rolling(window=winsz).mean().dropna().values
            # data = pd.DataFrame(data).rolling(window=winsz).mean().dropna().values
            # data = pd.DataFrame(data).rolling(window=winsz).mean().dropna().values
            data = np.transpose(data)
            return data
        
        # Build dataset from few-shot triplet paper
        common_path = '../Misc/JoshuaRWaite-AI_AFM-432d205/data/3class_matching/'
    
        data_pd_fs = pd.DataFrame() 
        data_pd_fs['No_molecule'] = []
        data_pd_fs['Training_stat'] = []# train or test or validation
        data_pd_fs['Fwlc'] = []
        data_pd_fs['Fwlc']  = data_pd_fs['Fwlc'].astype(object)
    
        data_pd_fs['Fwlc_ori'] = []
        data_pd_fs['Fwlc_ori']  = data_pd_fs['Fwlc_ori'].astype(object)
    
        train_arr = ['train', 'val', 'test']
    
        data_pd_ind = 0 # index of dataset
        for training_stat in train_arr:
            for No_molecule in No_molecule_arr:
                # Find all data inside folder
                data_path = common_path + training_stat + '/' + No_molecule + '/'
                data_txt_list = [item for item in os.listdir(data_path)]
            
                for data_path_cur in data_txt_list:
                    # Load dat
                    data_pd_fs.loc[data_pd_ind, 'No_molecule'] = No_molecule
                    data_pd_fs.loc[data_pd_ind, 'Training_stat'] = training_stat
                    
                    # Load Fwlc
                    Fwlc = np.expand_dims(np.loadtxt(data_path+data_path_cur,delimiter=','), axis=0) 
                    Fwlc = fewshot_preprocess(Fwlc)
                    Fwlc = Fwlc.reshape(-1)
                    data_pd_fs.at[data_pd_ind, 'Fwlc'] = Fwlc*1e-12 # change to N
                    data_pd_fs.at[data_pd_ind, 'Fwlc_ori'] = Fwlc*1e-12 # change to N
                    data_pd_ind = data_pd_ind + 1
                    
        data_pd_fs.to_pickle(df_save_path + 'ML_data_fewshot' + '.csv') # To preserve the exact structure of the DataFrame
    
        # Build reference data for fewshot 
    
        data_pd_fs_ref = pd.DataFrame() 
        no_sample_arr = np.array([10,10,10]) # To keep data balance
    
        ii = 0
        for No_mol in data_pd_fs['No_molecule'].unique():
          data_pd_cur_mol =  data_pd_fs.loc[data_pd_fs['No_molecule'] == No_mol]
          no_sample = no_sample_arr[ii]
          ii = ii + 1
          data_pd_cur_mol_cur_ini_pos = data_pd_cur_mol.loc[data_pd_cur_mol['Training_stat']=='train']
          data_selected = data_pd_cur_mol_cur_ini_pos.sample( n = no_sample )
          data_pd_fs_ref = pd.concat([data_pd_fs_ref, data_selected])
          
        data_pd_fs_ref.to_pickle(df_save_path + 'ML_data_refer_fewshot' + '.csv') 
        
        
        return (data_pd_fs, data_pd_fs_ref)
    


    #--------------------------------------------------------------------------
    
    
    
    
    