# =============================================================================
# Description
# =============================================================================

# Call script for constant speed Monte Carlo simulaitons
# =============================================================================

# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import sys
sys.path.append('../')
from APIs.MonteCarloSMFS import MonteCarloSMFS
from APIs.utils import utils
# =============================================================================


MCObj = MonteCarloSMFS()
UtObj = utils()

plot_fonts = UtObj.plot_fonts
plt.rcParams.update(plot_fonts)

color_set = UtObj.color_set
color_set_hist = UtObj.color_set_hist
marker_set = UtObj.marker_set
fmt_set = UtObj.fmt_set

#%%
# =============================================================================
# Arguments
# =============================================================================

# molecule: choose from 'UtrNR3', 'Titin' ,'DysNR3_bact', 'UtrNR3_bact'

# No_molecule: No. of Molecule being attached to cantilver tips: Choose from: 0, 1, 2, 3

# No_runs: the number of pulling simualtion

# initial_attachment: The number of folded domains at the begining, should be an array, for example [2,2] when No_molecule = 2

# vt_arr: the array of pulling speeds

# koff_method: Can choose 'DHS', to decide the unfolding event

# Energy_shape = 0 # 0 for cusp like, 1 for linear cubic for DHS model

# Save_All_Data: Save all data including but not limited to extension, time, xp, unfolding probabilities (For ML dataset)

# Noise_existence: Adding noise to simulated data

# N_one_exp: The number of data points per one experiment

# Detachment: Adding detachment forces

# xp_solver_method: the method to solve xp through F_{WLC}(xp) = kd, choose from 'fsolve', 'cubic_solve'

# =============================================================================

def run_sim(molecule, No_molecule, No_runs, initial_attachment, vt_arr, koff_method = 'DHS', Energy_shape = 0, 
            Adhesion = True, Save_All_Data = True, Noise_existence = True, N_one_exp = 300, 
            Detachment = True, xp_solver_method = 'fsolve'):
    
    #=====================================================
    # Simulation setups
    #=====================================================
        
    # The constants of the experiment
    kb = MCObj.kb  #Boltzman constant
    T = 300  # Temperature (K)
    kT = kb * T
    Kf = 10*10**-3 # Cantelever spring constant (N/m)
    
    
    
    # Define the protein characteristics
    if (molecule == 'Titin'):
        Lc0 =60e-9 # Contour Length (m)
        Lc = Lc0
        dLc = 28e-9 # Contour Length increment (m)
        P0 = 0.5*10**-9  # Persistence length (m)
        dP = 0.002*10**-9
    
        # Data Save path 
        df_save_path = '../Data/Titin_data/Constant_Speed/' + koff_method + '_Shape_' + str(Energy_shape) +'/'
        
    elif (molecule == 'UtrNR3'):
        Lc0 =40e-9 # Contour Length (m) 
        Lc = Lc0
        dLc = 28e-9  # Contour Length increment (m)
        P0 = 0.065*10**-9  # Persistence length (m) 
        dP = 0.0001*10**-9
        
        # Data Save path 
        df_save_path = '../Data/UTRNR3_data/Constant_Speed/' + koff_method + '_Shape_' + str(Energy_shape) +'/'
        
    elif (molecule == 'UtrNR3_bact'):
        Lc0 =40e-9 # Contour Length (m) 
        Lc = Lc0
        dLc = 35e-9  # Contour Length increment (m)
        P0 = 0.09*10**-9  # Persistence length (m) 
        dP = 0.002*10**-9

        # Data Save path 
        df_save_path = '../Data/UtrNR3_bact_data/Constant_Speed/' + koff_method + '_Shape_' + str(Energy_shape) +'/'

    elif (molecule == 'DysNR3_bact'):
        Lc0 =80e-9 # Contour Length (m) 
        Lc = Lc0
        dLc = 35e-9  # Contour Length increment (m)
        P0 = 0.06*10**-9  # Persistence length (m) 
        dP = 0.0002*10**-9
        
        # Data Save path 
        df_save_path = '../Data/DysNR3_bact_data/Constant_Speed/' + koff_method + '_Shape_' + str(Energy_shape) +'/'
                         
    else:
        print('The molecule is not available, please choose a valid one')       
        return None
    
    
    
    # choose No. of domains based on protein molecule
    if (molecule == 'Titin'):
        n = 8
    elif(molecule == 'UtrNR3') or (molecule == 'UtrNR3_bact') or (molecule == 'DysNR3_bact'):
        n = 4
    
    
    
    # Checking if the setup is correct
    # Finding No. of unfolded domains for each protein
    if (len(initial_attachment) != No_molecule) and (No_molecule > 0):
        print('Please make sure that initial attachment has the same length as No. of Molecules')
        return None
    else:
        n_mol = initial_attachment
        n = np.max(n_mol)
    
    if (No_molecule > 1) and (Detachment == False):
        print('Need to set detachment to be TRUE to run multi-molecule simulation')
        return None
    if (No_molecule > 1) and (xp_solver_method == 'cubic_solve' ):
        print('cubic_solve is not applicable to run multi-molecule simulation')
        return None 
    
    
    
    # Detachment, adhesion, noise setups
    if (molecule == 'Titin'):
        detachment_thres = 600 # pN Threshold to determine detachment
        detachment_prob_value = 0.05
    elif (molecule == 'UtrNR3') or (molecule == 'UtrNR3_bact') or (molecule == 'DysNR3_bact'):
        detachment_thres = 100 # pN Threshold to determine detachment
        detachment_prob_value = 0.1
        
    # Adhensive force 
    if (molecule == 'Titin'):
        ad_force_thres = 100 # pN
        ad_time_thres = 0.1 # controls when adhesive happens [0, 0.15]
    elif (molecule == 'UtrNR3') or (molecule == 'UtrNR3_bact') or (molecule == 'DysNR3_bact'):
        ad_force_thres = 100 # pN
        ad_time_thres = 0.05 # controls when adhesive happens [0, 0.15]
    
    # Noise scale of Fwlc force
    if (molecule == 'Titin'):
        Fwlc_noise_scale = 2e-12
    elif (molecule == 'UtrNR3') or (molecule == 'UtrNR3_bact') or (molecule == 'DysNR3_bact'):
        Fwlc_noise_scale = 3e-12 #4e-12
    
    if (No_molecule == 0):
        initial_attachment = np.array([0])


    
    # Save folders for ML dataset
    if (Save_All_Data == True):
        df_save_path = df_save_path + 'NoMol_' + str(No_molecule) + '_'
    
        for ii in range(len(initial_attachment)):
            df_save_path = df_save_path + 'IniPos_' + str(initial_attachment[ii]) + '_'
            
        if (Noise_existence == True):
            df_save_path = df_save_path + 'noise' + '/'
        else:
            df_save_path = df_save_path + '/'
        
    os.makedirs(df_save_path, exist_ok=True)
        

    fig_save_path = df_save_path + '/Plots/'
    os.makedirs(fig_save_path, exist_ok=True)
     
            

    # simulation methods, where [k0,dx,dG] is defined in MCObj
    [fingerprint, rate] = MCObj.choose_dhs(molecule, Energy_shape, kT, Lc)
        
        
    #=====================================================
    # Monte Carlo Constant Speed Experiments
    #=====================================================
    Nb = No_runs # Batch size 
    Niter = N_one_exp  # Number of time during one experiment
    N = Niter*Nb
    
    # Usage constants
    #Lc_arr = Lc + zeros(1,N)
    c = (kb*T)/(P0*Lc)
    b = 0
    No_event = np.sum(n_mol)
            
    # data for all unfolding force and loading rates
    Fwlc_cont_allvt_pd =  pd.DataFrame(0, index = (np.arange(Nb * No_event)), columns = vt_arr) #np.arange(len(vt_arr)))
    lr_unfold_allvt_pd =  pd.DataFrame(0, index = (np.arange(Nb * No_event)), columns = vt_arr) #np.arange(len(vt_arr)))
    
    # Avoid No_column = 0
    if (No_molecule == 0):
        No_column = 1
    else:
        No_column = No_molecule
    
    for vt_cnt in range(0,len(vt_arr),1):
        
        vt = vt_arr[vt_cnt]
        
        # Find the discretization time : total time to stretch the protein/ no. of samples
        if (Detachment == True):
            # Run a longer extension to have detachment event
            if (molecule == 'Titin'):
                dt0 = ((Lc0+dLc*(n+5))/vt)/Niter #1000  
            elif (molecule != 'Titin'):
                dt0 = ((Lc0+dLc*(n))/vt)/Niter #1000  
        else:
            dt0 = ((Lc0+dLc*n)/vt)/Niter #1000  
        dt = dt0
        
        
        if (No_molecule == 0):
            # Use large Lc and P to suppress Fwlc
            Lc = 100e-6*np.ones((Nb,No_column))
            P = 100e-6*np.ones((Nb,No_column))
        else:
            Lc = Lc0*np.ones((Nb,No_column))
            P = P0*np.ones((Nb,No_column))
        
        Fwlc_sep = np.zeros((Nb,No_column))
        rup_no_sep = np.ones((Nb,No_column))
            

        b = np.zeros((Nb,1)) # cantilver base to substrate distance
        rup_no = np.ones((Nb,1)) # the number of rupture event
        xp = np.zeros((Nb,1)) # molecular extension
        
        if (Detachment == True):
            r_detach_check = np.zeros((Nb,No_column), dtype = 'bool')
            
        Fwlc_cont = np.zeros((Nb,No_event))
        Fwlc_cont_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(No_event)+1)
        # Loading rate
        lr_unfold_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(No_event)+1)
        
        Fwlc_collection = []
        
        prob_unfold_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(Niter+1))
        rup_no_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(Niter+1))
        xp_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(Niter+1))
        Fwlc_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(Niter+1))
        work_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(Niter+1))
        ext_pd = pd.DataFrame(0, index = np.arange(Nb)+1, columns = np.arange(Niter+1)) # extension

        Fwlc_cont_arr = []
        lr_unfold_arr = []
        time_arr = [0]

        for j in range(0, N, Nb):
            
            #-- Step 1: Update b(m+1) --#
            cur_Niter = int(j//Nb + 1)
            #-- b = total extension = xp + xt = polymer extension + z-piezo displacement --#        
            
            dt = dt0#*np.random.rand()
        
            b = b + vt*dt
            # save all extension
            ext_pd.loc[:,cur_Niter] = b
            
            cur_time = time_arr[-1] + dt
            time_arr.append(cur_time)
            
            
            #-- Step 2: Solve for xp(m+1) --#
            if (xp_solver_method == 'cubic_solve'):
                xp_solver = MCObj.xp_solver_cubic(Kf,P,Lc,kb,T,b,Nb)
                xp = np.reshape(xp_solver,(Nb,1))
                
            elif (xp_solver_method == 'fsolve'):
                xp_solver = MCObj.xp_solver_fsolve(Lc,P,T,Kf,b,xp_initial=xp)
                xp = np.reshape(xp_solver,(Nb,1))
                
            else:
                print('Please provide a valid solver method!')
                
            # Save all xp
            xp_pd.loc[:,cur_Niter] = xp
            
            #-- Step 3: Solve for Fwlc --#
            c = (kb*T)/(P*Lc)
            a1 = (6*Lc*Lc)-(9*xp*Lc)+(4*xp*xp)
            a2 = 4*(Lc-xp)*(Lc-xp)
            Fwlc_sep = (c*xp*a1)/a2
            Fwlc_sep = np.reshape(Fwlc_sep,(Nb,No_column))
            Fwlc = np.sum(Fwlc_sep, axis = 1)
            Fwlc = np.reshape(Fwlc,(Nb,1))
            
            Fwlc = MCObj.add_noise(Fwlc, noise_scale = Fwlc_noise_scale, noise_loc = 0, Noise_existence = Noise_existence) # adding noise to Fwlc
                
            Fwlc_pd.loc[:,cur_Niter] = Fwlc
            
            
            #-- Step 4: Solve for Nue (Probility of rupture) --#

            prob_unfold_sep =  1-np.exp(-rate(Fwlc_sep,*fingerprint[1:])*dt)	
            prob_unfold_sep = (n_mol+1 - rup_no_sep) * prob_unfold_sep
            prob_unfold = np.sum(prob_unfold_sep, axis = 1)
            # Save all prob_unfold
            prob_unfold_pd.loc[:,cur_Niter] = prob_unfold
        
            #-- Step 5: Sampling with random variable --#
            r_sep = np.random.rand(Nb,No_column)
            r_check_sep = (r_sep <= prob_unfold_sep)
            r_check = np.sum(r_check_sep, axis = 1).astype(bool)
            r_check = np.reshape(r_check,(Nb,1))
    
            for lcl_rup_cnt in range(1,No_event+1,1):
                            
                for lcl_batch_cnt in range(1,Nb+1,1):
                    
                    if(r_check[lcl_batch_cnt-1]&(lcl_rup_cnt == rup_no[lcl_batch_cnt-1])):
                        Fwlc_cont_pd.loc[lcl_batch_cnt, lcl_rup_cnt] = Fwlc[lcl_batch_cnt-1][0]
                        # loading rate 
                        loading_rate = (Fwlc_pd.loc[lcl_batch_cnt,cur_Niter] - Fwlc_pd.loc[lcl_batch_cnt,cur_Niter-1] ) / dt
                        lr_unfold_pd.loc[lcl_batch_cnt, lcl_rup_cnt] = loading_rate
                        Fwlc_cont[lcl_batch_cnt-1, lcl_rup_cnt-1] = Fwlc[lcl_batch_cnt-1][0]
                        
                        # saving all force in a list
                        Fwlc_cont_arr.append( Fwlc[lcl_batch_cnt-1][0])
                        lr_unfold_arr.append(loading_rate)
    
            #-- Step 6: Update Lc,Lp if there was an unfolding --#
            rup_no_sep = rup_no_sep + r_check_sep
            rup_no = rup_no + np.reshape(np.sum(r_check_sep, axis = 1),(Nb,1))#+ r_check
    
            # Save all rup_no
            rup_no_pd.loc[:,cur_Niter] = rup_no
            
            Lc = Lc + np.multiply(r_check_sep, dLc) #Multiply arguments element-wise.
            P  = P + np.multiply(r_check_sep, dP)
            
            # Checking detachment
            if (Detachment == True):
                [Lc,P] = MCObj.detachment(rup_no_sep, n_mol, Fwlc_sep, Lc, P, 
                                          detachment_prob_value, detachment_thres*1e-12, detachment_method = 'variable')
                
        # Adding adhesive force
        if (Adhesion == True):
            for ii in Fwlc_pd.index:
                Fwlc_cur = np.array(Fwlc_pd.loc[ii,:])
                xp_cur = np.array(xp_pd.loc[ii,:])
                
                [adhen_force, xp_ad] = MCObj.adhesion_force(Fwlc_cur, xp_cur, ad_force_thres = ad_force_thres*1e-12, ad_xp_thres = 10e-9, ad_time_thres = ad_time_thres)
                Fwlc_pd.loc[ii,:] = Fwlc_cur + adhen_force
                xp_pd.loc[ii,:] = xp_ad
    
        #=====================================================
        # Saving all data
        #=====================================================
        # Saving all data into cvs: potential used for ML 
        if(Save_All_Data == True):
            
            prob_unfold_pd.to_csv(df_save_path + 'prob_unfold_Speed_' + str(vt) + '.csv', index = None)
            rup_no_pd.to_csv(df_save_path+ 'rup_no_Speed_' + str(vt) +'.csv', index = None)
            xp_pd.to_csv(df_save_path + 'xp_Speed_' + str(vt) +'.csv', index = None)
            Fwlc_pd.to_csv(df_save_path + 'Fwlc_Speed_' + str(vt) +'.csv', index = None)
            work_pd.to_csv(df_save_path + 'work_Speed_' + str(vt) +'.csv', index = None)
            ext_pd.to_csv(df_save_path + 'ext_Speed_' + str(vt) + '.csv', index = None)
            time_pd = pd.DataFrame(time_arr)
            time_pd.to_csv(df_save_path + 'Time_arr_Speed_' + str(vt) +'.csv', index = None)
        
        # save unfolding force and loading rate for diffferent velocity (Model dependent methods need this)
        Fwlc_cont_allvt_pd.loc[range(len(Fwlc_cont_arr)),vt_arr[vt_cnt]] = np.array(Fwlc_cont_arr)
        lr_unfold_allvt_pd.loc[range(len(Fwlc_cont_arr)),vt_arr[vt_cnt]] = np.array(lr_unfold_arr)
    
    
        #=====================================================
        # Diagonistics plots
        #=====================================================
        succ_no = rup_no_pd[rup_no_pd.loc[:,Niter] == int(No_event+1)].index
        no_inspects = np.random.choice(succ_no, 5)
        plt.figure() 
        for no_inspect in no_inspects:
            plt.plot(time_arr, Fwlc_pd.loc[no_inspect,:]*1e12)
        plt.xlabel('Time (s)')
        plt.ylabel('force (pN)')
        plt.savefig(fname = fig_save_path + koff_method + '_time_vs_force_vt_' + str(vt_arr[vt_cnt]) + '.png')
        plt.savefig(fname = fig_save_path + koff_method + '_time_vs_force_vt_' + str(vt_arr[vt_cnt]) + '.svg')

    plt.close("all") 
    
    return None

#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Building ML dataset: run simulation for different intial_attachment for multi_molecules
molecule = 'DysNR3_bact' # Choose from 'Titin' or 'UtrNR3', 'DysNR3_bact'
No_molecule = 1 # choose from [0,1,2,3,...]

if (molecule == 'Titin'):
    no_events = 8
    No_runs_arr = np.array([1200, 300, 100]) # The No_runs per each No_molecules
    vt_arr = np.array([1000e-9]) 
    koff_method = 'DHS'
    Energy_shape = 0
    Save_All_Data = True # Need to save all data for building ML dataset
    Noise_existence = True 
    Detachment = True
    xp_solver_method = 'fsolve'

if (molecule == 'UtrNR3') or (molecule == 'DysNR3_bact')  or (molecule == 'UtrNR3_bact'):
    no_events = 4
    No_runs_arr = np.array([1200, 300, 100])*2 # The No_runs per each No_molecules
    vt_arr = np.array([1000e-9]) 
    koff_method = 'DHS'
    Energy_shape = 0
    Save_All_Data = True # Need to save all data for building ML dataset
    Noise_existence = True 
    Detachment = True
    xp_solver_method = 'fsolve'

if (No_molecule == 0):
    initial_attachment = np.array([0])
    No_runs = No_runs_arr[No_molecule]
    print('We are currently runing No_molecule = ' + str(No_molecule) +' with intial attachment = ' + str(initial_attachment))
    run_sim(molecule, No_molecule, No_runs, initial_attachment, vt_arr, koff_method = koff_method, 
            Energy_shape = Energy_shape, Adhesion = True, Save_All_Data = Save_All_Data, 
            Noise_existence = Noise_existence, Detachment = Detachment, xp_solver_method = xp_solver_method)


if (No_molecule == 1):
    No_runs = No_runs_arr[No_molecule]
    for ii in range(1,no_events+1,1):
        initial_attachment = np.array([ii])
        print('We are currently runing No_molecule = ' + str(No_molecule) +' with intial attachment = ' + str(initial_attachment))
        run_sim(molecule, No_molecule, No_runs, initial_attachment, vt_arr, koff_method = koff_method, 
                Energy_shape = Energy_shape, Adhesion = True, Save_All_Data = Save_All_Data, 
                Noise_existence = Noise_existence, Detachment = Detachment, xp_solver_method = xp_solver_method)

if (No_molecule == 2):
    No_runs = No_runs_arr[No_molecule]
    for ii in range(1,no_events+1,1): 
        for jj in range(ii,no_events+1,1):
            initial_attachment = np.array([ii,jj])
            print('We are currently runing No_molecule = ' + str(No_molecule) +' with intial attachment = ' + str(initial_attachment))
            run_sim(molecule, No_molecule, No_runs, initial_attachment, vt_arr, koff_method = koff_method, 
                    Energy_shape = Energy_shape, Adhesion = True, Save_All_Data = Save_All_Data, 
                    Noise_existence = Noise_existence, Detachment = Detachment, xp_solver_method = xp_solver_method)

#%%


